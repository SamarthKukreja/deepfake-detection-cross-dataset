"""
train.py
========
Train EfficientNet-B4 for binary deepfake detection.

Key techniques:
  • Mixed precision (AMP)           — fits in 2 GB VRAM
  • Gradient accumulation (×4)      — effective batch size 64
  • Progressive unfreezing          — prevents catastrophic forgetting
  • Label smoothing                 — improves cross-dataset generalisation
  • Cosine annealing LR             — smooth decay
  • Early stopping on val AUC       — avoids wasted overnight time
  • WeightedRandomSampler           — guaranteed balanced batches

Usage
-----
    # Single experiment:
    python train.py --exp exp1

    # All experiments sequentially:
    python train.py --exp all

    # Custom paths:
    python train.py --exp exp1 --faces ./faces --output ./results
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

try:
    import timm
except ImportError:
    raise ImportError("Run: pip install timm")

from dataset import get_dataloader
from sklearn.metrics import roc_auc_score

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

EXP_NAMES = [
    "exp1_ffpp_to_ffpp",
    "exp2_ffpp_to_celebdf",
    "exp3_celebdf_to_ffpp",
    "exp4_mixed_to_dfd",
]

EXP_SHORT = {
    "exp1_ffpp_to_ffpp":    "Exp 1: FF++ → FF++",
    "exp2_ffpp_to_celebdf": "Exp 2: FF++ → CelebDF",
    "exp3_celebdf_to_ffpp": "Exp 3: CelebDF → FF++",
    "exp4_mixed_to_dfd":    "Exp 4: Mixed → DFD",
}

MODEL_NAME        = "efficientnet_b4"
MODEL_PRESETS = {
    # Main baseline
    "efficientnet_b4": "efficientnet_b4",
    # Classical deepfake/CNN baselines
    "xception": "xception",
    "resnet50": "resnet50",
    # Modern CNN / transformer comparisons
    "convnext_tiny": "convnext_tiny",
    "vit_base": "vit_base_patch16_224",
}
INPUT_SIZE        = 224    # resize faces to this (fits B4 in 2 GB with AMP)
BATCH_SIZE        = int(os.environ.get("BATCH_SIZE", 16))   # per-GPU batch (env-overridable)
ACCUM_STEPS       = int(os.environ.get("ACCUM_STEPS", 4))   # effective batch = BATCH_SIZE*ACCUM_STEPS
# Mixed precision. EfficientNet + fp16 overflows on some GPUs (e.g. MX450) and
# produces NaN loss. Set USE_AMP=0 to train in full fp32. Default: on.
USE_AMP           = os.environ.get("USE_AMP", "1") != "0"
MAX_EPOCHS        = 30
LR                = 1e-4
WEIGHT_DECAY      = 1e-4
ETA_MIN           = 1e-6   # cosine floor
EARLY_STOP_PAT    = 5      # epochs without val AUC improvement
LABEL_SMOOTH      = 0.05   # 0→0.05, 1→0.95
NUM_WORKERS       = 4
SEED              = 42

ABLATION_PRESETS = {
    "full": {
        "label_smooth": LABEL_SMOOTH,
        "progressive_unfreeze": True,
        "augment_train": True,
    },
    "no_label_smoothing": {
        "label_smooth": 0.0,
        "progressive_unfreeze": True,
        "augment_train": True,
    },
    "no_progressive_unfreeze": {
        "label_smooth": LABEL_SMOOTH,
        "progressive_unfreeze": False,
        "augment_train": True,
    },
    "no_train_augmentation": {
        "label_smooth": LABEL_SMOOTH,
        "progressive_unfreeze": True,
        "augment_train": False,
    },
}

# Progressive unfreezing schedule (epoch → #blocks to unfreeze from end)
UNFREEZE_SCHEDULE = {
    1:  0,   # head only
    6:  2,   # + last 2 stages
    11: 4,   # + last 4 stages
}


# ──────────────────────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set all local RNG seeds used by the training loop."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(device: torch.device, model_name: str = MODEL_NAME) -> nn.Module:
    """
    Load a pretrained timm model and replace its classifier with a binary head.

    Architecture:
        backbone (frozen initially) → GlobalAvgPool → Dropout(0.3) → Linear(1)

    The single output logit is passed through sigmoid during evaluation.
    BCEWithLogitsLoss handles the sigmoid during training (numerically stable).
    """
    try:
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=1,
            drop_rate=0.3,
        )
    except TypeError:
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=1,
        )
    # Freeze entire backbone — only head trains in epoch 1
    _freeze_backbone(model)
    return model.to(device)


def _freeze_backbone(model: nn.Module) -> None:
    """Freeze everything except the classifier head."""
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze common timm classifier/head modules plus EfficientNet's final norm.
    for name in ("classifier", "head", "fc", "conv_head", "bn2", "global_pool"):
        module = getattr(model, name, None)
        # Some timm models expose non-module attrs here (e.g. ViT's global_pool
        # is the string 'token'); only descend into actual modules.
        if isinstance(module, nn.Module):
            for param in module.parameters():
                param.requires_grad = True


def _stage_modules(model: nn.Module) -> List[nn.Module]:
    """Return backbone stages for common timm/CNN models."""
    if hasattr(model, "blocks"):
        return list(model.blocks.children())
    if hasattr(model, "stages"):
        return list(model.stages.children())
    stages = []
    for name in ("layer1", "layer2", "layer3", "layer4"):
        module = getattr(model, name, None)
        if module is not None:
            stages.append(module)
    return stages


def unfreeze_blocks(model: nn.Module, n_blocks: int) -> None:
    """
    Unfreeze the last *n_blocks* stages of model.blocks in-place.

    EfficientNet-B4 has 7 stages (model.blocks[0..6]).
    Progressive schedule:
        epoch  1-5  : n_blocks=0  → head only
        epoch  6-10 : n_blocks=2  → head + stages 5,6
        epoch 11+   : n_blocks=4  → head + stages 3,4,5,6
    """
    if n_blocks == 0:
        return
    stages = _stage_modules(model)
    for stage in stages[-n_blocks:]:
        for param in stage.parameters():
            param.requires_grad = True


def get_unfreeze_count(epoch: int) -> int:
    """Return how many blocks to unfreeze at *epoch* (1-indexed)."""
    count = 0
    for trigger_epoch, n in sorted(UNFREEZE_SCHEDULE.items()):
        if epoch >= trigger_epoch:
            count = n
    return count


# ──────────────────────────────────────────────────────────────────────────────
# LOSS WITH LABEL SMOOTHING
# ──────────────────────────────────────────────────────────────────────────────

def smooth_labels(labels: torch.Tensor, smoothing: float = LABEL_SMOOTH) -> torch.Tensor:
    """
    Apply label smoothing: 0 → smoothing, 1 → 1 - smoothing.

    Prevents the model from becoming overconfident, which hurts
    cross-dataset generalisation significantly.
    """
    return labels * (1.0 - smoothing) + smoothing * 0.5


def resolve_model_name(model: str, model_preset: str) -> str:
    """Resolve a CLI model or named preset into a timm model name."""
    if model_preset != "custom":
        return MODEL_PRESETS[model_preset]
    return model


def run_dir_name(exp_name: str, model_name: str, ablation: str) -> str:
    """Build a result directory name that keeps model/ablation runs separate."""
    if model_name == MODEL_NAME and ablation == "full":
        return exp_name
    safe_model = model_name.replace("/", "_")
    return f"{exp_name}__{safe_model}__{ablation}"


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN / VALIDATE ONE EPOCH
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    label_smooth: float = LABEL_SMOOTH,
) -> Tuple[float, float]:
    """
    Run one training epoch with AMP and gradient accumulation.

    Returns
    -------
    (avg_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"  Train E{epoch:02d}", leave=False, unit="batch")

    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        smooth = smooth_labels(labels, label_smooth) if label_smooth > 0 else labels

        with autocast(enabled=USE_AMP):
            logits = model(images).squeeze(1)
            loss   = criterion(logits, smooth) / ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * ACCUM_STEPS
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{total_loss/(step+1):.4f}",
                         acc=f"{correct/total:.3f}")

    return total_loss / len(loader), correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Run validation.

    Returns
    -------
    (avg_loss, accuracy, auc_roc)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels: List[float] = []
    all_scores: List[float] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=USE_AMP):
            logits = model(images).squeeze(1)
            loss   = criterion(logits, labels)

        total_loss += loss.item()
        scores = torch.sigmoid(logits)
        preds  = (scores > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_scores.extend(scores.cpu().tolist())

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except ValueError:
        auc = 0.5  # only one class in batch (edge case)

    return avg_loss, accuracy, auc


# ──────────────────────────────────────────────────────────────────────────────
# FULL TRAINING LOOP — ONE EXPERIMENT
# ──────────────────────────────────────────────────────────────────────────────

def train_experiment(
    exp_name: str,
    faces_root: Path,
    output_root: Path,
    device: torch.device,
    model_name: str = MODEL_NAME,
    seed: int = SEED,
    ablation: str = "full",
    label_smooth: float = LABEL_SMOOTH,
    progressive_unfreeze: bool = True,
    augment_train: bool = True,
) -> Dict:
    """
    Train EfficientNet-B4 for one experiment and save the best checkpoint.

    Parameters
    ----------
    exp_name    : e.g. 'exp1_ffpp_to_ffpp'
    faces_root  : path to /faces/ directory
    output_root : path to /results/ directory
    device      : torch device

    Returns
    -------
    dict with training history and best metrics
    """
    out_dir = output_root / run_dir_name(exp_name, model_name, ablation)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {EXP_SHORT[exp_name]}")
    print(f"{'='*60}")

    # ── Dataloaders ─────────────────────────────────────────────
    set_seed(seed)
    train_loader = get_dataloader(faces_root, exp_name, "train",
                                  BATCH_SIZE, INPUT_SIZE, NUM_WORKERS,
                                  use_sampler=True,
                                  augment_train=augment_train)
    try:
        val_loader = get_dataloader(faces_root, exp_name, "val",
                                    BATCH_SIZE, INPUT_SIZE, NUM_WORKERS,
                                    use_sampler=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exp_name} has no validation faces. Regenerate train/val/test "
            "splits and run face extraction for the val split. Training "
            "intentionally refuses to use test for early stopping."
        ) from exc

    n_real, n_fake = train_loader.dataset.class_counts()
    print(f"  Train : {len(train_loader.dataset):,} faces  "
          f"(real={n_real:,}, fake={n_fake:,})")
    print(f"  Val   : {len(val_loader.dataset):,} faces")

    # ── Model, optimiser, scheduler ─────────────────────────────
    model     = build_model(device, model_name)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=ETA_MIN)
    scaler    = GradScaler(enabled=USE_AMP)

    # ── Training loop ────────────────────────────────────────────
    best_auc      = 0.0
    best_epoch    = 0
    no_improve    = 0
    history: Dict = {"train_loss": [], "train_acc": [],
                     "val_loss":   [], "val_acc":   [], "val_auc": []}
    ckpt_path = out_dir / "best_model.pth"

    if not progressive_unfreeze:
        for param in model.parameters():
            param.requires_grad = True
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, MAX_EPOCHS + 1):
        # Progressive unfreezing
        n_unfreeze = get_unfreeze_count(epoch) if progressive_unfreeze else -1
        if progressive_unfreeze:
            _freeze_backbone(model)          # reset
            unfreeze_blocks(model, n_unfreeze)

        # Rebuild optimiser param groups when unfreezing changes
        if progressive_unfreeze and epoch in UNFREEZE_SCHEDULE:
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR * (0.5 ** (n_unfreeze // 2)),  # mildly lower LR as more layers unfreeze (never below ~2.5e-5)
                weight_decay=WEIGHT_DECAY,
            )
            scheduler = CosineAnnealingLR(
                optimizer, T_max=MAX_EPOCHS - epoch + 1, eta_min=ETA_MIN
            )

        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device, epoch,
            label_smooth=label_smooth,
        )
        vl_loss, vl_acc, vl_auc = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["val_auc"].append(vl_auc)

        print(
            f"  E{epoch:02d}/{MAX_EPOCHS}  "
            f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}  "
            f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.3f}  "
            f"val_auc={vl_auc:.4f}  "
            f"[{elapsed:.0f}s]"
            + ("  ← best" if vl_auc > best_auc else "")
        )

        if vl_auc > best_auc:
            best_auc   = vl_auc
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_auc":    vl_auc,
                "val_acc":    vl_acc,
                "exp_name":   exp_name,
                "model_name":  model_name,
                "seed":        seed,
                "ablation":    ablation,
            }, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PAT:
                print(f"\n  Early stopping — no val AUC improvement "
                      f"for {EARLY_STOP_PAT} epochs.")
                break

    print(f"\n  Best val AUC = {best_auc:.4f}  (epoch {best_epoch})")
    print(f"  Checkpoint   → {ckpt_path}")

    # Save training history
    result = {
        "exp_name":    exp_name,
        "model_name":  model_name,
        "ablation":    ablation,
        "best_epoch":  best_epoch,
        "best_val_auc": best_auc,
        "history":     history,
        "config": {
            "input_size":   INPUT_SIZE,
            "batch_size":   BATCH_SIZE,
            "accum_steps":  ACCUM_STEPS,
            "lr":           LR,
            "max_epochs":   MAX_EPOCHS,
            "label_smooth": label_smooth,
            "model_name":   model_name,
            "seed":         seed,
            "ablation":     ablation,
            "progressive_unfreeze": progressive_unfreeze,
            "augment_train": augment_train,
        },
    }
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B4 deepfake detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--exp",
        choices=["exp1", "exp2", "exp3", "exp4", "all"],
        default="all",
    )
    parser.add_argument("--faces",  type=Path, default=Path("../faces"))
    parser.add_argument("--output", type=Path, default=Path("../results"))
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="Custom timm model name used only with --model_preset custom.",
    )
    parser.add_argument(
        "--model_preset",
        choices=list(MODEL_PRESETS.keys()) + ["custom"],
        default="efficientnet_b4",
        help="Named research baseline preset.",
    )
    parser.add_argument(
        "--ablation",
        choices=list(ABLATION_PRESETS.keys()),
        default="full",
        help="Training ablation preset.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    if args.exp == "all":
        exps = EXP_NAMES
    else:
        idx = int(args.exp[-1]) - 1
        exps = [EXP_NAMES[idx]]

    all_results = []
    model_name = resolve_model_name(args.model, args.model_preset)
    ablation_cfg = ABLATION_PRESETS[args.ablation]
    for exp_name in exps:
        result = train_experiment(
            exp_name=exp_name,
            faces_root=args.faces,
            output_root=args.output,
            device=device,
            model_name=model_name,
            seed=args.seed,
            ablation=args.ablation,
            label_smooth=ablation_cfg["label_smooth"],
            progressive_unfreeze=ablation_cfg["progressive_unfreeze"],
            augment_train=ablation_cfg["augment_train"],
        )
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("  Training Complete")
    print(f"{'='*60}")
    print(f"  {'Experiment':<30} {'Best AUC':>10}  {'Epoch':>6}")
    print(f"  {'-'*48}")
    for r in all_results:
        print(f"  {EXP_SHORT[r['exp_name']]:<30} "
              f"{r['best_val_auc']:>10.4f}  "
              f"{r['best_epoch']:>6}")


if __name__ == "__main__":
    main()
