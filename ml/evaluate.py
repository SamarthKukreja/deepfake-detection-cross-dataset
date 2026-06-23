"""
evaluate.py
===========
Evaluate a trained EfficientNet-B4 deepfake detector on the test set.

Produces (per experiment):
  • metrics.json          — AUC, Accuracy, F1, EER
  • plots/roc_curve.png   — ROC curve
  • plots/confusion_matrix.png
  • plots/gradcam_samples.png — Grad-CAM on 4 sample images

Usage
-----
    # Evaluate one experiment:
    python evaluate.py --exp exp1

    # Evaluate all:
    python evaluate.py --exp all

    # Custom paths:
    python evaluate.py --exp all --faces ./faces --results ./results
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.cuda.amp import autocast
from torchvision import transforms
from tqdm import tqdm

import os
# Match train.py: fp16 autocast overflows on some GPUs (e.g. MX450). Set USE_AMP=0 for fp32.
USE_AMP = os.environ.get("USE_AMP", "1") != "0"

try:
    import timm
except ImportError:
    raise ImportError("Run: pip install timm")

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("Warning: pytorch-grad-cam not installed. Skipping Grad-CAM.")
    print("         Run: pip install grad-cam")

from dataset import DeepfakeDataset, get_dataloader, get_tta_transforms

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG (must match train.py)
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

INPUT_SIZE   = 224
MODEL_NAME   = "efficientnet_b4"
MODEL_PRESETS = {
    "efficientnet_b4": "efficientnet_b4",
    "xception": "xception",
    "resnet50": "resnet50",
    "convnext_tiny": "convnext_tiny",
    "vit_base": "vit_base_patch16_224",
}
ABLATION_PRESETS = (
    "full",
    "no_label_smoothing",
    "no_progressive_unfreeze",
    "no_train_augmentation",
    "no_tta",
)
BATCH_SIZE   = 32    # can be larger at test time (no gradients)
NUM_WORKERS  = 4
TTA_N        = 5     # test-time augmentation variants
GRADCAM_SAMPLES = 4  # 2 real + 2 fake for Grad-CAM figure
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def resolve_model_name(model: str, model_preset: str) -> str:
    if model_preset != "custom":
        return MODEL_PRESETS[model_preset]
    return model


def run_dir_name(exp_name: str, model_name: str, ablation: str) -> str:
    if model_name == MODEL_NAME and ablation == "full":
        return exp_name
    return f"{exp_name}__{model_name.replace('/', '_')}__{ablation}"


def checkpoint_dir_name(exp_name: str, model_name: str, ablation: str) -> str:
    """Evaluation-only ablations reuse the full-training checkpoint."""
    train_ablation = "full" if ablation == "no_tta" or ablation.startswith("agg_") else ablation
    return run_dir_name(exp_name, model_name, train_ablation)


def eval_ablation_name(ablation: str, aggregation: str) -> str:
    """Name evaluation-only result variants without overwriting full metrics."""
    if ablation == "full" and aggregation != "mean":
        return f"agg_{aggregation}"
    return ablation


# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """
    Load EfficientNet-B4 from a saved checkpoint.

    Parameters
    ----------
    checkpoint_path : path to best_model.pth saved by train.py
    device          : torch device

    Returns
    -------
    model in eval mode, on *device*
    """
    # weights_only defaults to True in PyTorch 2.6+, which rejects the NumPy
    # scalars saved in the checkpoint; the checkpoint is our own/trusted.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", MODEL_NAME)
    try:
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=1,
            drop_rate=0.3,
        )
    except TypeError:
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=1,
        )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    model.model_name = model_name
    print(f"  Loaded checkpoint (epoch {ckpt['epoch']}, "
          f"val AUC={ckpt['val_auc']:.4f}, model={model_name})")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# TEST-TIME AUGMENTATION INFERENCE
# ──────────────────────────────────────────────────────────────────────────────

def tta_predict(
    model: nn.Module,
    loader,
    device: torch.device,
    n: int = TTA_N,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Run TTA inference: average sigmoid scores over *n* augmented views.

    Parameters
    ----------
    model  : trained model in eval mode
    loader : DataLoader (test split, no shuffle)
    device : torch device
    n      : number of TTA transforms

    Returns
    -------
    y_true  : (N,) integer labels
    y_score : (N,) averaged sigmoid probabilities
    paths   : source face image paths, same order
    """
    # Build *n* transform pipelines. n=1 is the no-TTA ablation.
    tta_tfms = get_tta_transforms(INPUT_SIZE, max(1, n))

    # Collect raw file paths + labels from the dataset
    dataset = loader.dataset
    all_paths  = [str(p) for p, _ in dataset.samples]
    all_labels = [lbl for _, lbl in dataset.samples]

    all_scores = np.zeros(len(all_paths), dtype=np.float32)

    for tfm in tta_tfms:
        scores_this_aug: List[float] = []
        # Batch manually to avoid rebuilding DataLoader each time
        for i in range(0, len(all_paths), BATCH_SIZE):
            batch_paths = all_paths[i : i + BATCH_SIZE]
            imgs = torch.stack([
                tfm(Image.open(p).convert("RGB")) for p in batch_paths
            ]).to(device, non_blocking=True)

            with torch.no_grad(), autocast(enabled=USE_AMP):
                logits = model(imgs).squeeze(1)
                scores = torch.sigmoid(logits).cpu().tolist()
            scores_this_aug.extend(scores)

        all_scores += np.array(scores_this_aug, dtype=np.float32)

    all_scores /= n  # average over augmentations
    return np.array(all_labels, dtype=int), all_scores, all_paths


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_eer(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    Compute Equal Error Rate (EER) — the threshold where FPR = FNR.

    EER is the standard security/biometrics metric and expected in
    deepfake detection papers.
    """
    fnr = 1.0 - tpr
    # Find index where |FPR - FNR| is minimised
    idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer)


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """
    Compute all paper-ready classification metrics.

    Returns
    -------
    dict with: auc, accuracy, f1, eer
    """
    y_pred = (y_score >= threshold).astype(int)
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = float(roc_auc_score(y_true, y_score))
        eer = compute_eer(fpr, tpr)
    except ValueError:
        auc = 0.5
        eer = 0.5
    return {
        "auc":      auc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1":       float(f1_score(y_true, y_pred, zero_division=0)),
        "eer":      eer,
    }


def _metadata_keys(path: str) -> List[str]:
    p = Path(path)
    keys = [str(p), str(p.resolve())]
    try:
        keys.append(str(p.as_posix()))
    except Exception:
        pass
    return keys


def load_face_metadata(faces_root: Path) -> Dict[str, Dict]:
    """Load face_metadata.jsonl if extraction generated it."""
    metadata_path = faces_root / "face_metadata.jsonl"
    meta: Dict[str, Dict] = {}
    if not metadata_path.exists():
        return meta
    with open(metadata_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            face_path = rec.get("face_path")
            if not face_path:
                continue
            for key in _metadata_keys(face_path):
                meta[key] = rec
    return meta


def fallback_metadata(face_path: str, label: int) -> Dict:
    """Infer enough metadata for video-level grouping from the face filename."""
    p = Path(face_path)
    stem = p.stem
    marker = "_frame"
    video_id = stem.split(marker, 1)[0] if marker in stem else stem
    return {
        "face_path": face_path,
        "video_path": "",
        "video_id": video_id,
        "dataset": "unknown",
        "experiment": p.parents[2].name if len(p.parents) > 2 else "",
        "split": p.parents[1].name if len(p.parents) > 1 else "",
        "label": "fake" if label == 1 else "real",
    }


def record_for_path(face_path: str, label: int, metadata: Dict[str, Dict]) -> Dict:
    for key in _metadata_keys(face_path):
        if key in metadata:
            return metadata[key]
    return fallback_metadata(face_path, label)


def aggregate_video_predictions(
    paths: List[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
    metadata: Dict[str, Dict],
    method: str = "mean",
    top_k: int = 5,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Aggregate face scores into one score per source video."""
    grouped: Dict[str, Dict] = {}
    for path, label, score in zip(paths, y_true.tolist(), y_score.tolist()):
        rec = record_for_path(path, int(label), metadata)
        video_key = rec.get("video_path") or rec.get("video_id") or Path(path).stem
        item = grouped.setdefault(video_key, {
            "video_key": video_key,
            "video_id": rec.get("video_id", Path(path).stem),
            "video_path": rec.get("video_path", ""),
            "dataset": rec.get("dataset", "unknown"),
            "label": int(label),
            "scores": [],
            "n_faces": 0,
        })
        item["scores"].append(float(score))
        item["n_faces"] += 1

    rows: List[Dict] = []
    labels: List[int] = []
    scores: List[float] = []
    for item in grouped.values():
        vals = np.array(item["scores"], dtype=np.float32)
        if method == "max":
            video_score = float(vals.max())
        elif method == "median":
            video_score = float(np.median(vals))
        elif method == "topk":
            k = min(top_k, len(vals))
            video_score = float(np.sort(vals)[-k:].mean())
        else:
            video_score = float(vals.mean())
        row = {
            "video_key": item["video_key"],
            "video_id": item["video_id"],
            "video_path": item["video_path"],
            "dataset": item["dataset"],
            "label": item["label"],
            "score": video_score,
            "n_faces": item["n_faces"],
        }
        rows.append(row)
        labels.append(item["label"])
        scores.append(video_score)

    return np.array(labels, dtype=int), np.array(scores, dtype=np.float32), rows


def save_failure_cases(rows: List[Dict], save_path: Path, threshold: float = 0.5) -> None:
    """Save video-level false positives/negatives and low-confidence cases."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    enriched = []
    for row in rows:
        pred = int(row["score"] >= threshold)
        confidence = row["score"] if pred == 1 else 1.0 - row["score"]
        enriched.append({**row, "prediction": pred, "confidence": confidence,
                         "is_error": pred != int(row["label"])})
    errors = [r for r in enriched if r["is_error"]]
    uncertain = sorted(enriched, key=lambda r: abs(r["score"] - threshold))[:20]
    selected = errors + [r for r in uncertain if r not in errors]
    fields = [
        "video_id", "video_path", "dataset", "label", "prediction",
        "score", "confidence", "n_faces", "is_error",
    ]
    with open(save_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({k: row.get(k, "") for k in fields})


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> Dict:
    """Bootstrap 95 percent confidence intervals over videos."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples: Dict[str, List[float]] = {
        "auc": [],
        "accuracy": [],
        "f1": [],
        "eer": [],
    }
    if n == 0:
        return {}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        ys = y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        m = compute_metrics(yt, ys)
        for key in samples:
            samples[key].append(m[key])

    ci = {}
    for key, vals in samples.items():
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float32)
        ci[key] = {
            "mean": float(arr.mean()),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
        }
    return ci


# ──────────────────────────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    exp_name: str,
    save_path: Path,
) -> None:
    """Save ROC curve figure with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, color="#4F46E5",
            label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="#9CA3AF", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC Curve — {EXP_SHORT[exp_name]}", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"    ROC curve  → {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_score: np.ndarray,
    exp_name: str,
    save_path: Path,
) -> None:
    """Save confusion matrix heatmap."""
    try:
        import seaborn as sns
    except ImportError:
        sns = None

    y_pred = (y_score >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(4, 4))
    if sns:
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"],
            ax=ax,
        )
    else:
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=14)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Real", "Fake"])
        ax.set_yticklabels(["Real", "Fake"])
        plt.colorbar(im, ax=ax)

    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"Confusion Matrix — {EXP_SHORT[exp_name]}", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"    Confusion  → {save_path}")


def plot_training_curves(history_path: Path, save_path: Path) -> None:
    """
    Load training_history.json from train.py and plot loss + AUC curves.

    Called during evaluation so both plots live in the same results folder.
    """
    if not history_path.exists():
        return

    with open(history_path) as f:
        data = json.load(f)

    h   = data["history"]
    eps = range(1, len(h["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Loss
    ax1.plot(eps, h["train_loss"], label="Train", color="#4F46E5")
    ax1.plot(eps, h["val_loss"],   label="Val",   color="#F59E0B")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()

    # AUC
    ax2.plot(eps, h["val_auc"], color="#10B981", label="Val AUC")
    ax2.axhline(y=max(h["val_auc"]), color="#6B7280", linestyle="--",
                label=f"Best = {max(h['val_auc']):.4f}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC-ROC")
    ax2.set_title("Validation AUC")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"    Loss curve → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# GRAD-CAM
# ──────────────────────────────────────────────────────────────────────────────

def run_gradcam(
    model: nn.Module,
    faces_root: Path,
    exp_name: str,
    save_path: Path,
    device: torch.device,
    n_samples: int = GRADCAM_SAMPLES,
) -> None:
    """
    Generate Grad-CAM heatmaps for *n_samples* test images (real + fake).

    Saves a figure with two rows:
        Row 1: original face crops
        Row 2: Grad-CAM heatmap overlay

    This is the most visually impactful figure for a Masters application.
    """
    if not GRADCAM_AVAILABLE:
        return
    if "vit" in getattr(model, "model_name", "").lower():
        print("    Grad-CAM   skipped for ViT-style model")
        return

    # Target layer: use common timm/CNN stage names when available.
    if hasattr(model, "blocks"):
        target_layers = [list(model.blocks.children())[-1]]
    elif hasattr(model, "stages"):
        target_layers = [list(model.stages.children())[-1]]
    elif hasattr(model, "layer4"):
        target_layers = [model.layer4]
    else:
        print("    Grad-CAM   skipped (no supported target layer)")
        return
    cam = GradCAM(model=model, target_layers=target_layers)

    test_dir = faces_root / exp_name / "test"
    tfm_tensor = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    tfm_display = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ])

    # Collect samples: half real, half fake
    half = n_samples // 2
    samples: List[Tuple[Path, int, str]] = []
    for label_name, label_idx in [("real", 0), ("fake", 1)]:
        paths = sorted((test_dir / label_name).glob("*.jpg"))
        chosen = random.sample(paths, min(half, len(paths)))
        samples.extend([(p, label_idx, label_name) for p in chosen])

    if not samples:
        return

    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    fig.suptitle(f"Grad-CAM — {EXP_SHORT[exp_name]}", fontsize=13)

    for col, (img_path, label, label_name) in enumerate(samples):
        pil  = Image.open(img_path).convert("RGB")
        inp  = tfm_tensor(pil).unsqueeze(0).to(device)
        disp = tfm_display(pil).permute(1, 2, 0).numpy()
        disp = np.clip(disp, 0, 1).astype(np.float32)

        # Grad-CAM
        grayscale_cam = cam(input_tensor=inp)[0]
        overlay = show_cam_on_image(disp, grayscale_cam, use_rgb=True)

        # Prediction label
        with torch.no_grad(), autocast(enabled=USE_AMP):
            score = torch.sigmoid(model(inp).squeeze()).item()
        pred = "Fake" if score > 0.5 else "Real"
        conf = score if pred == "Fake" else 1 - score

        # Original image
        axes[0, col].imshow(disp)
        axes[0, col].set_title(
            f"GT: {label_name.capitalize()}", fontsize=9
        )
        axes[0, col].axis("off")

        # Grad-CAM overlay
        color = "#EF4444" if pred == "Fake" else "#10B981"
        axes[1, col].imshow(overlay)
        axes[1, col].set_title(
            f"Pred: {pred} ({conf:.2f})", fontsize=9, color=color
        )
        axes[1, col].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Grad-CAM   → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATE ONE EXPERIMENT
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_experiment(
    exp_name: str,
    faces_root: Path,
    results_root: Path,
    device: torch.device,
    model_name: str = MODEL_NAME,
    ablation: str = "full",
    tta_n: int = TTA_N,
    aggregation: str = "mean",
) -> Dict:
    """
    Load best checkpoint, run TTA on test set, compute metrics, save figures.

    Returns
    -------
    metrics dict
    """
    exp_dir    = results_root / run_dir_name(exp_name, model_name, ablation)
    ckpt_dir   = results_root / checkpoint_dir_name(exp_name, model_name, ablation)
    ckpt_path  = ckpt_dir / "best_model.pth"
    plots_dir  = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Evaluating {EXP_SHORT[exp_name]}")
    print(f"{'='*60}")

    if not ckpt_path.exists():
        print(f"  ✗  No checkpoint found at {ckpt_path}")
        print(f"     Run: python train.py --exp {exp_name[:4]}")
        return {}

    model = load_model(ckpt_path, device)

    # Test loader (for Grad-CAM sample access via .dataset)
    test_loader = get_dataloader(
        faces_root, exp_name, "test",
        BATCH_SIZE, INPUT_SIZE, NUM_WORKERS, use_sampler=False
    )
    print(f"  Test set: {len(test_loader.dataset):,} faces")

    # TTA inference
    print(f"  Running TTA (×{tta_n}) …")
    y_true, y_score, face_paths = tta_predict(model, test_loader, device, tta_n)

    # Metrics
    face_metrics = compute_metrics(y_true, y_score)
    metadata = load_face_metadata(faces_root)
    y_video, score_video, video_rows = aggregate_video_predictions(
        face_paths, y_true, y_score, metadata, method=aggregation
    )
    video_metrics = compute_metrics(y_video, score_video)
    video_ci = bootstrap_metric_ci(y_video, score_video)
    metrics = {
        "exp_name": exp_name,
        "model_name": getattr(model, "model_name", MODEL_NAME),
        "primary_level": "video",
        "aggregation": aggregation,
        "tta_n": tta_n,
        "ablation": ablation,
        "face_level": face_metrics,
        "video_level": video_metrics,
        "video_level_ci95": video_ci,
        "n_faces": int(len(y_true)),
        "n_videos": int(len(y_video)),
        **video_metrics,
    }
    print(f"\n  Results:")
    print(f"    Face AUC : {face_metrics['auc']:.4f}  "
          f"Video AUC : {video_metrics['auc']:.4f}")
    print(f"    Video Acc: {video_metrics['accuracy']:.4f}")
    print(f"    Video F1 : {video_metrics['f1']:.4f}")
    print(f"    Video EER: {video_metrics['eer']:.4f}")

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    save_failure_cases(video_rows, exp_dir / "failure_cases.csv")

    # Figures
    print(f"\n  Saving figures …")
    plot_roc_curve(y_video, score_video, exp_name,
                   plots_dir / "roc_curve.png")
    plot_confusion_matrix(y_video, score_video, exp_name,
                          plots_dir / "confusion_matrix.png")
    plot_training_curves(ckpt_dir / "training_history.json",
                         plots_dir / "loss_curves.png")
    run_gradcam(model, faces_root, exp_name,
                plots_dir / "gradcam_samples.png", device)

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate deepfake detector and generate paper figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--exp",
        choices=["exp1", "exp2", "exp3", "exp4", "all"],
        default="all",
    )
    parser.add_argument("--faces",   type=Path, default=Path("../faces"))
    parser.add_argument("--results", type=Path, default=Path("../results"))
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument(
        "--model_preset",
        choices=list(MODEL_PRESETS.keys()) + ["custom"],
        default="efficientnet_b4",
    )
    parser.add_argument(
        "--ablation",
        choices=list(ABLATION_PRESETS),
        default="full",
    )
    parser.add_argument("--tta_n", type=int, default=TTA_N)
    parser.add_argument(
        "--aggregation",
        choices=["mean", "median", "max", "topk"],
        default="mean",
    )
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.exp == "all":
        exps = EXP_NAMES
    else:
        idx  = int(args.exp[-1]) - 1
        exps = [EXP_NAMES[idx]]

    all_metrics = []
    model_name = resolve_model_name(args.model, args.model_preset)
    tta_n = 1 if args.ablation == "no_tta" else args.tta_n
    eval_ablation = eval_ablation_name(args.ablation, args.aggregation)
    for exp_name in exps:
        m = evaluate_experiment(
            exp_name,
            args.faces,
            args.results,
            device,
            model_name=model_name,
            ablation=eval_ablation,
            tta_n=tta_n,
            aggregation=args.aggregation,
        )
        if m:
            all_metrics.append(m)

    if len(all_metrics) > 1:
        print(f"\n{'='*60}")
        print("  Final Results Table")
        print(f"{'='*60}")
        print(f"  {'Experiment':<30} {'AUC':>7} {'Acc':>7} {'F1':>7} {'EER':>7}")
        print(f"  {'-'*54}")
        for m in all_metrics:
            print(
                f"  {EXP_SHORT[m['exp_name']]:<30} "
                f"{m['auc']:>7.4f} "
                f"{m['accuracy']:>7.4f} "
                f"{m['f1']:>7.4f} "
                f"{m['eer']:>7.4f}"
            )


if __name__ == "__main__":
    main()
