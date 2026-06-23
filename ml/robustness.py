"""
robustness.py
=============
Evaluation-only robustness study. Re-evaluates an already-trained checkpoint on
the test set under common image degradations at several severities, and reports
video-level degradation curves. No retraining — uses the checkpoints produced by
train.py / run_all.py.

Corruptions (each at several severities):
  • JPEG compression   (quality 90 → 10)
  • Gaussian blur      (radius/sigma 0.5 → 3.0)
  • Gaussian noise     (sigma 5 → 40, on 0–255)
  • Downscale          (downscale-then-upscale, factor 0.75 → 0.25)

For each corruption×severity it recomputes video-level AUC/Acc/F1/EER (mean
aggregation), writes `robustness.json` to the run folder, and saves a degradation
curve plot. The clean baseline is computed with the SAME single-view inference path
so the drop is measured fairly.

Usage
-----
    cd ml
    # headline model, all experiments:
    python robustness.py --exp all --model_preset efficientnet_b4

    # a specific backbone:
    python robustness.py --exp all --model_preset xception

    # custom paths:
    python robustness.py --exp all --faces ../faces --results ../results
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.cuda.amp import autocast
from torchvision import transforms

# Reuse the real evaluation machinery so nothing drifts from the main metrics.
from evaluate import (
    BATCH_SIZE,
    EXP_NAMES,
    EXP_SHORT,
    INPUT_SIZE,
    MODEL_NAME,
    MODEL_PRESETS,
    USE_AMP,
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    aggregate_video_predictions,
    checkpoint_dir_name,
    compute_metrics,
    load_face_metadata,
    load_model,
    resolve_model_name,
    run_dir_name,
)
from dataset import DeepfakeDataset

# A single seeded RNG so the additive-noise corruption is reproducible run-to-run.
_RNG = np.random.default_rng(42)


# ──────────────────────────────────────────────────────────────────────────────
# CORRUPTIONS  (PIL image → corrupted PIL image)
# ──────────────────────────────────────────────────────────────────────────────

def apply_jpeg(img: Image.Image, quality: float) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=float(radius)))


def apply_noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    arr = arr + _RNG.normal(0.0, float(sigma), arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_downscale(img: Image.Image, factor: float) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    sw, sh = max(1, int(w * factor)), max(1, int(h * factor))
    small = img.resize((sw, sh), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


# corruption name → (function, severity list, x-axis label)
CORRUPTIONS: Dict[str, Dict] = {
    "jpeg": {"fn": apply_jpeg, "severities": [90, 70, 50, 30, 10], "label": "JPEG quality"},
    "blur": {"fn": apply_blur, "severities": [0.5, 1.0, 2.0, 3.0], "label": "Gaussian blur (sigma)"},
    "noise": {"fn": apply_noise, "severities": [5, 10, 20, 40], "label": "Gaussian noise (sigma)"},
    "downscale": {"fn": apply_downscale, "severities": [0.75, 0.5, 0.25], "label": "Downscale factor"},
}


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE  (single deterministic view, optional corruption)
# ──────────────────────────────────────────────────────────────────────────────

def _eval_transform() -> Callable:
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def predict_corrupted(
    model: torch.nn.Module,
    samples: List[Tuple[Path, int]],
    device: torch.device,
    corrupt_fn: Optional[Callable],
    batch_size: int = BATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Score every test crop once; apply *corrupt_fn* before normalisation."""
    tfm = _eval_transform()
    paths = [str(p) for p, _ in samples]
    labels = [int(lbl) for _, lbl in samples]
    scores: List[float] = []

    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i: i + batch_size]
        imgs = []
        for p in batch_paths:
            im = Image.open(p).convert("RGB")
            if corrupt_fn is not None:
                im = corrupt_fn(im)
            imgs.append(tfm(im))
        x = torch.stack(imgs).to(device, non_blocking=True)
        with torch.no_grad(), autocast(enabled=USE_AMP):
            logits = model(x).squeeze(1)
            s = torch.sigmoid(logits).float().cpu().numpy()
        scores.extend(np.atleast_1d(s).tolist())

    return np.array(labels, dtype=int), np.array(scores, dtype=np.float32), paths


def video_metrics(
    model, samples, device, metadata, corrupt_fn, batch_size,
) -> Dict:
    """Run (optionally corrupted) inference and return video-level metrics."""
    y_true, y_score, paths = predict_corrupted(model, samples, device, corrupt_fn, batch_size)
    y_vid, score_vid, _ = aggregate_video_predictions(
        paths, y_true, y_score, metadata, method="mean"
    )
    return compute_metrics(y_vid, score_vid)


# ──────────────────────────────────────────────────────────────────────────────
# PLOT
# ──────────────────────────────────────────────────────────────────────────────

def plot_curves(results: Dict, save_path: Path, exp_name: str) -> None:
    clean_auc = results["clean"]["auc"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(clean_auc, ls="--", color="#6B7280", lw=1.5,
               label=f"clean AUC = {clean_auc:.3f}")
    colors = {"jpeg": "#4F46E5", "blur": "#10B981", "noise": "#F59E0B", "downscale": "#EF4444"}
    for cname, cfg in results["corruptions"].items():
        sev = cfg["severities"]
        levels = list(range(1, len(sev) + 1))
        aucs = [sev[k]["auc"] for k in sev]
        ax.plot(levels, aucs, marker="o", color=colors.get(cname, None), label=cname)
    ax.set_xlabel("Severity level (mild → severe)")
    ax.set_ylabel("Video-level AUC")
    ax.set_ylim(0.45, 1.02)
    ax.set_title(f"Robustness — {EXP_SHORT.get(exp_name, exp_name)}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"    Curves    -> {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# ONE EXPERIMENT
# ──────────────────────────────────────────────────────────────────────────────

def robustness_experiment(
    exp_name: str,
    faces_root: Path,
    results_root: Path,
    device: torch.device,
    model_name: str,
    ablation: str = "full",
    batch_size: int = BATCH_SIZE,
) -> Optional[Dict]:
    run_dir = results_root / run_dir_name(exp_name, model_name, ablation)
    ckpt_dir = results_root / checkpoint_dir_name(exp_name, model_name, ablation)
    ckpt = ckpt_dir / "best_model.pth"

    print(f"\n{'='*60}\n  Robustness: {EXP_SHORT.get(exp_name, exp_name)}\n{'='*60}")
    if not ckpt.exists():
        print(f"  skip — no checkpoint at {ckpt}")
        return None

    model = load_model(ckpt, device)
    dataset = DeepfakeDataset(faces_root, exp_name, "test", transform=None)
    samples = dataset.samples
    metadata = load_face_metadata(faces_root)
    print(f"  Test crops: {len(samples):,}")

    clean = video_metrics(model, samples, device, metadata, None, batch_size)
    print(f"  clean: AUC={clean['auc']:.4f}")

    results: Dict = {
        "exp_name": exp_name,
        "model_name": model_name,
        "ablation": ablation,
        "clean": clean,
        "corruptions": {},
    }
    for cname, cfg in CORRUPTIONS.items():
        sev_results: Dict = {}
        for sev in cfg["severities"]:
            fn = (lambda im, s=sev, f=cfg["fn"]: f(im, s))
            m = video_metrics(model, samples, device, metadata, fn, batch_size)
            sev_results[str(sev)] = m
            drop = clean["auc"] - m["auc"]
            print(f"  {cname:10s} {cfg['label']}={sev:<5} -> AUC={m['auc']:.4f}  (drop {drop:+.4f})")
        results["corruptions"][cname] = {"label": cfg["label"], "severities": sev_results}

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "robustness.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"    robustness.json -> {run_dir / 'robustness.json'}")
    plot_curves(results, run_dir / "plots" / "robustness_curves.png", exp_name)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluation-only robustness study on trained checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--exp", choices=["exp1", "exp2", "exp3", "exp4", "all"], default="all")
    p.add_argument("--faces", type=Path, default=Path("../faces"))
    p.add_argument("--results", type=Path, default=Path("../results"))
    p.add_argument("--model", type=str, default=MODEL_NAME)
    p.add_argument("--model_preset", choices=list(MODEL_PRESETS.keys()) + ["custom"],
                   default="efficientnet_b4")
    p.add_argument("--ablation", default="full")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model_name = resolve_model_name(args.model, args.model_preset)
    exps = EXP_NAMES if args.exp == "all" else [EXP_NAMES[int(args.exp[-1]) - 1]]

    summary = []
    for exp_name in exps:
        r = robustness_experiment(
            exp_name, args.faces, args.results, device, model_name,
            ablation=args.ablation, batch_size=args.batch_size,
        )
        if r:
            summary.append(r)

    if summary:
        print(f"\n{'='*60}\n  Robustness summary (relative AUC drop at worst severity)\n{'='*60}")
        for r in summary:
            worst = {}
            for cname, cfg in r["corruptions"].items():
                aucs = [v["auc"] for v in cfg["severities"].values()]
                worst[cname] = r["clean"]["auc"] - min(aucs)
            drops = "  ".join(f"{k}:{v:+.3f}" for k, v in worst.items())
            print(f"  {EXP_SHORT.get(r['exp_name'], r['exp_name']):<24} clean={r['clean']['auc']:.3f}  {drops}")


if __name__ == "__main__":
    main()
