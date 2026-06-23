"""
run_all.py
==========
Run the full training + evaluation pipeline for all 4 experiments
sequentially overnight.

Produces:
  results/
    exp1_ffpp_to_ffpp/
      best_model.pth
      training_history.json
      metrics.json
      plots/  (roc, confusion, gradcam, loss curves)
    exp2_ffpp_to_celebdf/  ...
    exp3_celebdf_to_ffpp/  ...
    exp4_mixed_to_dfd/     ...
    all_results.json       ← combined metrics
    results_table.csv      ← paper-ready table

Usage
-----
    # Run everything overnight:
    python run_all.py

    # Single experiment (for testing):
    python run_all.py --exp exp1

    # Skip training (evaluate only, if models already trained):
    python run_all.py --eval_only
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

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

EXP_FLAG = {
    "exp1_ffpp_to_ffpp":    "exp1",
    "exp2_ffpp_to_celebdf": "exp2",
    "exp3_celebdf_to_ffpp": "exp3",
    "exp4_mixed_to_dfd":    "exp4",
}

MODEL_NAME = "efficientnet_b4"
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


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], desc: str) -> int:
    """Run a subprocess command, streaming output live."""
    print(f"\n{'─'*60}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'─'*60}\n")
    result = subprocess.run(cmd, check=False)
    return result.returncode


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"


def resolve_model_name(model: str, model_preset: str) -> str:
    if model_preset != "custom":
        return MODEL_PRESETS[model_preset]
    return model


def run_dir_name(exp_name: str, model_name: str, ablation: str) -> str:
    if model_name == MODEL_NAME and ablation == "full":
        return exp_name
    return f"{exp_name}__{model_name.replace('/', '_')}__{ablation}"


def eval_ablation_name(ablation: str, aggregation: str) -> str:
    if ablation == "full" and aggregation != "mean":
        return f"agg_{aggregation}"
    return ablation


def _load_metrics(results_root: Path, exp_name: str, model_name: str, ablation: str) -> dict | None:
    path = results_root / run_dir_name(exp_name, model_name, ablation) / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# RESULTS TABLE
# ──────────────────────────────────────────────────────────────────────────────

def print_results_table(all_metrics: list[dict]) -> None:
    """Print a paper-ready results table to stdout."""
    print(f"\n{'═'*70}")
    print("  FINAL RESULTS — Cross-Dataset Deepfake Detection")
    print(f"  Model: EfficientNet-B4  |  Input: 224×224  |  TTA: ×5")
    print(f"{'═'*70}")
    print(f"  {'Experiment':<30} {'AUC':>8} {'Acc':>8} {'F1':>8} {'EER':>8}")
    print(f"  {'─'*62}")
    for m in all_metrics:
        name = EXP_SHORT.get(m.get("exp_name", ""), m.get("exp_name", ""))
        print(
            f"  {name:<30} "
            f"{m.get('auc', 0):>8.4f} "
            f"{m.get('accuracy', 0):>8.4f} "
            f"{m.get('f1', 0):>8.4f} "
            f"{m.get('eer', 0):>8.4f}"
        )
    print(f"{'═'*70}")


def save_results_csv(all_metrics: list[dict], results_root: Path) -> None:
    """Save results table as CSV for easy copy-paste into LaTeX."""
    csv_path = results_root / "results_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment", "model_name", "primary_level",
                "video_auc", "video_accuracy", "video_f1", "video_eer",
                "face_auc", "n_videos", "n_faces",
            ],
        )
        writer.writeheader()
        for m in all_metrics:
            face = m.get("face_level", {})
            video = m.get("video_level", m)
            writer.writerow({
                "experiment": EXP_SHORT.get(m.get("exp_name", ""), ""),
                "model_name": m.get("model_name", ""),
                "primary_level": m.get("primary_level", "video"),
                "video_auc": round(video.get("auc", 0), 4),
                "video_accuracy": round(video.get("accuracy", 0), 4),
                "video_f1": round(video.get("f1", 0), 4),
                "video_eer": round(video.get("eer", 0), 4),
                "face_auc": round(face.get("auc", 0), 4),
                "n_videos": m.get("n_videos", ""),
                "n_faces": m.get("n_faces", ""),
            })
    print(f"  Results CSV  → {csv_path}")


def save_results_json(all_metrics: list[dict], results_root: Path) -> None:
    """Save combined metrics JSON."""
    json_path = results_root / "all_results.json"
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Results JSON → {json_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full train + eval pipeline for all experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--exp",
        choices=["exp1", "exp2", "exp3", "exp4", "all"],
        default="all",
        help="Which experiment(s) to run.",
    )
    parser.add_argument(
        "--faces",
        type=Path,
        default=Path("../faces"),
        help="Path to extracted faces root.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("../results"),
        help="Path to save results.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training, run evaluation only (models must already exist).",
    )
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
        choices=list(ABLATION_PRESETS),
        default="full",
        help="Ablation preset. no_tta affects evaluation only.",
    )
    parser.add_argument("--tta_n", type=int, default=5)
    parser.add_argument(
        "--aggregation",
        choices=["mean", "median", "max", "topk"],
        default="mean",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.exp == "all":
        exps = EXP_NAMES
    else:
        idx  = int(args.exp[-1]) - 1
        exps = [EXP_NAMES[idx]]

    args.results.mkdir(parents=True, exist_ok=True)
    model_name = resolve_model_name(args.model, args.model_preset)
    train_ablation = "full" if args.ablation == "no_tta" else args.ablation
    eval_ablation = eval_ablation_name(args.ablation, args.aggregation)
    tta_n = 1 if args.ablation == "no_tta" else args.tta_n

    python = sys.executable  # use same venv python
    total_start = time.time()
    exp_times: list[dict] = []

    print(f"\n{'═'*60}")
    print(f"  Overnight Run — {len(exps)} experiment(s)")
    print(f"  Faces   : {args.faces}")
    print(f"  Results : {args.results}")
    print(f"  Eval only: {args.eval_only}")
    print(f"{'═'*60}")

    for exp_name in exps:
        flag     = EXP_FLAG[exp_name]
        exp_start = time.time()

        # ── Training ──────────────────────────────────────────────
        if not args.eval_only:
            rc = _run(
                [python, "train.py",
                 "--exp", flag,
                 "--faces", str(args.faces),
                 "--output", str(args.results),
                 "--model", args.model,
                 "--model_preset", args.model_preset,
                 "--ablation", train_ablation],
                f"TRAINING — {EXP_SHORT[exp_name]}",
            )
            if rc != 0:
                print(f"\n  ✗  Training failed for {exp_name} (exit {rc}). "
                      "Continuing to next experiment.")
                exp_times.append({"exp": exp_name, "status": "train_failed",
                                   "elapsed": time.time() - exp_start})
                continue

        # ── Evaluation ────────────────────────────────────────────
        rc = _run(
            [python, "evaluate.py",
             "--exp", flag,
             "--faces", str(args.faces),
             "--results", str(args.results),
             "--model", args.model,
             "--model_preset", args.model_preset,
             "--ablation", args.ablation,
             "--tta_n", str(tta_n),
             "--aggregation", args.aggregation],
            f"EVALUATION — {EXP_SHORT[exp_name]}",
        )
        if rc != 0:
            print(f"\n  ✗  Evaluation failed for {exp_name} (exit {rc}).")

        elapsed = time.time() - exp_start
        exp_times.append({
            "exp":     exp_name,
            "status":  "ok" if rc == 0 else "eval_failed",
            "elapsed": elapsed,
        })
        print(f"\n  ✓  {EXP_SHORT[exp_name]} done in {_fmt_time(elapsed)}")

    # ── Aggregate results ─────────────────────────────────────────
    all_metrics = []
    for exp_name in exps:
        m = _load_metrics(args.results, exp_name, model_name, eval_ablation)
        if m:
            all_metrics.append(m)

    if all_metrics:
        print_results_table(all_metrics)
        save_results_csv(all_metrics, args.results)
        save_results_json(all_metrics, args.results)

    # ── Time summary ──────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print(f"\n{'─'*60}")
    print("  Per-experiment times:")
    for t in exp_times:
        status = "✓" if t["status"] == "ok" else "✗"
        print(f"    {status} {EXP_SHORT.get(t['exp'], t['exp']):<30} "
              f"{_fmt_time(t['elapsed'])}")
    print(f"\n  Total time: {_fmt_time(total_elapsed)}")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
