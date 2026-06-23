"""
extract_faces.py
================
Extract faces from deepfake experiment videos using MTCNN.

Reads video paths directly from experiment_manifests.json — no need to
reorganise the dataset into experiment sub-folders first.

GPU mode  (auto-detected): 1 worker, MTCNN batch inference — fastest.
CPU mode  (fallback)     : min(4, cpu_count()) workers, frame-by-frame.

Supports all four experiment layouts:
    exp1_ffpp_to_ffpp      (train/val/test)
    exp2_ffpp_to_celebdf   (train/val/test)
    exp3_celebdf_to_ffpp   (train/val/test)
    exp4_mixed_to_dfd      (train/val/test)

Usage
-----
    # Dry-run on 5 videos per folder first:
    python extract_faces.py --exp exp1 --dry_run --batch_size 8

    # Full run on everything:
    python extract_faces.py --exp all --batch_size 8

    # Crash-safe: re-run and it skips already done videos:
    python extract_faces.py --exp all --batch_size 8

    # Custom paths:
    python extract_faces.py --exp all \\
        --manifest ./experiments/experiment_manifests.json \\
        --base     ./  \\
        --output   ./faces
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import traceback
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import imageio
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

FRAME_STEP = 10           # sample every Nth frame (overridable via --frame_step)
FACE_SIZE = 380           # EfficientNet-B4 input (px)
MARGIN_RATIO = 0.20       # 20 % padding around bounding box
MAX_FACES_PER_VIDEO = 20  # cap per video (overridable via --max_faces)
MIN_FACES_WARNING = 3     # warn if fewer faces extracted
CONFIDENCE_THRESHOLD = 0.95
MIN_BOX_SIZE = 60         # px — reject tiny detections
FRAME_BLUR_THRESHOLD = 4.0   # Laplacian variance — whole frame (H.264 frames ~10, motion blur ~2)
FACE_BLUR_THRESHOLD  = 20.0  # Laplacian variance — face crop (higher res patch, stricter)
JPG_QUALITY = 95
MIN_FREE_DISK_GB = 5.0
DEFAULT_BATCH_SIZE_GPU = 32   # frames per MTCNN forward pass on GPU
DEFAULT_BATCH_SIZE_CPU = 1    # frame-by-frame on CPU

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Manifest key → output folder name
MANIFEST_KEY_TO_EXP: Dict[str, str] = {
    "experiment_1": "exp1_ffpp_to_ffpp",
    "experiment_2": "exp2_ffpp_to_celebdf",
    "experiment_3": "exp3_celebdf_to_ffpp",
    "experiment_4": "exp4_mixed_to_dfd",
}

# Experiment → list of splits present
EXP_SPLITS: Dict[str, List[str]] = {
    "exp1_ffpp_to_ffpp":    ["train", "val", "test"],
    "exp2_ffpp_to_celebdf": ["train", "val", "test"],
    "exp3_celebdf_to_ffpp": ["train", "val", "test"],
    "exp4_mixed_to_dfd":    ["train", "val", "test"],
}

EXP_SHORT_NAMES = {
    "exp1_ffpp_to_ffpp":    "Exp 1: FF++ → FF++",
    "exp2_ffpp_to_celebdf": "Exp 2: FF++ → CelebDF",
    "exp3_celebdf_to_ffpp": "Exp 3: CelebDF → FF++",
    "exp4_mixed_to_dfd":    "Exp 4: Mixed → DFD",
}

DATASET_NAMES = ("FF++", "CelebDF", "DFD")

# ──────────────────────────────────────────────────────────────────────────────
# DEVICE DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def detect_device() -> Tuple[str, int]:
    """
    Detect best available compute device.

    Returns
    -------
    device      : 'cuda', 'mps', or 'cpu'
    num_workers : 1 for GPU devices (single CUDA context), cpu_count otherwise
    """
    if torch.cuda.is_available():
        return "cuda", 1
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", 1
    return "cpu", min(4, cpu_count())


DEVICE, NUM_WORKERS = detect_device()

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(log_path: Path) -> logging.Logger:
    """Configure root logger to write errors to file + stderr."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("extract_faces")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger  # already set up (worker re-init guard)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.WARNING)
    sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ──────────────────────────────────────────────────────────────────────────────
# DISK SPACE CHECK
# ──────────────────────────────────────────────────────────────────────────────

def free_disk_gb(path: Path) -> float:
    """Return free disk space in GB at *path*."""
    return shutil.disk_usage(path).free / (1024 ** 3)


def check_disk_space(output_root: Path) -> None:
    """Pause and warn if free disk space drops below MIN_FREE_DISK_GB."""
    gb = free_disk_gb(output_root)
    if gb < MIN_FREE_DISK_GB:
        print(
            f"\n⚠  WARNING: only {gb:.1f} GB free on disk "
            f"(threshold {MIN_FREE_DISK_GB} GB).",
            file=sys.stderr,
        )
        answer = input("Continue anyway? [y/N] ").strip().lower()
        if answer != "y":
            sys.exit("Aborted by user due to low disk space.")


# ──────────────────────────────────────────────────────────────────────────────
# IMAGE UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def _is_blurry(image: np.ndarray, threshold: float) -> bool:
    """Return True when Laplacian variance is below *threshold*."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def _crop_face(frame: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
    """
    Crop a face from *frame*, add margin, resize to FACE_SIZE × FACE_SIZE.

    Parameters
    ----------
    frame : H×W×3 BGR numpy array
    box   : [x1, y1, x2, y2] float pixel coordinates from MTCNN

    Returns
    -------
    Resized BGR crop, or None if box is too small or crop is empty.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box.astype(int)

    bw, bh = x2 - x1, y2 - y1
    if bw < MIN_BOX_SIZE or bh < MIN_BOX_SIZE:
        return None

    mx, my = int(bw * MARGIN_RATIO), int(bh * MARGIN_RATIO)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return cv2.resize(crop, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LANCZOS4)


def _infer_dataset(video_path: Path) -> str:
    """Infer dataset name from the path components used in this project."""
    parts = {p.lower() for p in video_path.parts}
    for name in DATASET_NAMES:
        if name.lower() in parts:
            return name
    return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# FACE EXTRACTION — single video, batch-aware
# ──────────────────────────────────────────────────────────────────────────────

def _read_sampled_frames(
    video_path: Path,
) -> List[Tuple[int, np.ndarray]]:
    """
    Read all sampled, non-blurry frames from a video.

    Uses imageio + ffmpeg backend for robust H.264 support on Windows.
    Falls back to OpenCV if imageio fails.

    Returns list of (frame_index, bgr_frame). The caller batches these
    for MTCNN to avoid loading the entire video into RAM twice.
    """
    frames: List[Tuple[int, np.ndarray]] = []

    # ── Try imageio/ffmpeg first (reliable H.264 on Windows) ──────────────
    try:
        reader = imageio.get_reader(str(video_path), "ffmpeg")
        idx = 0
        for rgb_frame in reader:
            if idx % FRAME_STEP == 0:
                bgr = cv2.cvtColor(np.asarray(rgb_frame), cv2.COLOR_RGB2BGR)
                if not _is_blurry(bgr, FRAME_BLUR_THRESHOLD):
                    frames.append((idx, bgr))
            idx += 1
        reader.close()
        return frames
    except Exception:
        pass  # fall through to OpenCV

    # ── OpenCV fallback ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % FRAME_STEP == 0 and not _is_blurry(frame, FRAME_BLUR_THRESHOLD):
                frames.append((idx, frame))
            idx += 1
    finally:
        cap.release()
    return frames


def extract_faces_from_video(
    video_path: Path,
    output_dir: Path,
    mtcnn: MTCNN,
    log_path: Path,
    batch_size: int,
    context: Dict[str, str],
) -> Tuple[int, List[Dict]]:
    """
    Extract up to MAX_FACES_PER_VIDEO face images from a single video.

    When *batch_size* > 1 (GPU mode), frames are grouped and passed to
    MTCNN in one forward pass per batch — much faster than one-by-one.

    Parameters
    ----------
    video_path : Source video file.
    output_dir : Directory where face JPGs are saved.
    mtcnn      : Initialised MTCNN detector (per-process instance).
    log_path   : Path to the shared error log file.
    batch_size : Number of frames per MTCNN forward pass.

    Returns
    -------
    (number of faces successfully saved, metadata records)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    saved = 0
    face_idx = 0
    records: List[Dict] = []

    try:
        sampled = _read_sampled_frames(video_path)
        if not sampled:
            msg = f"Cannot open or no frames: {video_path}"
            logging.getLogger("extract_faces").warning(msg)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
            return 0, records

        # ── Batch MTCNN inference ──────────────────────────────────────────
        for batch_start in range(0, len(sampled), batch_size):
            if saved >= MAX_FACES_PER_VIDEO:
                break

            batch = sampled[batch_start : batch_start + batch_size]
            pil_batch = [
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                for _, frame in batch
            ]

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # MTCNN accepts a list → returns list of boxes/probs
                    if len(pil_batch) == 1:
                        # detect() returns arrays directly for single image
                        boxes_list, probs_list = mtcnn.detect(pil_batch[0])
                        boxes_list = [boxes_list]
                        probs_list = [probs_list]
                    else:
                        boxes_list, probs_list = mtcnn.detect(pil_batch)
            except Exception:
                continue  # skip this batch silently

            for (frame_idx, frame), boxes, probs in zip(
                batch, boxes_list, probs_list
            ):
                if saved >= MAX_FACES_PER_VIDEO:
                    break
                if boxes is None or probs is None:
                    continue

                for box, prob in zip(boxes, probs):
                    if saved >= MAX_FACES_PER_VIDEO:
                        break
                    if prob < CONFIDENCE_THRESHOLD:
                        continue

                    cropped = _crop_face(frame, box)
                    if cropped is None or _is_blurry(cropped, FACE_BLUR_THRESHOLD):
                        continue

                    fname = f"{stem}_frame{frame_idx:04d}_face{face_idx}.jpg"
                    cv2.imwrite(
                        str(output_dir / fname),
                        cropped,
                        [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY],
                    )
                    records.append({
                        "face_path": str((output_dir / fname).resolve()),
                        "video_path": str(video_path.resolve()),
                        "video_id": stem,
                        "dataset": context.get("dataset", _infer_dataset(video_path)),
                        "experiment": context.get("experiment", ""),
                        "split": context.get("split", ""),
                        "label": context.get("label", ""),
                        "frame_index": frame_idx,
                        "face_index": face_idx,
                        "mtcnn_confidence": float(prob),
                    })
                    saved += 1
                    face_idx += 1

    except Exception:
        err = traceback.format_exc()
        logging.getLogger("extract_faces").error(
            "Error processing %s:\n%s", video_path, err
        )
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"ERROR {video_path}:\n{err}\n")

    return saved, records


# ──────────────────────────────────────────────────────────────────────────────
# WORKER INITIALISER & TASK FUNCTION  (multiprocessing)
# ──────────────────────────────────────────────────────────────────────────────

_worker_mtcnn: Optional[MTCNN] = None
_worker_batch_size: int = DEFAULT_BATCH_SIZE_CPU


def _worker_init(log_path_str: str, device: str, batch_size: int) -> None:
    """
    Initialise one MTCNN instance per worker process.

    Uses *device* passed from main so GPU workers use 'cuda'/'mps'
    and CPU workers use 'cpu'.
    """
    global _worker_mtcnn, _worker_batch_size

    _worker_batch_size = batch_size
    _worker_mtcnn = MTCNN(
        keep_all=True,
        device=device,
        post_process=False,
        select_largest=False,
    )

    log_path = Path(log_path_str)
    logger = logging.getLogger("extract_faces")
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.WARNING)
        fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
        logger.addHandler(fh)
        logger.setLevel(logging.WARNING)


def _worker_task(args: Tuple) -> Tuple[str, int, List[Dict]]:
    """
    Worker entry point (packed tuple for multiprocessing compatibility).

    args: (video_path_str, output_dir_str, log_path_str, context)
    Returns: (video_path_str, faces_saved, metadata_records)
    """
    video_path_str, output_dir_str, log_path_str, context = args
    try:
        n, records = extract_faces_from_video(
            video_path=Path(video_path_str),
            output_dir=Path(output_dir_str),
            mtcnn=_worker_mtcnn,
            log_path=Path(log_path_str),
            batch_size=_worker_batch_size,
            context=context,
        )
    except Exception:
        err = traceback.format_exc()
        with open(log_path_str, "a", encoding="utf-8") as f:
            f.write(f"WORKER ERROR {video_path_str}:\n{err}\n")
        n = 0
        records = []
    return (video_path_str, n, records)


# ──────────────────────────────────────────────────────────────────────────────
# VIDEO COLLECTION — reads directly from experiment_manifests.json
# ──────────────────────────────────────────────────────────────────────────────

def _fix_path(raw: str, base: Path) -> Path:
    """
    Resolve a manifest path (may use Windows backslashes) against *base*.

    e.g. 'dataset\\FF++\\real\\881.mp4' → base / 'dataset/FF++/real/881.mp4'
    """
    return base / raw.replace("\\", "/")


def collect_video_tasks(
    exp_names: List[str],
    manifest_path: Path,
    base_dir: Path,
    output_root: Path,
    log_path: Path,
    dry_run: bool,
) -> Tuple[List[Tuple], int]:
    """
    Read video paths from *manifest_path* and build worker task tuples,
    skipping videos that already have extracted face files.

    Parameters
    ----------
    exp_names     : e.g. ['exp1_ffpp_to_ffpp', ...]
    manifest_path : path to experiment_manifests.json
    base_dir      : root used to resolve relative paths inside manifest
    output_root   : where face JPGs are saved
    log_path      : shared error log
    dry_run       : if True, cap each split/label to 5 videos

    Returns
    -------
    tasks   : list of (video_path_str, output_dir_str, log_path_str, context)
    skipped : number of already-completed videos
    """
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # Build reverse map: exp_name → manifest key
    exp_to_key = {v: k for k, v in MANIFEST_KEY_TO_EXP.items()}

    tasks: List[Tuple] = []
    skipped = 0
    missing = 0

    for exp_name in exp_names:
        mkey = exp_to_key.get(exp_name)
        if mkey is None or mkey not in manifest:
            print(f"  ⚠  {exp_name} not found in manifest — skipping.")
            continue

        exp_data = manifest[mkey]
        splits = [s for s in EXP_SPLITS.get(exp_name, []) if s in exp_data]

        for split in splits:
            for label in ("real", "fake"):
                raw_paths: List[str] = exp_data[split].get(label, [])
                if dry_run:
                    raw_paths = raw_paths[:5]

                dst_dir = output_root / exp_name / split / label

                for raw in raw_paths:
                    vid = _fix_path(raw, base_dir)

                    if not vid.exists():
                        missing += 1
                        with open(log_path, "a", encoding="utf-8") as lf:
                            lf.write(f"MISSING {vid}\n")
                        continue

                    # Resume: skip if any face already extracted for this video
                    existing = list(dst_dir.glob(f"{vid.stem}_frame*_face*.jpg"))
                    if existing:
                        skipped += 1
                        continue

                    context = {
                        "experiment": exp_name,
                        "split": split,
                        "label": label,
                        "dataset": _infer_dataset(vid),
                    }
                    tasks.append((str(vid), str(dst_dir), str(log_path), context))

    if missing:
        print(f"  ⚠  {missing} video path(s) in manifest not found on disk "
              f"(logged to {log_path})")

    return tasks, skipped


def append_metadata_jsonl(metadata_path: Path, records: List[Dict]) -> None:
    """Append face metadata records in JSON Lines format."""
    if not records:
        return
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────────────────

def count_faces(output_root: Path, exp_names: List[str]) -> Dict:
    """Count extracted face JPGs per experiment / split / label."""
    counts: Dict = {}
    for exp_name in exp_names:
        counts[exp_name] = {}
        for split in EXP_SPLITS.get(exp_name, []):
            counts[exp_name][split] = {}
            for label in ("real", "fake"):
                d = output_root / exp_name / split / label
                counts[exp_name][split][label] = (
                    len(list(d.glob("*.jpg"))) if d.exists() else 0
                )
    return counts


def print_summary_table(counts: Dict, exp_names: List[str]) -> None:
    """Print a formatted ASCII summary table for each experiment."""
    W = 53
    for exp_name in exp_names:
        short = EXP_SHORT_NAMES.get(exp_name, exp_name)
        splits = EXP_SPLITS.get(exp_name, [])

        print(f"\n┌{'─'*W}┐")
        print(f"│ {short:<{W-2}} │")
        print(f"├{'─'*15}┬{'─'*11}┬{'─'*11}┬{'─'*13}┤")
        print(f"│ {'Split':<13} │ {'Real':^9} │ {'Fake':^9} │ {'Total':^11} │")
        print(f"├{'─'*15}┼{'─'*11}┼{'─'*11}┼{'─'*13}┤")

        grand_real = grand_fake = 0
        for split in splits:
            r = counts[exp_name][split].get("real", 0)
            f = counts[exp_name][split].get("fake", 0)
            grand_real += r
            grand_fake += f
            print(f"│ {split.capitalize():<13} │ {r:^9,} │ {f:^9,} │ {r+f:^11,} │")

        print(f"├{'─'*15}┼{'─'*11}┼{'─'*11}┼{'─'*13}┤")
        grand = grand_real + grand_fake
        print(f"│ {'TOTAL':<13} │ {grand_real:^9,} │ {grand_fake:^9,} │ {grand:^11,} │")
        print(f"└{'─'*15}┴{'─'*11}┴{'─'*11}┴{'─'*13}┘")


def save_summary_json(
    counts: Dict, exp_names: List[str], output_root: Path
) -> None:
    """Persist face counts to face_extraction_summary.json."""
    summary: Dict = {}
    for exp_name in exp_names:
        short = EXP_SHORT_NAMES.get(exp_name, exp_name)
        splits = EXP_SPLITS.get(exp_name, [])
        summary[exp_name] = {"label": short, "splits": {}, "grand_total": 0}
        for split in splits:
            r = counts[exp_name][split].get("real", 0)
            f = counts[exp_name][split].get("fake", 0)
            summary[exp_name]["splits"][split] = {"real": r, "fake": f, "total": r + f}
            summary[exp_name]["grand_total"] += r + f

    out = output_root / "face_extraction_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSummary saved → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract faces from deepfake experiment videos using MTCNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--exp",
        choices=["exp1", "exp2", "exp3", "exp4", "all"],
        default="all",
        help="Which experiment(s) to process.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("../experiments/experiment_manifests.json"),
        help="Path to experiment_manifests.json.",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("../"),
        help="Base directory used to resolve relative video paths in the manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../faces"),
        help="Root folder where extracted faces are saved.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Process only the first 5 videos per split/label (pipeline check).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help=(
            "Frames per MTCNN forward pass. "
            f"Defaults to {DEFAULT_BATCH_SIZE_GPU} on GPU, "
            f"{DEFAULT_BATCH_SIZE_CPU} on CPU."
        ),
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=None,
        help=f"Sample every Nth frame from each video. Default: {FRAME_STEP}.",
    )
    parser.add_argument(
        "--max_faces",
        type=int,
        default=None,
        help=f"Maximum faces to extract per video. Default: {MAX_FACES_PER_VIDEO}.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # CUDA requires 'spawn' start method for child processes
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=False)
    except RuntimeError:
        pass  # already set (e.g. on Windows where spawn is default)

    args = parse_args()

    # Resolve device / batch size / workers
    device, num_workers = detect_device()

    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        batch_size = DEFAULT_BATCH_SIZE_GPU if device != "cpu" else DEFAULT_BATCH_SIZE_CPU

    # GPU info string
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        device_str = f"CUDA — {gpu_name} ({vram_gb:.1f} GB)"
    elif device == "mps":
        device_str = "Apple MPS"
    else:
        device_str = "CPU"

    # Resolve experiment list
    all_exps = list(EXP_SPLITS.keys())
    if args.exp == "all":
        exp_names = all_exps
    else:
        idx = int(args.exp[-1]) - 1
        exp_names = [all_exps[idx]]

    # Override module-level constants if user provided CLI values
    global FRAME_STEP, MAX_FACES_PER_VIDEO
    if args.frame_step is not None:
        FRAME_STEP = args.frame_step
    if args.max_faces is not None:
        MAX_FACES_PER_VIDEO = args.max_faces

    manifest_path: Path = args.manifest.resolve()
    base_dir: Path = args.base.resolve()
    output_root: Path = args.output.resolve()
    log_path: Path = output_root / "extraction_errors.log"
    metadata_path: Path = output_root / "face_metadata.jsonl"

    output_root.mkdir(parents=True, exist_ok=True)
    setup_logging(log_path)

    if not manifest_path.exists():
        sys.exit(f"Manifest not found: {manifest_path}")

    print("=" * 62)
    print("  Face Extraction Pipeline — MTCNN + EfficientNet-B4 prep")
    print("=" * 62)
    print(f"  Device      : {device_str}")
    print(f"  Workers     : {num_workers}  (1 on GPU to share single context)")
    print(f"  Manifest    : {manifest_path}")
    print(f"  Base dir    : {base_dir}")
    print(f"  Output root : {output_root}")
    print(f"  Metadata    : {metadata_path}")
    print(f"  Experiments : {', '.join(exp_names)}")
    print(f"  Frame step  : every {FRAME_STEP} frames")
    print(f"  Max faces   : {MAX_FACES_PER_VIDEO} per video")
    print(f"  Batch size  : {batch_size} frames per MTCNN pass")
    print(f"  Dry run     : {args.dry_run}")
    print("=" * 62)

    if args.dry_run:
        print("\n[DRY RUN] Processing only the first 5 videos per split/label.\n")

    print("Scanning manifest for videos …")
    tasks, skipped = collect_video_tasks(
        exp_names=exp_names,
        manifest_path=manifest_path,
        base_dir=base_dir,
        output_root=output_root,
        log_path=log_path,
        dry_run=args.dry_run,
    )
    print(f"  Videos to process : {len(tasks)}")
    print(f"  Videos skipped    : {skipped} (already extracted)")

    if not tasks:
        print("\nNothing to do. All videos already processed.")
    else:
        check_disk_space(output_root)

        total_faces = 0
        low_yield: List[str] = []

        with Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(str(log_path), device, batch_size),
        ) as pool:
            with tqdm(
                total=len(tasks),
                desc="Extracting faces",
                unit="video",
                dynamic_ncols=True,
                smoothing=0.1,
            ) as pbar:
                for vid_str, n, records in pool.imap_unordered(
                    _worker_task, tasks, chunksize=1
                ):
                    total_faces += n
                    append_metadata_jsonl(metadata_path, records)
                    if 0 < n < MIN_FACES_WARNING:
                        low_yield.append(f"{Path(vid_str).name} ({n} face(s))")
                    pbar.set_postfix(faces=f"{total_faces:,}")
                    pbar.update(1)
                    if pbar.n % 100 == 0:
                        check_disk_space(output_root)

        print(f"\nTotal faces extracted : {total_faces:,}")
        if low_yield:
            print(
                f"\n⚠  {len(low_yield)} video(s) yielded fewer than "
                f"{MIN_FACES_WARNING} faces:"
            )
            for v in low_yield:
                print(f"   • {v}")

    print("\nCounting extracted faces …")
    counts = count_faces(output_root, exp_names)
    print_summary_table(counts, exp_names)
    save_summary_json(counts, exp_names, output_root)

    if log_path.exists() and log_path.stat().st_size > 0:
        print(f"\n⚠  Some errors were logged → {log_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
