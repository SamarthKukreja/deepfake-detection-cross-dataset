#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_datasets.py
====================
Downloads minimal deepfake-detection datasets (Celeb-DF v2 + DeepFake-TIMIT)
for cross-dataset evaluation experiments.

Usage:
    python download_datasets.py [--output ./dataset] [--dry_run] [--skip_timit]

Author: ML Engineer
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

# Force UTF-8 output on Windows so box-drawing chars don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
elif hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure file + console logging. Log file goes to <output_dir>/download_errors.log."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "download_errors.log"

    logger = logging.getLogger("downloader")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ── Dependency management ─────────────────────────────────────────────────────

REQUIRED_PACKAGES: dict[str, str] = {
    "gdown": "gdown",
    "tqdm": "tqdm",
    "huggingface_hub": "huggingface_hub",
    "kaggle": "kaggle",
    "requests": "requests",
}


def ensure_dependencies(logger: logging.Logger) -> None:
    """Auto-install any missing pip packages from REQUIRED_PACKAGES."""
    missing = []
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        logger.info("Installing missing dependencies: %s", ", ".join(missing))
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing
        )
        logger.info("Dependencies installed successfully.")
    else:
        logger.info("All dependencies already installed.")


# ── Disk space checks ─────────────────────────────────────────────────────────

GB = 1024 ** 3
MIN_HARD_GB = 8
MIN_WARN_GB = 15


def check_disk_space(path: Path, logger: logging.Logger) -> None:
    """
    Abort if < 8 GB free. Warn and ask for confirmation if < 15 GB free.

    Parameters
    ----------
    path : Path
        Directory to check (uses its mount point).
    logger : logging.Logger
    """
    usage = shutil.disk_usage(path)
    free_gb = usage.free / GB

    if free_gb < MIN_HARD_GB:
        logger.error(
            "Only %.1f GB free on disk (minimum required: %d GB). Aborting.",
            free_gb, MIN_HARD_GB,
        )
        sys.exit(1)

    if free_gb < MIN_WARN_GB:
        logger.warning(
            "Only %.1f GB free on disk (recommended: %d GB).",
            free_gb, MIN_WARN_GB,
        )
        answer = input("Continue anyway? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            logger.info("User aborted.")
            sys.exit(0)
    else:
        logger.info("Disk space OK: %.1f GB free.", free_gb)


# ── File utilities ────────────────────────────────────────────────────────────

VIDEO_EXTS = {".mp4", ".avi", ".mov"}


def collect_videos(directory: Path) -> list[Path]:
    """Return sorted list of video files under *directory* (recursive)."""
    return sorted(
        p for p in directory.rglob("*") if p.suffix.lower() in VIDEO_EXTS
    )


def prune_to_limit(
    source_dir: Path,
    dest_dir: Path,
    limit: int,
    label: str,
    logger: logging.Logger,
) -> int:
    """
    Move the first *limit* video files from *source_dir* (alphabetically) to
    *dest_dir*, then delete every remaining video in *source_dir*.

    Returns the number of videos moved.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    videos = collect_videos(source_dir)

    if not videos:
        logger.warning("No videos found in %s", source_dir)
        return 0

    keep = videos[:limit]
    discard = videos[limit:]

    moved = 0
    for v in keep:
        dst = dest_dir / v.name
        if dst.exists():
            dst = dest_dir / f"{v.stem}_{moved}{v.suffix}"
        shutil.move(str(v), dst)
        moved += 1

    freed_bytes = 0
    for v in discard:
        if v.exists():
            freed_bytes += v.stat().st_size
            v.unlink()

    logger.info(
        "[%s] Kept %d / %d videos. Freed %.2f MB.",
        label, moved, len(videos), freed_bytes / (1024 ** 2),
    )
    return moved


def dir_size_gb(path: Path) -> float:
    """Return total size of all files under *path* in GB."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / GB


# ── Retry helper ──────────────────────────────────────────────────────────────

def with_retry(
    fn: Callable,
    retries: int = 3,
    wait: float = 10.0,
    logger: logging.Logger | None = None,
) -> bool:
    """
    Call *fn()* up to *retries* times. Return True on first success.
    Waits *wait* seconds between attempts.
    """
    for attempt in range(1, retries + 1):
        try:
            fn()
            return True
        except Exception as exc:
            msg = f"Attempt {attempt}/{retries} failed: {exc}"
            if logger:
                logger.error(msg)
            else:
                print(msg, file=sys.stderr)
            if attempt < retries:
                time.sleep(wait)
    return False


# ── Kaggle credential check ───────────────────────────────────────────────────

def kaggle_creds_available() -> bool:
    """
    Return True if Kaggle credentials are available via env vars OR
    the standard ~/.kaggle/kaggle.json file.
    """
    # Env vars
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    # JSON file (Windows: %USERPROFILE%\.kaggle\kaggle.json, Unix: ~/.kaggle/kaggle.json)
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    return False


# ── Download helpers ──────────────────────────────────────────────────────────

def try_huggingface(
    repo_id: str,
    local_dir: Path,
    logger: logging.Logger,
) -> bool:
    """
    Download a HuggingFace dataset repo via huggingface_hub.snapshot_download.

    Parameters
    ----------
    repo_id : str
        HuggingFace dataset ID, e.g. "OpenRL/CelebDF".
    local_dir : Path
        Destination directory.
    logger : logging.Logger

    Returns
    -------
    bool
        True on success.
    """
    logger.info("[HuggingFace] Attempting download of %s ...", repo_id)
    try:
        from huggingface_hub import snapshot_download

        local_dir.mkdir(parents=True, exist_ok=True)

        def _do():
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )

        success = with_retry(_do, retries=3, wait=10, logger=logger)
        if success:
            logger.info("[HuggingFace] Download complete -> %s", local_dir)
        return success
    except Exception as exc:
        logger.error("[HuggingFace] Failed: %s", exc)
        return False


def try_gdown(
    folder_url: str,
    local_dir: Path,
    logger: logging.Logger,
) -> bool:
    """
    Download a Google Drive folder via gdown.

    Parameters
    ----------
    folder_url : str
        Google Drive folder URL.
    local_dir : Path
        Destination directory.
    logger : logging.Logger

    Returns
    -------
    bool
        True on success.
    """
    logger.info("[gdown] Attempting folder download: %s ...", folder_url)
    try:
        import gdown

        local_dir.mkdir(parents=True, exist_ok=True)

        def _do():
            gdown.download_folder(folder_url, output=str(local_dir), quiet=False)

        success = with_retry(_do, retries=3, wait=10, logger=logger)
        if success:
            logger.info("[gdown] Download complete -> %s", local_dir)
        return success
    except Exception as exc:
        logger.error("[gdown] Failed: %s", exc)
        return False


def try_kaggle(
    dataset_slug: str,
    local_dir: Path,
    logger: logging.Logger,
) -> bool:
    """
    Download a Kaggle dataset using the kaggle CLI.

    Checks for credentials via env vars OR ~/.kaggle/kaggle.json.

    Parameters
    ----------
    dataset_slug : str
        Kaggle dataset slug, e.g. "sakshigoyal7/deepfake-timit".
    local_dir : Path
        Destination directory.
    logger : logging.Logger

    Returns
    -------
    bool
        True on success.
    """
    if not kaggle_creds_available():
        logger.error(
            "Kaggle credentials not found.\n"
            "Run these two commands then retry:\n"
            "  export KAGGLE_USERNAME=your_username\n"
            "  export KAGGLE_KEY=your_api_key\n"
            "Get your key from: https://www.kaggle.com/settings"
        )
        return False

    logger.info("[Kaggle] Attempting download of %s ...", dataset_slug)
    local_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", dataset_slug,
        "-p", str(local_dir),
        "--unzip",
    ]

    def _do():
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())

    success = with_retry(_do, retries=3, wait=10, logger=logger)
    if success:
        logger.info("[Kaggle] Download complete -> %s", local_dir)
    return success


def try_wget_timit(
    raw_dir: Path,
    logger: logging.Logger,
) -> bool:
    """
    Download DeepFake-TIMIT via direct HTTP from conradsanderson.id.au.

    Parameters
    ----------
    raw_dir : Path
        Root raw directory; real/ and fake/ subdirs are created inside.
    logger : logging.Logger

    Returns
    -------
    bool
        True on success.
    """
    import requests
    from tqdm import tqdm

    base_urls = {
        "fake": "https://conradsanderson.id.au/vidtimit/higher_quality/",
        "real": "https://conradsanderson.id.au/vidtimit/original/",
    }

    all_ok = True
    for label, base_url in base_urls.items():
        dest = raw_dir / label
        dest.mkdir(parents=True, exist_ok=True)
        logger.info("[wget-TIMIT] Fetching index for %s from %s", label, base_url)

        try:
            resp = requests.get(base_url, timeout=60)
            resp.raise_for_status()
        except Exception as exc:
            logger.error("[wget-TIMIT] Could not fetch index for %s: %s", label, exc)
            all_ok = False
            continue

        from html.parser import HTMLParser

        class _LinkParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.links: list[str] = []

            def handle_starttag(self, tag, attrs):
                if tag == "a":
                    href = dict(attrs).get("href", "")
                    if href.lower().endswith(".avi"):
                        self.links.append(href)

        parser = _LinkParser()
        parser.feed(resp.text)
        links = parser.links

        if not links:
            logger.warning(
                "[wget-TIMIT] No .avi links found at %s. "
                "The URL may have changed or require authentication.",
                base_url,
            )
            all_ok = False
            continue

        logger.info("[wget-TIMIT] Found %d .avi files for %s", len(links), label)

        for href in tqdm(links, desc=f"TIMIT/{label}", unit="file"):
            url = base_url + href if not href.startswith("http") else href
            fname = dest / Path(href).name
            if fname.exists():
                continue

            def _fetch(u=url, f=fname):
                r = requests.get(u, timeout=60, stream=True)
                r.raise_for_status()
                with open(f, "wb") as fout:
                    for chunk in r.iter_content(chunk_size=8192):
                        fout.write(chunk)

            if not with_retry(_fetch, retries=3, wait=10, logger=logger):
                logger.error("[wget-TIMIT] Failed to download %s", url)
                all_ok = False

    return all_ok


def print_manual_instructions_celebdf() -> None:
    """Print manual download instructions for Celeb-DF v2."""
    msg = """
==============================================================
MANUAL DOWNLOAD -- Celeb-DF v2
==============================================================
All automatic options failed. Download manually:

1. Visit https://github.com/yuezunli/celeb-deepfakeforensics
   and fill in the Google Form to get access.
2. Once you have the Drive link, run:
     gdown --folder <your-drive-link> -O ./dataset/CelebDF/raw
3. Or download via Kaggle:
     kaggle datasets download -d reubensuju/celeb-df-v2 -p ./dataset/CelebDF/raw --unzip
==============================================================
"""
    print(msg)


def print_manual_instructions_timit() -> None:
    """Print manual download instructions for DeepFake-TIMIT."""
    msg = """
==============================================================
MANUAL DOWNLOAD -- DeepFake-TIMIT
==============================================================
All automatic options failed. Download manually:

1. Via Kaggle:
     kaggle datasets download -d sakshigoyal7/deepfake-timit -p ./dataset/TIMIT/raw --unzip

2. Or via direct links:
   Fake (HQ): https://conradsanderson.id.au/vidtimit/higher_quality/
   Real:      https://conradsanderson.id.au/vidtimit/original/
==============================================================
"""
    print(msg)


# ── Directory helpers ─────────────────────────────────────────────────────────

def _find_subdir(parent: Path, candidates: list[str]) -> Path | None:
    """
    Return the first existing subdirectory under *parent* whose name
    matches any entry in *candidates* (case-insensitive).
    """
    if not parent.exists():
        return None
    for entry in parent.rglob("*"):
        if entry.is_dir() and entry.name.lower() in [c.lower() for c in candidates]:
            return entry
    return None


def _safe_rmtree(path: Path, logger: logging.Logger) -> None:
    """Remove a directory tree, logging any errors rather than crashing."""
    try:
        if path.exists():
            shutil.rmtree(path)
            logger.info("Deleted temporary raw dir: %s", path)
    except Exception as exc:
        logger.warning("Could not delete %s: %s", path, exc)


# ── Dataset downloaders ───────────────────────────────────────────────────────

def download_celebdf(
    output_dir: Path,
    dry_run: bool,
    logger: logging.Logger,
) -> dict:
    """
    Download Celeb-DF v2, prune to 300 real + 300 fake, and organise.

    Tries (in order):
      A - HuggingFace snapshot_download
      B - gdown Google Drive folder
      C - Kaggle CLI

    Parameters
    ----------
    output_dir : Path
        Root dataset directory.
    dry_run : bool
        If True, print plan only without downloading.
    logger : logging.Logger

    Returns
    -------
    dict
        Metadata suitable for dataset_info.json.
    """
    raw_dir  = output_dir / "CelebDF" / "raw"
    real_dir = output_dir / "CelebDF" / "real"
    fake_dir = output_dir / "CelebDF" / "fake"

    if dry_run:
        print("[DRY RUN] Would download Celeb-DF v2 -> organize 300 real + 300 fake")
        return {}

    logger.info("--- Celeb-DF v2 ---")

    success = False

    # Option A -- HuggingFace
    if not success:
        success = try_huggingface("OpenRL/CelebDF", raw_dir, logger)
        if success:
            logger.info("Celeb-DF: Option A (HuggingFace) succeeded.")

    # Option B -- gdown
    if not success:
        gdrive_url = (
            "https://drive.google.com/drive/folders/"
            "1iLx76ok0TtLB6lBHkjFNrMQ5FHMjI5BO"
        )
        success = try_gdown(gdrive_url, raw_dir, logger)
        if success:
            logger.info("Celeb-DF: Option B (gdown) succeeded.")

    # Option C -- Kaggle
    if not success:
        success = try_kaggle("reubensuju/celeb-df-v2", raw_dir, logger)
        if success:
            logger.info("Celeb-DF: Option C (Kaggle) succeeded.")

    if not success:
        print_manual_instructions_celebdf()
        return {
            "real_count": 0, "fake_count": 0,
            "real_path": str(real_dir), "fake_path": str(fake_dir),
            "size_gb": 0.0, "status": "failed",
        }

    source_real = _find_subdir(raw_dir, ["Celeb-real", "real", "Real"])
    source_fake = _find_subdir(raw_dir, ["Celeb-synthesis", "fake", "Fake", "synthesis"])

    if not source_real:
        source_real = raw_dir / "real"
    if not source_fake:
        source_fake = raw_dir / "fake"

    real_count = prune_to_limit(source_real, real_dir, 150, "CelebDF/real", logger)
    fake_count = prune_to_limit(source_fake, fake_dir, 150, "CelebDF/fake", logger)

    _safe_rmtree(raw_dir, logger)

    size = dir_size_gb(output_dir / "CelebDF")
    return {
        "real_count": real_count,
        "fake_count": fake_count,
        "real_path": str(real_dir),
        "fake_path": str(fake_dir),
        "size_gb": round(size, 2),
        "status": "complete" if (real_count == 150 and fake_count == 150) else "partial",
    }


def download_timit(
    output_dir: Path,
    dry_run: bool,
    logger: logging.Logger,
) -> dict:
    """
    Download DeepFake-TIMIT (HQ version), prune to 100 real + 100 fake.

    Tries (in order):
      A - Kaggle CLI
      B - Direct HTTP from conradsanderson.id.au

    Parameters
    ----------
    output_dir : Path
        Root dataset directory.
    dry_run : bool
        If True, print plan only without downloading.
    logger : logging.Logger

    Returns
    -------
    dict
        Metadata suitable for dataset_info.json.
    """
    raw_dir  = output_dir / "TIMIT" / "raw"
    real_dir = output_dir / "TIMIT" / "real"
    fake_dir = output_dir / "TIMIT" / "fake"

    if dry_run:
        print("[DRY RUN] Would download DeepFake-TIMIT -> organize 100 real + 100 fake")
        return {}

    logger.info("--- DeepFake-TIMIT ---")

    success = False

    # Option A -- Kaggle
    if not success:
        success = try_kaggle("sakshigoyal7/deepfake-timit", raw_dir, logger)
        if success:
            logger.info("TIMIT: Option A (Kaggle) succeeded.")

    # Option B -- Direct HTTP
    if not success:
        success = try_wget_timit(raw_dir, logger)
        if success:
            logger.info("TIMIT: Option B (wget) succeeded.")

    if not success:
        print_manual_instructions_timit()
        return {
            "real_count": 0, "fake_count": 0,
            "real_path": str(real_dir), "fake_path": str(fake_dir),
            "size_gb": 0.0, "status": "failed",
        }

    source_real = _find_subdir(raw_dir, ["original", "real", "Real", "vidtimit"])
    source_fake = _find_subdir(raw_dir, ["higher_quality", "fake", "Fake", "hq"])

    if not source_real:
        source_real = raw_dir / "real"
    if not source_fake:
        source_fake = raw_dir / "fake"

    real_count = prune_to_limit(source_real, real_dir, 50, "TIMIT/real", logger)
    fake_count = prune_to_limit(source_fake, fake_dir, 50, "TIMIT/fake", logger)

    _safe_rmtree(raw_dir, logger)

    size = dir_size_gb(output_dir / "TIMIT")
    return {
        "real_count": real_count,
        "fake_count": fake_count,
        "real_path": str(real_dir),
        "fake_path": str(fake_dir),
        "size_gb": round(size, 2),
        "status": "complete" if (real_count == 50 and fake_count == 50) else "partial",
    }


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(info: dict) -> None:
    """
    Print a formatted summary table of downloaded datasets.

    Parameters
    ----------
    info : dict
        As returned by the per-dataset download functions.
    """
    sep = "-" * 56
    print("\n+" + sep + "+")
    print("| {:<12} {:>6} {:>6} {:>9} {:<10} |".format(
        "Dataset", "Real", "Fake", "Size", "Status"))
    print("+" + sep + "+")

    total_real = total_fake = 0
    total_size = 0.0

    for name, d in info.items():
        if not d:
            continue
        real   = d.get("real_count", 0)
        fake   = d.get("fake_count", 0)
        size   = d.get("size_gb", 0.0)
        status = "OK" if d.get("status") == "complete" else "PARTIAL"
        total_real += real
        total_fake += fake
        total_size += size
        print("| {:<12} {:>6} {:>6} {:>7.1f}GB {:<10} |".format(
            name, real, fake, size, status))

    print("+" + sep + "+")
    print("| {:<12} {:>6} {:>6} {:>7.1f}GB {:<10} |".format(
        "TOTAL", total_real, total_fake, total_size, ""))
    print("+" + sep + "+\n")


# ── Experiment design note ────────────────────────────────────────────────────

EXPERIMENT_NOTE = """
======================================================
DATA READY -- YOUR EXPERIMENTS WILL RUN AS FOLLOWS
======================================================
Experiment 1 -- In-distribution baseline:
  Train: 120 Celeb-DF real + 120 Celeb-DF fake  (80%)
  Test:   30 Celeb-DF real +  30 Celeb-DF fake  (20%)

Experiment 2 -- Cross-dataset (primary finding):
  Train: 150 Celeb-DF real + 150 Celeb-DF fake (100%)
  Test:   50 TIMIT real    +  50 TIMIT fake    (100%)
  Model has NEVER seen TIMIT -- true generalization test.

Experiment 3 -- Reverse cross-dataset:
  Train:  50 TIMIT real    +  50 TIMIT fake    (100%)
  Test:  150 Celeb-DF real + 150 Celeb-DF fake (100%)

Note: Small dataset size means results may have higher
variance. This is acceptable for a research finding --
report confidence intervals alongside point estimates.
======================================================
"""


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download Celeb-DF v2 + DeepFake-TIMIT for cross-dataset eval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./dataset"),
        help="Root directory for all downloaded data (default: ./dataset)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the download plan without actually downloading anything.",
    )
    parser.add_argument(
        "--skip_timit",
        action="store_true",
        help="Skip DeepFake-TIMIT; download Celeb-DF v2 only.",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point: orchestrate the full download, prune, and report pipeline."""
    args = parse_args()
    output_dir: Path = args.output.resolve()

    logger = setup_logging(output_dir)
    logger.info("Output directory: %s", output_dir)

    if args.dry_run:
        print("\n[DRY RUN MODE] -- no files will be downloaded.\n")

    # 1. Dependencies
    if not args.dry_run:
        ensure_dependencies(logger)

    # 2. Disk space
    output_dir.mkdir(parents=True, exist_ok=True)
    check_disk_space(output_dir, logger)

    # 3. Download datasets
    dataset_info: dict[str, dict] = {}

    dataset_info["CelebDF"] = download_celebdf(output_dir, args.dry_run, logger)

    if not args.skip_timit:
        dataset_info["TIMIT"] = download_timit(output_dir, args.dry_run, logger)
    else:
        logger.info("Skipping TIMIT (--skip_timit flag set).")

    if args.dry_run:
        print("\n[DRY RUN] Plan complete. Re-run without --dry_run to execute.")
        print(EXPERIMENT_NOTE)
        return

    # 4. Save dataset_info.json
    info_path = output_dir / "dataset_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)
    logger.info("Saved dataset info -> %s", info_path)

    # 5. Summary table + experiment note
    print_summary(dataset_info)
    print(EXPERIMENT_NOTE)


if __name__ == "__main__":
    main()
