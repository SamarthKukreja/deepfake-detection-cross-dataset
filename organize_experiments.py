#!/usr/bin/env python3
"""
Create leakage-safe train/val/test manifests for the deepfake experiments.

Expected input layout:
    dataset/
        FF++/
            real/
            fake/
        CelebDF/
            real/
            fake/
        DFD/
            real/
            fake/

The script writes experiments/experiment_manifests.json without copying videos.
All experiments include a validation split from the training/source domain, so
training never has to use the external test set for early stopping.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

SEED = 42
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def load_files(folder: Path) -> List[str]:
    """Return sorted video paths under a class folder."""
    if not folder.exists():
        raise FileNotFoundError(f"Directory not found: {folder}")
    return sorted(
        str(p) for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def shuffled(files: List[str]) -> List[str]:
    out = files.copy()
    random.shuffle(out)
    return out


def downsample(files: List[str], n: int) -> List[str]:
    if len(files) < n:
        raise ValueError(f"Cannot sample {n} files from only {len(files)} files.")
    return random.sample(files, n)


def split_80_10_10(files: List[str]) -> Tuple[List[str], List[str], List[str]]:
    f = shuffled(files)
    n = len(f)
    t1 = int(n * 0.8)
    t2 = int(n * 0.9)
    return f[:t1], f[t1:t2], f[t2:]


def split_90_10(files: List[str]) -> Tuple[List[str], List[str]]:
    f = shuffled(files)
    n = len(f)
    t = int(n * 0.9)
    return f[:t], f[t:]


def rel_paths(paths: List[str], base: Path) -> List[str]:
    """Store paths relative to the workspace/base directory."""
    out = []
    for raw in paths:
        p = Path(raw)
        try:
            out.append(str(p.relative_to(base)))
        except ValueError:
            out.append(str(p))
    return out


def count_split(exp: Dict) -> Dict:
    counts = {}
    for split, classes in exp.items():
        if split == "description":
            continue
        counts[split] = {label: len(paths) for label, paths in classes.items()}
    return counts


# ──────────────────────────────────────────────────────────────────────────────
# IDENTITY-DISJOINT FF++ SPLIT (Fix C1)
#
# FF++ reals are named by identity id (e.g. 035.mp4). Classic FF++ fakes are
# named Method_TARGET_SOURCE (e.g. Deepfakes_035_036.mp4). Splitting reals and
# fakes independently at random leaks the same person across train/val/test.
# Here we partition the 1000 FF++ identities into train/val/test ID sets (using
# the OFFICIAL FaceForensics++ split lists) and assign:
#   • a REAL video to a split iff its id is in that split's ID set;
#   • a classic FAKE Method_T_S to a split iff BOTH T and S are in the SAME set.
# DeepFakeDetection_* fakes use 2-digit actor ids outside the FF++ 000–999 space
# and belong to the held-out DFD set, so they are dropped from Exp1.
# ──────────────────────────────────────────────────────────────────────────────

CLASSIC_FFPP_METHODS = (
    "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter",
)


def real_identity(path: str) -> str:
    """FF++ real filename 'NNN.mp4' → identity id 'NNN'."""
    return Path(path).stem


def classic_fake_identities(path: str):
    """Return (target, source) ids for a classic FF++ fake, else None.

    Returns None for DeepFakeDetection_* (held-out DFD; 2-digit actor ids) and
    anything not matching Method_<digits>_<digits>.
    """
    stem = Path(path).stem
    for method in CLASSIC_FFPP_METHODS:
        prefix = method + "_"
        if stem.startswith(prefix):
            parts = stem[len(prefix):].split("_")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                return parts[0], parts[1]
            return None
    return None


def load_official_splits(splits_dir: Path) -> Tuple[Dict[str, str], Dict[str, set]]:
    """Load official FF++ train/val/test.json → (id→split map, split→id-set)."""
    id_to_split: Dict[str, str] = {}
    sets: Dict[str, set] = {}
    for name in ("train", "val", "test"):
        with open(Path(splits_dir) / f"{name}.json", encoding="utf-8") as f:
            pairs = json.load(f)
        ids = set()
        for pair in pairs:
            for vid in pair:
                ids.add(str(vid))
        sets[name] = ids
        for vid in ids:
            id_to_split[vid] = name
    # sanity: disjoint
    assert not (sets["train"] & sets["val"]), "official train/val overlap"
    assert not (sets["train"] & sets["test"]), "official train/test overlap"
    assert not (sets["val"] & sets["test"]), "official val/test overlap"
    return id_to_split, sets


def build_identity_disjoint_ffpp(
    dataset_dir: Path,
    id_to_split: Dict[str, str],
    seed: int,
) -> Tuple[Dict[str, list], Dict[str, list], Dict[str, int]]:
    """Partition FF++ reals + classic fakes into identity-disjoint splits.

    Fakes are downsampled per split to match that split's real count (keeping
    the existing balance convention). Returns (reals, fakes, stats).
    """
    rng = random.Random(seed)
    real_files = load_files(dataset_dir / "FF++" / "real")
    fake_files = load_files(dataset_dir / "FF++" / "fake")

    reals: Dict[str, list] = {"train": [], "val": [], "test": []}
    for p in real_files:
        sp = id_to_split.get(real_identity(p))
        if sp:
            reals[sp].append(p)

    fakes_all: Dict[str, list] = {"train": [], "val": [], "test": []}
    dropped_dfd = 0
    dropped_cross = 0
    for p in fake_files:
        ids = classic_fake_identities(p)
        if ids is None:
            dropped_dfd += 1
            continue
        t, s = ids
        st, ss = id_to_split.get(t), id_to_split.get(s)
        if st is not None and st == ss:
            fakes_all[st].append(p)
        else:
            dropped_cross += 1

    fakes: Dict[str, list] = {}
    for sp in ("train", "val", "test"):
        n_real = len(reals[sp])
        pool = sorted(fakes_all[sp])
        keep = min(n_real, len(pool))
        fakes[sp] = sorted(rng.sample(pool, keep)) if keep else []
        if len(pool) < n_real:
            print(f"  WARN {sp}: only {len(pool)} classic fakes for {n_real} reals")

    stats = {
        "dropped_dfd_fakes": dropped_dfd,
        "dropped_cross_partition_fakes": dropped_cross,
        **{f"{sp}_real": len(reals[sp]) for sp in ("train", "val", "test")},
        **{f"{sp}_fake": len(fakes[sp]) for sp in ("train", "val", "test")},
    }
    return reals, fakes, stats


def regenerate_exp1_only(
    manifest_path: Path,
    dataset_dir: Path,
    base: Path,
    splits_dir: Path,
    seed: int,
) -> None:
    """Replace ONLY experiment_1 with an identity-disjoint split; preserve 2/3/4."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    id_to_split, sets = load_official_splits(splits_dir)
    print("Official FF++ id sets: "
          f"train={len(sets['train'])} val={len(sets['val'])} test={len(sets['test'])}")

    reals, fakes, stats = build_identity_disjoint_ffpp(dataset_dir, id_to_split, seed)
    print("Exp1 identity-disjoint composition:", stats)

    exp1 = {
        "description": (
            "In-distribution baseline: identity-disjoint FF++ split using the "
            "official FaceForensics++ id partition (train/val/test). A real is "
            "placed by its id; a classic fake Method_T_S only if both T and S are "
            "in the same split. DeepFakeDetection (DFD) fakes are excluded — they "
            "are the held-out test set and use a separate 2-digit actor id space."
        ),
        "train": {"real": reals["train"], "fake": fakes["train"]},
        "val": {"real": reals["val"], "fake": fakes["val"]},
        "test": {"real": reals["test"], "fake": fakes["test"]},
    }
    for split, classes in exp1.items():
        if split == "description":
            continue
        for label, paths in classes.items():
            classes[label] = rel_paths(paths, base)

    manifest["experiment_1"] = exp1
    notes = manifest.setdefault("notes", {})
    notes["exp1_identity_split"] = (
        "Exp1 re-split identity-disjoint (Fix C1) using official FaceForensics++ "
        "splits in experiments/ffpp_official_splits/. Reals by id; classic fakes "
        "kept only if both target+source ids share the split; DeepFakeDetection/DFD "
        "fakes dropped (held-out). Seed 42 for fake downsampling. exp2/3/4 untouched. "
        f"Counts: {stats}."
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"experiment_1 regenerated in-place -> {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create leakage-safe deepfake experiment manifests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir", type=Path, default=Path("./dataset"))
    parser.add_argument("--output", type=Path, default=Path("./experiments"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--exp1-only",
        dest="exp1_only",
        action="store_true",
        help="Regenerate ONLY experiment_1 (identity-disjoint FF++); preserve 2/3/4.",
    )
    parser.add_argument(
        "--official_splits",
        type=Path,
        default=Path("./experiments/ffpp_official_splits"),
        help="Directory with official FF++ train/val/test.json id splits.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base = dataset_dir.parent

    if args.exp1_only:
        manifest_path = output_dir / "experiment_manifests.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found — run a full generation first."
            )
        regenerate_exp1_only(
            manifest_path=manifest_path,
            dataset_dir=dataset_dir,
            base=base,
            splits_dir=args.official_splits.resolve(),
            seed=args.seed,
        )
        for key in ("experiment_1", "experiment_2", "experiment_3", "experiment_4"):
            with open(manifest_path, encoding="utf-8") as f:
                m = json.load(f)
            print(key, count_split(m[key]))
        return

    ffpp_real_all = load_files(dataset_dir / "FF++" / "real")
    ffpp_fake_all = load_files(dataset_dir / "FF++" / "fake")
    celeb_real_all = load_files(dataset_dir / "CelebDF" / "real")
    celeb_fake_all = load_files(dataset_dir / "CelebDF" / "fake")
    dfd_real_all = load_files(dataset_dir / "DFD" / "real")
    dfd_fake_all = load_files(dataset_dir / "DFD" / "fake")

    n_ffpp = len(ffpp_real_all)
    n_celeb = len(celeb_real_all)
    n_dfd = len(dfd_real_all)

    ffpp_real = shuffled(ffpp_real_all)
    ffpp_fake = shuffled(downsample(ffpp_fake_all, n_ffpp))
    celeb_real = shuffled(celeb_real_all)
    celeb_fake = shuffled(downsample(celeb_fake_all, n_celeb))
    dfd_real = shuffled(dfd_real_all)
    dfd_fake = shuffled(downsample(dfd_fake_all, n_dfd))

    ffpp_real_tr, ffpp_real_val, ffpp_real_te = split_80_10_10(ffpp_real)
    ffpp_fake_tr, ffpp_fake_val, ffpp_fake_te = split_80_10_10(ffpp_fake)

    celeb_real_tr, celeb_real_val = split_90_10(celeb_real)
    celeb_fake_tr, celeb_fake_val = split_90_10(celeb_fake)

    exp1 = {
        "description": "In-distribution baseline: train/val/test all from FF++",
        "train": {"real": ffpp_real_tr, "fake": ffpp_fake_tr},
        "val": {"real": ffpp_real_val, "fake": ffpp_fake_val},
        "test": {"real": ffpp_real_te, "fake": ffpp_fake_te},
    }

    exp2 = {
        "description": "Cross-dataset: train/val FF++, test CelebDF",
        "train": {"real": ffpp_real_tr, "fake": ffpp_fake_tr},
        "val": {"real": ffpp_real_val, "fake": ffpp_fake_val},
        "test": {"real": celeb_real, "fake": celeb_fake},
    }

    exp3 = {
        "description": "Reverse cross-dataset: train/val CelebDF, test FF++",
        "train": {"real": celeb_real_tr, "fake": celeb_fake_tr},
        "val": {"real": celeb_real_val, "fake": celeb_fake_val},
        "test": {"real": ffpp_real, "fake": ffpp_fake},
    }

    combined_real = shuffled(ffpp_real + celeb_real)
    combined_fake = shuffled(ffpp_fake + celeb_fake)
    comb_real_tr, comb_real_val, _ = split_80_10_10(combined_real)
    comb_fake_tr, comb_fake_val, _ = split_80_10_10(combined_fake)

    exp4 = {
        "description": "Mixed training/validation on FF+++CelebDF, test on unseen DFD",
        "train": {"real": comb_real_tr, "fake": comb_fake_tr},
        "val": {"real": comb_real_val, "fake": comb_fake_val},
        "test": {"real": dfd_real, "fake": dfd_fake},
    }

    experiments = {
        "seed": args.seed,
        "balance_strategy": "downsample_fake_to_real_count",
        "validation_protocol": "validation is always drawn from the training/source domain",
        "dataset_counts_after_balance": {
            "FF++": {"real": n_ffpp, "fake": n_ffpp},
            "CelebDF": {"real": n_celeb, "fake": n_celeb},
            "DFD": {"real": n_dfd, "fake": n_dfd},
        },
        "experiment_1": exp1,
        "experiment_2": exp2,
        "experiment_3": exp3,
        "experiment_4": exp4,
    }

    for key in ("experiment_1", "experiment_2", "experiment_3", "experiment_4"):
        for split, classes in experiments[key].items():
            if split == "description":
                continue
            for label, paths in classes.items():
                classes[label] = rel_paths(paths, base)

    out_path = output_dir / "experiment_manifests.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(experiments, f, indent=2)

    print(f"Manifest saved -> {out_path}")
    for key in ("experiment_1", "experiment_2", "experiment_3", "experiment_4"):
        print(key, count_split(experiments[key]))


if __name__ == "__main__":
    main()
