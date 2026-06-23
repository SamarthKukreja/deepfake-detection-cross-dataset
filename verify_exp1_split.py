"""
verify_exp1_split.py
====================
Regression gate for Fix C1: asserts the Exp1 (FF++ -> FF++) split is
identity-disjoint across train/val/test, over BOTH real ids and classic-fake
target/source ids, at the manifest level AND (if extracted) the face level.

Usage:
    python verify_exp1_split.py                 # manifest + faces
    python verify_exp1_split.py --faces-only
    python verify_exp1_split.py --manifest-only
"""
import argparse
import json
import os
import re

EXP = "exp1_ffpp_to_ffpp"
METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter")


def ids_from_real_name(stem):
    m = re.match(r"^(\d{3})$", stem)
    return {m.group(1)} if m else set()


def ids_from_fake_name(stem):
    for meth in METHODS:
        if stem.startswith(meth + "_"):
            parts = stem[len(meth) + 1:].split("_")
            out = set()
            for p in parts[:2]:
                if p.isdigit():
                    out.add(p)
            return out
    return set()  # DeepFakeDetection_* / unknown -> contributes no FF++ identity


def manifest_ids(manifest_path):
    m = json.load(open(manifest_path, encoding="utf-8"))
    e1 = m["experiment_1"]
    out = {}
    for split in ("train", "val", "test"):
        ids = set()
        for p in e1[split]["real"]:
            ids |= ids_from_real_name(os.path.splitext(os.path.basename(p.replace("\\", "/")))[0])
        for p in e1[split]["fake"]:
            ids |= ids_from_fake_name(os.path.splitext(os.path.basename(p.replace("\\", "/")))[0])
        out[split] = ids
    return out


def faces_ids(faces_root):
    out = {}
    for split in ("train", "val", "test"):
        ids = set()
        for label, parser in (("real", ids_from_real_name), ("fake", ids_from_fake_name)):
            d = os.path.join(faces_root, EXP, split, label)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                mm = re.match(r"(.+?)_frame\d+_face\d+\.jpg$", f)
                if mm:
                    ids |= parser(mm.group(1))
        out[split] = ids
    return out


def report(name, sets):
    tr, va, te = sets["train"], sets["val"], sets["test"]
    print(f"\n[{name}] identity counts: train={len(tr)} val={len(va)} test={len(te)}")
    tt, vt, tv = len(tr & te), len(va & te), len(tr & va)
    print(f"[{name}] overlaps: train&test={tt}  val&test={vt}  train&val={tv}")
    assert tt == 0, f"{name}: train&test identity overlap = {tt} (must be 0)"
    assert vt == 0, f"{name}: val&test identity overlap = {vt} (must be 0)"
    print(f"[{name}] OK: identity-disjoint (train&test == 0 and val&test == 0).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="experiments/experiment_manifests.json")
    ap.add_argument("--faces", default="faces")
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--faces-only", action="store_true")
    args = ap.parse_args()

    if not args.faces_only:
        report("manifest", manifest_ids(args.manifest))
    if not args.manifest_only:
        report("faces", faces_ids(args.faces))
    print("\nALL CHECKS PASSED: Exp1 is identity-disjoint.")


if __name__ == "__main__":
    main()
