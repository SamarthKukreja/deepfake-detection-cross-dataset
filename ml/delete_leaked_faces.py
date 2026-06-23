"""
Deletes face crops for the 13 leaked videos from exp4 train/fake.
Run once from the code/ directory:
    python ml/delete_leaked_faces.py
"""
import glob
import os
from pathlib import Path

FACE_DIR = Path(__file__).parent.parent / "faces" / "exp4_mixed_to_dfd" / "train" / "fake"

LEAKED_STEMS = [
    "DeepFakeDetection_06_27__talking_angry_couch__JOG5PB18",
    "DeepFakeDetection_14_15__kitchen_pan__RF38N10R",
    "DeepFakeDetection_06_04__walking_outside_cafe_disgusted__ZK95PQDE",
    "DeepFakeDetection_17_08__exit_phone_room__CFIY1YEQ",
    "DeepFakeDetection_25_18__talking_against_wall__SEGFKFJG",
    "DeepFakeDetection_16_17__walking_outside_cafe_disgusted__S7UMSIQV",
    "DeepFakeDetection_24_22__kitchen_pan__XL557XC6",
    "DeepFakeDetection_01_04__walk_down_hall_angry__GBC7ZGDP",
    "DeepFakeDetection_28_16__walking_down_street_outside_angry__6DWLCU6T",
    "DeepFakeDetection_09_25__talking_angry_couch__5041ODBN",
    "DeepFakeDetection_20_13__hugging_happy__C50CSOEX",
    "DeepFakeDetection_23_19__walking_outside_cafe_disgusted__WHQ1229T",
    "DeepFakeDetection_07_13__walking_outside_cafe_disgusted__RVQCPCJF",
]

total = 0
for stem in LEAKED_STEMS:
    files = list(FACE_DIR.glob(f"{stem}_frame*_face*.jpg"))
    for f in files:
        f.unlink()
    print(f"  {stem}: deleted {len(files)} files")
    total += len(files)

print(f"\nDone. Total deleted: {total} face crops.")
