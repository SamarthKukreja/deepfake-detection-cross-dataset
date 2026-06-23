"""
Quick debug: show raw MTCNN confidence scores for a video.
Usage: python debug_detect.py <path_to_video>
"""
import sys
import cv2
import imageio
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image

VIDEO = sys.argv[1] if len(sys.argv) > 1 else None
if not VIDEO:
    print("Usage: python debug_detect.py <video_path>")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=True, device=device, post_process=False, select_largest=False)

print(f"Device: {device}")
print(f"Video : {VIDEO}")

# Read up to 20 sampled frames
frames = []
try:
    reader = imageio.get_reader(VIDEO, "ffmpeg")
    for i, rgb in enumerate(reader):
        if i % 15 == 0:
            bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
            frames.append((i, bgr))
        if len(frames) >= 20:
            break
    reader.close()
except Exception as e:
    print(f"imageio failed: {e}")

print(f"Sampled frames : {len(frames)}")
if not frames:
    print("No frames read — video unreadable.")
    sys.exit(1)

# Run MTCNN on each frame individually, print raw scores + blur variance
all_scores = []
blur_vars = []
for idx, bgr in frames:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_vars.append(var)
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    boxes, probs = mtcnn.detect(pil)
    if probs is not None:
        scores = [round(float(p), 3) for p in probs]
        all_scores.extend(scores)
        print(f"  frame {idx:4d}  blur={var:6.2f}  conf={scores}")
    else:
        print(f"  frame {idx:4d}  blur={var:6.2f}  no detection")

print(f"\nBlur variance range : {min(blur_vars):.2f} – {max(blur_vars):.2f}")
print(f"Below threshold 4.0 : {sum(v < 4.0 for v in blur_vars)}/{len(blur_vars)} frames filtered")

if all_scores:
    print(f"\nMax confidence : {max(all_scores):.3f}")
    print(f"Above 0.95     : {sum(s >= 0.95 for s in all_scores)}")
    print(f"Above 0.90     : {sum(s >= 0.90 for s in all_scores)}")
    print(f"Above 0.85     : {sum(s >= 0.85 for s in all_scores)}")
else:
    print("\nMTCNN found NO faces in any frame.")
