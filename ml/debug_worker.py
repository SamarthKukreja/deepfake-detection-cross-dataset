"""
Simulate exactly what the pool worker does for one video.
Run: python debug_worker.py
"""
import sys
import os
sys.path.insert(0, ".")

VIDEO = r"..\dataset\FF++\fake\Deepfakes_413_372.mp4"
LOG   = r"..\faces\debug_worker.log"
OUT   = r"..\faces\exp1_ffpp_to_ffpp\train\fake"

if __name__ == "__main__":
    from multiprocessing import Pool
    from extract_faces import _worker_init, _worker_task

    BATCH_SIZE = 8  # match the real run
    context = {"experiment": "exp1_ffpp_to_ffpp", "split": "train", "label": "fake", "dataset": "FF++"}
    task = (VIDEO, OUT, LOG, context)

    print(f"Simulating pool worker on: {VIDEO}")
    print(f"Worker device: cuda  batch_size: {BATCH_SIZE}")

    with Pool(
        processes=1,
        initializer=_worker_init,
        initargs=(LOG, "cuda", BATCH_SIZE),
    ) as pool:
        result = pool.apply(_worker_task, args=(task,))

    vid_path, n_faces, records = result
    print(f"\nFaces saved : {n_faces}")
    print(f"Records     : {len(records)}")
    if records:
        for r in records[:5]:
            print(f"  confidence={r['mtcnn_confidence']:.3f}  frame={r['frame_index']}")

    if os.path.exists(LOG) and os.path.getsize(LOG) > 0:
        print(f"\nWorker log:")
        with open(LOG) as f:
            print(f.read())
    else:
        print("\nWorker log: empty (no errors)")
