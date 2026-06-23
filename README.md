# Robust Cross-Dataset Deepfake Detection

### A Comparison of CNN and Transformer Backbones

## Motivation

Most published deepfake detectors are evaluated on the same dataset they were trained on,
inflating reported accuracy. A detector that only works on FaceForensics++ is not useful
in the real world where the manipulation method is unknown. This project asks a harder
question: **does a model trained on one dataset generalise to an entirely different one?**

We train five backbone architectures under an identical recipe across four
cross-dataset experiments, using identity-disjoint splits throughout so that no person's
face appears in both train and test.

---

## Research Questions

1. Which backbone generalises best across datasets — a classic CNN, a modern CNN, or a Vision Transformer?
2. Does the choice of architecture matter more than training-set scale and diversity?
3. How robust are the detectors to common image degradations (JPEG, blur, noise, downscaling)?

---

## Experiment Protocol

| Experiment | Train           | Test     | Setting                  |
| ---------- | --------------- | -------- | ------------------------ |
| Exp 1      | FF++            | FF++     | In-distribution          |
| Exp 2      | FF++            | Celeb-DF | Cross-dataset            |
| Exp 3      | Celeb-DF        | FF++     | Cross-dataset            |
| Exp 4      | FF++ + Celeb-DF | DFD      | Cross-dataset (held-out) |

**Identity-disjoint splits.** Exp 1 uses the official FaceForensics++ id partition
(720 / 140 / 140). A fake clip `Method_Target_Source` is assigned to a split only if both
the target and source identities belong to that split, so no person appears in both
train and test. DFD videos are fully held out for Exp 4 and never seen during training.
Cross-dataset experiments (Exp 2 / 3 / 4) are identity-disjoint by construction.

Verify Exp 1 leakage is zero:

```bash
python verify_exp1_split.py   # asserts train∩test == 0  and  val∩test == 0
```

---

## Backbones

All five models are loaded from `timm` and fine-tuned from ImageNet weights.

| Model           | timm name                | Category                            |
| --------------- | ------------------------ | ----------------------------------- |
| EfficientNet-B4 | `efficientnet_b4`      | Literature-standard baseline        |
| Xception        | `xception`             | Classic deepfake detection baseline |
| ResNet-50       | `resnet50`             | Strong classical CNN                |
| ConvNeXt-Tiny   | `convnext_tiny`        | Modern CNN                          |
| ViT-Base        | `vit_base_patch16_224` | Vision Transformer                  |

Every backbone shares an identical training recipe so that differences in results are
attributable to architecture alone, not hyper-parameter tuning.

---

## Training Recipe

| Component               | Choice                            | Rationale                             |
| ----------------------- | --------------------------------- | ------------------------------------- |
| Optimiser               | AdamW, lr = 1e-4, wd = 1e-4       | Standard for fine-tuning              |
| LR schedule             | Cosine annealing                  | Smooth decay without step-tuning      |
| Progressive unfreezing  | Unfreeze one stage every 3 epochs | Prevents catastrophic forgetting      |
| Label smoothing         | ε = 0.05                         | Improves cross-dataset generalisation |
| Mixed precision         | AMP fp16                          | Fits T4 VRAM budget                   |
| Gradient accumulation   | ×4 (effective batch 64)          | Stable gradient estimates             |
| Class balance           | WeightedRandomSampler             | Real / fake imbalance in Exp 4        |
| Early stopping          | Val AUC, patience = 5             | Avoids wasted compute                 |
| Test-time augmentation  | 5 views, mean pooling             | ~1–2 % AUC gain                      |
| Video-level aggregation | Mean of face-crop scores          | One prediction per source video       |
| Confidence intervals    | Bootstrap, 1000 re-samples        | Over videos, not crops                |

---

## Results (video-level AUC)

| Backbone                | Exp 1 | Exp 2 | Exp 3 | Exp 4           |
| ----------------------- | ----- | ----- | ----- | --------------- |
| EfficientNet-B4         | 0.901 | 0.788 | ~0.51 | 0.788           |
| Xception                | —    | 0.735 | —    | —              |
| ResNet-50               | —    | 0.812 | —    | —              |
| **ConvNeXt-Tiny** | —    | —    | —    | **0.915** |
| ViT-Base                | —    | 0.713 | —    | —              |

Full tables with 95 % bootstrap CIs are in `results_bundle/`.

**Key findings:**

- **ConvNeXt-Tiny is the strongest backbone overall**, leading on Exp 1, Exp 2, and Exp 4.
  ResNet-50 is a close second; both outperform ViT-Base consistently.
- **ViT-Base does not lead.** Under this leakage-controlled, modest-data regime the
  convolutional inductive bias is advantageous over the attention mechanism.
- **Celeb-DF → FF++ (Exp 3) collapses to near chance (~0.51–0.59) for every backbone.**
  Training on the smaller, more homogeneous Celeb-DF does not generalise to FF++'s varied
  manipulations. Training-set scale and diversity dominate cross-dataset behaviour more
  than architecture choice — answering research question 2.
- **Noise is the most damaging corruption** (AUC drops to ~0.49 under heavy noise on
  the DFD detector), followed by blur, downscaling, and JPEG compression.

---

## Pipeline

### 1 — Prepare the dataset

```
dataset/
├── FF++/    real/  fake/
├── CelebDF/ real/  fake/
└── DFD/     real/  fake/
```

```bash
python organize_experiments.py --dataset_dir ./dataset --output ./experiments
python verify_exp1_split.py
```

### 2 — Extract faces

```bash
cd ml
python extract_faces.py --exp all \
    --manifest ../experiments/experiment_manifests.json \
    --base ../ --output ../faces
```

### 3 — Train

```bash
# One experiment, one backbone
python train.py --exp exp1 --model_preset efficientnet_b4 \
    --faces ../faces --output ../results_bundle

# All experiments × all backbones (run on Kaggle T4)
python run_all.py --exp all --faces ../faces --results ../results_bundle
```

### 4 — Evaluate

```bash
python evaluate.py --exp all --faces ../faces --results ../results_bundle
```

### 5 — Robustness study

```bash
python robustness.py --exp all --faces ../faces --results ../results_bundle
```

### 6 — Generate paper tables

```bash
cd ..
python _gen_results_section.py   # writes research paper/results_section.tex
```

---

## Cloud Training (Kaggle)

The full study (5 backbones × 4 experiments + ablations ≈ 40 jobs) was run on Kaggle
free-tier T4 GPUs. `cloud/kaggle_run.ipynb` provides:

- Recursive auto-detection of the uploaded dataset zip (walks to depth 4)
- Guard cell that aborts on identity leakage or wrong dataset
- Auto-patch cell for PyTorch 2.6 compatibility (`weights_only=False`, LR factor fix, ViT isinstance guard)
- Resumable runner that skips already-finished jobs

---

## Repository Structure

```
ml/
  train.py            # backbone training
  evaluate.py         # TTA, video-level metrics, bootstrap CIs, Grad-CAM
  robustness.py       # eval-only corruption study
  extract_faces.py    # MTCNN face extraction
  dataset.py          # DeepfakeDataset + dataloaders
  run_all.py          # orchestration (all experiments × backbones)
  app.py              # Streamlit demo (see below)
  requirements.txt

cloud/
  kaggle_run.ipynb    # Kaggle T4 notebook

experiments/
  experiment_manifests.json
  ffpp_official_splits/        # official FF++ id partition files

results_bundle/                # metrics JSON per run (weights gitignored)
  {exp}__{backbone}__full/
    metrics.json
    robustness.json
    training_history.json

organize_experiments.py
verify_exp1_split.py
_gen_results_section.py        # generates results_section.tex from results_bundle/
```

Large artifacts (`dataset/`, `faces/`, `*.pth`, `results_bundle/**/plots/`) are
gitignored and rebuilt via the pipeline above.

---

## Demo App

An interactive Streamlit app (`ml/app.py`) lets you test the trained models on any video.

```bash
cd ml
streamlit run app.py
```

Upload a local video file, or paste a link from YouTube, Instagram, TikTok, Twitter/X,
or any of the 1000+ sites supported by `yt-dlp` — the stream is read directly without
downloading. All five checkpoints run simultaneously; MTCNN detects faces per frame and
draws colour-coded bounding boxes (green = real, red = fake). A vote-based ensemble
produces the final verdict:

| Sensitivity | Votes needed |
| ----------- | ------------ |
| Normal      | 3 / 5 models |
| Strict      | 2 / 5 models |
| Very Strict | 1 / 5 models |
