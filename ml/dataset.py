"""
dataset.py
==========
PyTorch Dataset and DataLoader utilities for deepfake detection training.

Reads extracted face images from:
    faces/
        {exp_name}/
            {split}/
                real/   ← label 0
                fake/   ← label 1

Usage
-----
    from dataset import get_dataloader

    train_loader = get_dataloader(
        faces_root="./faces",
        exp_name="exp1_ffpp_to_ffpp",
        split="train",
        batch_size=16,
        input_size=224,
        use_sampler=True,   # WeightedRandomSampler for balanced batches
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


# ──────────────────────────────────────────────────────────────────────────────
# TRANSFORMS
# ──────────────────────────────────────────────────────────────────────────────

# ImageNet normalisation — standard for pretrained models
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(
    split: str,
    input_size: int = 224,
    augment_train: bool = True,
) -> Callable:
    """
    Return torchvision transform pipeline for a given split.

    Train augmentations:
        - RandomHorizontalFlip (faces can be mirrored)
        - ColorJitter (compression/lighting variations across datasets)
        - RandomRotation ±10° (slight pose variation)
        - Random erasing (simulates occlusion, improves generalisation)

    Val / Test:
        - Resize + CenterCrop only (deterministic)

    Parameters
    ----------
    split      : 'train', 'val', or 'test'
    input_size : target H×W fed to the model (224 for EfficientNet-B4 on 2GB GPU)
    """
    if split == "train" and augment_train:
        return transforms.Compose([
            transforms.Resize((input_size + 20, input_size + 20)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
            ),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def get_tta_transforms(input_size: int = 224, n: int = 5) -> List[Callable]:
    """
    Return *n* deterministic transform pipelines for Test Time Augmentation.

    The first is always the clean centre-crop (identity augmentation).
    The rest are: h-flip, slight rotations (+5°, -5°), brightness shift.

    Parameters
    ----------
    input_size : model input size
    n          : number of augmentation variants (default 5)
    """
    base = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]

    variants = [
        transforms.Compose(base),  # clean — always first
        transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]),
        transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.functional.rotate if False else
            transforms.RandomRotation(degrees=(5, 5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]),
        transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomRotation(degrees=(-5, -5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]),
        transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ColorJitter(brightness=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]),
    ]
    return variants[:n]


# ──────────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    Face-image dataset for binary deepfake classification.

    Labels
    ------
    0 = real
    1 = fake

    Parameters
    ----------
    faces_root : path to /faces/ root folder
    exp_name   : e.g. 'exp1_ffpp_to_ffpp'
    split      : 'train', 'val', or 'test'
    transform  : torchvision transform (use get_transforms())
    """

    def __init__(
        self,
        faces_root: str | Path,
        exp_name: str,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        root = Path(faces_root) / exp_name / split
        for label_name, label_idx in [("real", 0), ("fake", 1)]:
            label_dir = root / label_name
            if not label_dir.exists():
                continue
            for img_path in sorted(label_dir.glob("*.jpg")):
                self.samples.append((img_path, label_idx))

        if not self.samples:
            raise FileNotFoundError(
                f"No face images found in {root}. "
                "Run extract_faces.py first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

    @property
    def labels(self) -> List[int]:
        """Return list of integer labels (0/1) — used by WeightedRandomSampler."""
        return [lbl for _, lbl in self.samples]

    def class_counts(self) -> Tuple[int, int]:
        """Return (n_real, n_fake)."""
        labels = self.labels
        return labels.count(0), labels.count(1)


# ──────────────────────────────────────────────────────────────────────────────
# DATALOADER FACTORY
# ──────────────────────────────────────────────────────────────────────────────

def get_dataloader(
    faces_root: str | Path,
    exp_name: str,
    split: str,
    batch_size: int = 16,
    input_size: int = 224,
    num_workers: int = 4,
    use_sampler: bool = False,
    augment_train: bool = True,
) -> DataLoader:
    """
    Build a DataLoader for one experiment split.

    Parameters
    ----------
    faces_root  : path to /faces/ root
    exp_name    : experiment folder name
    split       : 'train', 'val', or 'test'
    batch_size  : images per batch
    input_size  : resize target for model input
    num_workers : parallel data-loading workers
    use_sampler : if True, use WeightedRandomSampler to guarantee
                  balanced batches each epoch (train only)

    Returns
    -------
    torch.utils.data.DataLoader
    """
    transform = get_transforms(split, input_size, augment_train=augment_train)
    dataset = DeepfakeDataset(faces_root, exp_name, split, transform)

    sampler = None
    shuffle = split == "train"

    if use_sampler and split == "train":
        labels = dataset.labels
        n_real, n_fake = dataset.class_counts()
        # Weight inversely proportional to class frequency
        class_weights = {0: 1.0 / max(n_real, 1), 1: 1.0 / max(n_fake, 1)}
        sample_weights = [class_weights[lbl] for lbl in labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False  # mutually exclusive with sampler

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
    )
