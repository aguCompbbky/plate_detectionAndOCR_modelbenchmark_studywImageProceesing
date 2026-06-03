"""
Dataset and K-Fold splits for LPRNet training.
Turkish license plate dataset loader with sequential K-fold.
"""

import os
import re
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# ─── Character Set ───────────────────────────────────────────────────────────
CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U',
    'V', 'Y', 'Z'
]
CHARS_DICT = {c: i for i, c in enumerate(CHARS)}
BLANK_IDX = len(CHARS)  # CTC blank token index
NUM_CLASSES = len(CHARS) + 1  # 34

# Turkish license plate regex
TR_PLATE_RE = re.compile(
    r'^(0[1-9]|[1-7][0-9]|8[01])'
    r'(([A-Z])(\d{4,5})'
    r'|([A-Z]{2})(\d{3,4})'
    r'|([A-Z]{3})(\d{2}))$'
)

# Image dimensions expected by LPRNet
IMG_W, IMG_H = 94, 24


def is_valid_plate(plate_str):
    """Check if a plate matches Turkish plate format."""
    return bool(TR_PLATE_RE.match(plate_str))


def encode_label(plate_str):
    """Encode plate string to list of character indices."""
    return [CHARS_DICT[c] for c in plate_str if c in CHARS_DICT]


def decode_indices(indices):
    """Decode list of character indices to string."""
    return ''.join(CHARS[i] for i in indices if 0 <= i < len(CHARS))


class PlateDataset(Dataset):
    """
    Dataset for license plate images and labels.
    Loads from lrp_newdata/images/ and lrp_newdata/plates/.
    """

    def __init__(self, img_dir, plate_dir, index_list, augment=False):
        """
        Args:
            img_dir: path to images directory
            plate_dir: path to plates (JSON) directory
            index_list: list of sample indices to use
            augment: whether to apply augmentation (training)
        """
        self.img_dir = img_dir
        self.plate_dir = plate_dir
        self.augment = augment

        # Build full file list
        all_samples = self._load_all_samples()
        self.samples = [all_samples[i] for i in index_list]

        # Transforms
        self.transform = self._build_transform(augment)

    def _load_all_samples(self):
        """Load all (image_path, plate_str) pairs, filtering invalids."""
        samples = []
        img_files = sorted(os.listdir(self.img_dir))
        for img_file in img_files:
            if not img_file.endswith('.jpg'):
                continue
            stem = img_file.replace('.jpg', '')
            json_file = stem + '.json'
            json_path = os.path.join(self.plate_dir, json_file)
            img_path = os.path.join(self.img_dir, img_file)

            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                plate = data.get('plates', '').strip().upper()
            except Exception:
                continue

            if not plate or not all(c in CHARS_DICT for c in plate):
                continue
            if len(plate) < 5 or len(plate) > 10:
                continue

            samples.append((img_path, plate))
        return samples

    def _build_transform(self, augment=False):
        ops = [
            transforms.Resize((IMG_H, IMG_W)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
        if augment:
            ops = [
                transforms.Resize((IMG_H, IMG_W)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]
        return transforms.Compose(ops)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, plate = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(encode_label(plate), dtype=torch.long)
        return img, label, plate


def collate_fn(batch):
    """Custom collate for variable-length labels (CTC)."""
    images, labels, plates = zip(*batch)
    images = torch.stack(images, dim=0)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_concat = torch.cat(labels, dim=0)
    return images, labels_concat, label_lengths, plates


def load_all_samples(img_dir, plate_dir):
    """Return all valid samples as list of (img_path, plate_str)."""
    samples = []
    img_files = sorted(os.listdir(img_dir))
    for img_file in img_files:
        if not img_file.endswith('.jpg'):
            continue
        stem = img_file.replace('.jpg', '')
        json_path = os.path.join(plate_dir, stem + '.json')
        img_path = os.path.join(img_dir, img_file)
        if not os.path.exists(json_path):
            continue
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            plate = data.get('plates', '').strip().upper()
        except Exception:
            continue
        if not plate or not all(c in CHARS_DICT for c in plate):
            continue
        if len(plate) < 5 or len(plate) > 10:
            continue
        samples.append((img_path, plate))
    return samples


def get_sequential_kfold_splits(n_samples, n_splits=3, seed=42):
    """
    Sequential K-fold: shuffle once, then split into N equal parts.
    Returns list of (train_indices, val_indices) per fold.

    Fold 0: first  1/3 → val, rest → train
    Fold 1: middle 1/3 → val, rest → train
    Fold 2: last   1/3 → val, rest → train
    """
    rng = np.random.RandomState(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    fold_size = n_samples // n_splits
    splits = []

    for fold in range(n_splits):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_splits - 1 else n_samples
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        splits.append((train_idx.tolist(), val_idx.tolist()))

    return splits
