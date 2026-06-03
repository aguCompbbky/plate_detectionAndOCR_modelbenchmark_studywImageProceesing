"""
dataset.py — PlateDataset: loads images + JSON labels, filters invalids.
"""
import os
import json
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import (
    IMAGES_DIR, LABELS_DIR, INVALID_FILE,
    IMG_H, IMG_W, CHAR2IDX, ALPHABET, BLANK_IDX
)


# ─── Load valid samples ───────────────────────────────────────────────────────

def _load_invalid_set(path: str) -> set:
    """Return the set of stem names listed in invalid_plates.txt."""
    avoid = set()
    if not os.path.exists(path):
        return avoid
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stem = line.split()[0]               # e.g. "1001_1.json"
            avoid.add(os.path.splitext(stem)[0]) # e.g. "1001_1"
    return avoid


def build_dataset(images_dir: str = IMAGES_DIR,
                  labels_dir: str = LABELS_DIR,
                  invalid_file: str = INVALID_FILE):
    """
    Returns list of (image_path, plate_string) for every valid sample.
    Both images_dir and labels_dir must share the same stem names.
    """
    invalid_stems = _load_invalid_set(invalid_file)
    samples = []

    for fname in sorted(os.listdir(labels_dir)):
        if not fname.endswith('.json'):
            continue
        stem = os.path.splitext(fname)[0]
        if stem in invalid_stems:
            continue

        json_path = os.path.join(labels_dir, fname)
        with open(json_path, 'r') as jf:
            label = json.load(jf).get('plates', '').strip().upper()

        # Skip empty or non-decodable labels
        if not label:
            continue
        if any(c not in CHAR2IDX for c in label):
            continue

        # Match image (try .jpg, .png, .jpeg)
        img_path = None
        for ext in ('.jpg', '.jpeg', '.png'):
            candidate = os.path.join(images_dir, stem + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            continue

        samples.append((img_path, label))

    return samples


# ─── Augmentation helpers ─────────────────────────────────────────────────────

def _augment(img: Image.Image) -> Image.Image:

    return img


# ─── Dataset class ────────────────────────────────────────────────────────────

class PlateDataset(Dataset):
    """Character-level OCR dataset for Turkish licence plates."""

    # Shared normalisation stats (grayscale)
    _mean = 0.5
    _std  = 0.5

    def __init__(self, samples, augment: bool = False):
        """
        Args:
            samples: list of (image_path, plate_string)
            augment: whether to apply training augmentations
        """
        self.samples = samples
        self.augment = augment
        self._base_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[self._mean], std=[self._std]),
        ])

    # ── label encoding ──

    @staticmethod
    def encode_label(plate: str):
        """plate string → 1-D LongTensor of char indices."""
        return torch.LongTensor([CHAR2IDX[c] for c in plate])

    # ── item ──

    def __getitem__(self, idx):
        img_path, plate = self.samples[idx]

        # Load as grayscale
        img = Image.open(img_path).convert('L')

        # Optional augmentation
        if self.augment:
            img = _augment(img)

        # Resize to fixed H × W
        img = img.resize((IMG_W, IMG_H), Image.BILINEAR)

        # To tensor + normalise → shape [1, H, W]
        img_tensor = self._base_tf(img)

        label_tensor = self.encode_label(plate)
        return img_tensor, label_tensor, plate

    def __len__(self):
        return len(self.samples)


# ─── Collate function ─────────────────────────────────────────────────────────

def collate_fn(batch):
    """
    Pads labels to the same length for batch processing.
    Returns:
        images   : FloatTensor [B, 1, H, W]
        labels   : LongTensor  [sum(label_lens)]  (concatenated, CTC-style)
        label_lens: LongTensor [B]
        plates   : list[str]
    """
    images, labels, plates = zip(*batch)
    images = torch.stack(images, 0)          # [B, 1, H, W]
    label_lens = torch.LongTensor([len(l) for l in labels])
    labels_cat = torch.cat(labels, 0)        # [sum of label lengths]
    return images, labels_cat, label_lens, list(plates)
