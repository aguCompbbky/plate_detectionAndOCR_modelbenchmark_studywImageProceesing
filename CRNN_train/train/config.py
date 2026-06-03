"""
config.py — CRNN licence-plate OCR hyperparameters and paths.
"""
import os
import torch

# ─── Hyperparameters ─────────────────────────────────────────────────────────
SEEDS       = [42, 123]
N_FOLDS     = 3
EPOCHS      = 100
BATCH_SIZE  = 8
LR          = 1e-3
BEAM_WIDTH  = 5
PATIENCE    = 200          # early-stopping patience (effectively disabled)
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── Image pre-processing ────────────────────────────────────────────────────
IMG_H = 32
IMG_W = 128

# ─── Character alphabet ──────────────────────────────────────────────────────
# Turkish plates: digits 0-9, letters A-Z (no I, Q, X in official alphabet
# but we include every character found in the dataset for safety)
ALPHABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
BLANK_CHAR  = '-'                          # CTC blank token
BLANK_IDX   = 0                            # blank is always index 0
CHAR2IDX = {c: i + 1 for i, c in enumerate(ALPHABET)}
CHAR2IDX[BLANK_CHAR] = BLANK_IDX
IDX2CHAR = {v: k for k, v in CHAR2IDX.items()}
NUM_CLASSES = len(ALPHABET) + 1           # +1 for blank

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, 'crnn_newdata')
IMAGES_DIR    = os.path.join(DATA_DIR, 'images')
LABELS_DIR    = os.path.join(DATA_DIR, 'plates')
INVALID_FILE  = os.path.join(DATA_DIR, 'invalid_plates.txt')
MODELS_DIR    = os.path.join(BASE_DIR, 'models')
PURE_DIR      = os.path.join(BASE_DIR, 'pure_model')
EVIDENCE_DIR  = os.path.join(BASE_DIR, 'evidence')

for _d in (MODELS_DIR, PURE_DIR, EVIDENCE_DIR):
    os.makedirs(_d, exist_ok=True)
