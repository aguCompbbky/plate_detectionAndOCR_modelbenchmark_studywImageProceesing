"""
Main LPRNet Training Script
Trains LPRNet with 3 K-fold cross validation × 2 seeds.
Uses CTC loss with beam search decoding.
Saves models to models/ and evidence to evidence/.

Usage:
    cd /home/muk/Masaüstü/python_scripts/plaka2/LPRNet_train
    python3 train/train.py
"""

import os
import sys
import json
import time
import copy
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# ── Project root (LPRNet_train/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'train'))

from model import build_lprnet, NUM_CLASSES
from dataset import (
    PlateDataset, collate_fn, load_all_samples,
    get_sequential_kfold_splits, decode_indices, IMG_W, IMG_H
)
from beam_search import beam_search_decode_batch
from evaluate import (
    compute_metrics,
    save_confusion_matrix,
    save_training_curves,
    save_per_char_accuracy,
    save_summary_plots,
    save_metrics_report,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
IMG_DIR    = os.path.join(ROOT, 'lrp_newdata', 'images')
PLATE_DIR  = os.path.join(ROOT, 'lrp_newdata', 'plates')
MODEL_DIR  = os.path.join(ROOT, 'models')
RAW_DIR    = os.path.join(ROOT, 'raw_model')
EVIDENCE_DIR = os.path.join(ROOT, 'evidence')
SUMMARY_DIR  = os.path.join(EVIDENCE_DIR, 'summary')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

# ─── Hyperparameters ──────────────────────────────────────────────────────────
SEEDS       = [42, 123]
N_FOLDS     = 3
EPOCHS      = 100
BATCH_SIZE  = 8
LR          = 1e-3
BEAM_WIDTH  = 5
PATIENCE    = 200  # early stopping patience (disabled)
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__); import sys; logging.getLogger().handlers[0].stream = sys.stdout


# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# ─── CTC decode wrapper ───────────────────────────────────────────────────────

def decode_batch(log_probs):
    """Beam search decode (T, B, C) → list of plate strings."""
    idx_seqs = beam_search_decode_batch(log_probs, beam_width=BEAM_WIDTH)
    return [decode_indices(seq) for seq in idx_seqs]


# ─── One epoch train ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, ctc_loss, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for images, labels_concat, label_lengths, plates in loader:
        images = images.to(device)
        labels_concat = labels_concat.to(device)

        log_probs = model(images)   # (T, B, C)
        T, B, C = log_probs.shape
        input_lengths = torch.full((B,), T, dtype=torch.long, device=device)

        loss = ctc_loss(log_probs, labels_concat, input_lengths, label_lengths.to(device))
        if torch.isnan(loss) or torch.isinf(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ─── Validation ───────────────────────────────────────────────────────────────

def evaluate_epoch(model, loader, ctc_loss, device):
    """Evaluate one epoch. Returns loss and metrics dict."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    pred_plates_all = []
    true_plates_all = []

    with torch.no_grad():
        for images, labels_concat, label_lengths, plates in loader:
            images = images.to(device)
            labels_concat = labels_concat.to(device)

            log_probs = model(images)
            T, B, C = log_probs.shape
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)

            loss = ctc_loss(log_probs, labels_concat, input_lengths, label_lengths.to(device))
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                n_batches += 1

            preds = decode_batch(log_probs)
            pred_plates_all.extend(preds)
            true_plates_all.extend(plates)

    val_loss = total_loss / max(n_batches, 1)
    metrics = compute_metrics(pred_plates_all, true_plates_all)
    return val_loss, metrics, pred_plates_all, true_plates_all


# ─── Main Training Loop ───────────────────────────────────────────────────────

def train():
    log.info(f"Device: {DEVICE}")
    log.info(f"NUM_CLASSES={NUM_CLASSES}, BEAM_WIDTH={BEAM_WIDTH}")

    # ── 1. Save raw pretrained model (freshly initialized) ───────────────────
    raw_path = os.path.join(RAW_DIR, 'lprnet_pretrained.pth')
    if not os.path.exists(raw_path):
        log.info("Saving raw (randomly initialized) LPRNet to raw_model/...")
        raw_model = build_lprnet(num_classes=NUM_CLASSES)
        torch.save(raw_model.state_dict(), raw_path)
        log.info(f"  Saved → {raw_path}")
    else:
        log.info(f"Raw model already exists: {raw_path}")

    # ── 2. Load all valid samples ─────────────────────────────────────────────
    all_samples = load_all_samples(IMG_DIR, PLATE_DIR)
    n_samples = len(all_samples)
    log.info(f"Total valid samples: {n_samples}")

    # CTC Loss
    ctc_loss_fn = nn.CTCLoss(blank=NUM_CLASSES - 1, reduction='mean', zero_infinity=True)

    # Experiment config
    experiment_config = {
        'n_folds': N_FOLDS,
        'seeds': SEEDS,
        'beam_width': BEAM_WIDTH,
        'epochs': EPOCHS,
        'lr': LR,
        'batch_size': BATCH_SIZE,
        'optimizer': 'Adam',
        'loss': 'CTCLoss',
        'input_size': [3, IMG_H, IMG_W],
        'num_classes': NUM_CLASSES,
        'early_stopping_patience': PATIENCE,
    }

    all_results = []

    # ── 3. Training loop: Seeds × Folds ──────────────────────────────────────
    for seed in SEEDS:
        log.info(f"\n{'='*60}")
        log.info(f"SEED = {seed}")
        log.info(f"{'='*60}")

        set_seed(seed)
        splits = get_sequential_kfold_splits(n_samples, n_splits=N_FOLDS, seed=seed)

        for fold, (train_idx, val_idx) in enumerate(splits):
            log.info(f"\n  ── Fold {fold} | train={len(train_idx)}, val={len(val_idx)}")

            set_seed(seed + fold)  # slight variation per fold

            # Datasets
            train_ds = PlateDataset(IMG_DIR, PLATE_DIR, train_idx, augment=True)
            val_ds   = PlateDataset(IMG_DIR, PLATE_DIR, val_idx,   augment=False)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                      shuffle=True, num_workers=0,
                                      collate_fn=collate_fn, drop_last=True)
            val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                                      shuffle=False, num_workers=0,
                                      collate_fn=collate_fn)

            # Model
            model = build_lprnet(num_classes=NUM_CLASSES, pretrained_path=raw_path)
            model = model.to(DEVICE)

            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)

            # History
            history = {
                'train_loss': [], 'val_loss': [],
                'val_plate_acc': [], 'val_char_acc': [], 'val_cer': [],
                'val_precision': [], 'val_recall': [], 'val_f1': [],
            }

            best_plate_acc = 0.0
            best_state = None
            no_improve = 0
            best_epoch = 0
            best_metrics = {}

            t0 = time.time()
            for epoch in range(1, EPOCHS + 1):
                tr_loss = train_one_epoch(model, train_loader, optimizer, ctc_loss_fn, DEVICE)
                val_loss, val_metrics, preds, trues = evaluate_epoch(
                    model, val_loader, ctc_loss_fn, DEVICE)

                scheduler.step(val_loss)

                history['train_loss'].append(tr_loss)
                history['val_loss'].append(val_loss)
                history['val_plate_acc'].append(val_metrics['plate_accuracy'])
                history['val_char_acc'].append(val_metrics['char_accuracy'])
                history['val_cer'].append(val_metrics['cer'])
                history['val_precision'].append(val_metrics['precision'])
                history['val_recall'].append(val_metrics['recall'])
                history['val_f1'].append(val_metrics['f1_score'])

                plate_acc = val_metrics['plate_accuracy']
                elapsed = time.time() - t0
                log.info(
                    f"  Epoch {epoch:3d}/{EPOCHS} | "
                    f"tr_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | "
                    f"plate_acc={plate_acc:.4f} | cer={val_metrics['cer']:.4f} | "
                    f"t={elapsed:.0f}s"
                )

                # Best model tracking
                if plate_acc > best_plate_acc:
                    best_plate_acc = plate_acc
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    best_metrics = val_metrics
                    best_preds = preds
                    best_trues = trues
                    no_improve = 0
                else:
                    no_improve += 1

                # Early stopping
                if no_improve >= PATIENCE:
                    log.info(f"  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                    break

            # Save best model
            model_fname = f'lprnet_seed{seed}_fold{fold}.pth'
            model_path = os.path.join(MODEL_DIR, model_fname)
            if best_state is not None:
                torch.save(best_state, model_path)
            else:
                torch.save(model.state_dict(), model_path)
            log.info(f"  Model saved → {model_path}")

            # Load best state for final eval
            if best_state is not None:
                model.load_state_dict(best_state)

            # ── Evidence generation ──────────────────────────────────────────
            cm = best_metrics.get('confusion_matrix', None)
            if cm is not None:
                save_confusion_matrix(cm, seed, fold, EVIDENCE_DIR, normalize=False)
                save_confusion_matrix(cm, seed, fold, EVIDENCE_DIR, normalize=True)
                save_per_char_accuracy(cm, seed, fold, EVIDENCE_DIR)

            save_training_curves(history, seed, fold, EVIDENCE_DIR)

            # Collect result
            result = {
                'seed': seed,
                'fold': fold,
                'best_epoch': best_epoch,
                'plate_accuracy': best_metrics.get('plate_accuracy', 0.0),
                'char_accuracy': best_metrics.get('char_accuracy', 0.0),
                'cer': best_metrics.get('cer', 1.0),
                'precision': best_metrics.get('precision', 0.0),
                'recall': best_metrics.get('recall', 0.0),
                'f1_score': best_metrics.get('f1_score', 0.0),
                'confusion_matrix': cm,
                'training_history': {
                    k: [float(x) for x in v] for k, v in history.items()
                },
                'model_path': model_path,
            }
            all_results.append(result)

            log.info(
                f"  ✓ Fold {fold} done | best_epoch={best_epoch} | "
                f"plate_acc={result['plate_accuracy']:.4f} | "
                f"cer={result['cer']:.4f} | f1={result['f1_score']:.4f}"
            )

    # ── 4. Save summary evidence ──────────────────────────────────────────────
    log.info("\nGenerating summary plots and metrics report...")
    save_summary_plots(all_results, SUMMARY_DIR)

    # Strip confusion_matrix from JSON (too large)
    results_for_json = []
    for r in all_results:
        r2 = {k: v for k, v in r.items() if k != 'confusion_matrix'}
        results_for_json.append(r2)

    report_path = save_metrics_report(results_for_json, experiment_config, SUMMARY_DIR)
    log.info(f"Metrics report saved → {report_path}")

    # ── 5. Final summary ──────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("TRAINING COMPLETE")
    log.info("="*60)
    for r in all_results:
        log.info(
            f"  Seed={r['seed']} Fold={r['fold']} | "
            f"PlateAcc={r['plate_accuracy']:.4f} | "
            f"CharAcc={r['char_accuracy']:.4f} | "
            f"CER={r['cer']:.4f} | F1={r['f1_score']:.4f}"
        )

    means = {
        k: float(np.mean([r[k] for r in all_results]))
        for k in ['plate_accuracy', 'char_accuracy', 'cer', 'f1_score']
    }
    log.info(f"\nCV Mean → {means}")
    log.info(f"\nModels saved to : {MODEL_DIR}")
    log.info(f"Evidence saved to: {EVIDENCE_DIR}")


if __name__ == '__main__':
    train()
