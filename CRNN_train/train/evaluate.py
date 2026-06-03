"""
evaluate.py — Script to independently test a trained CRNN model.

It loads a specific model checkpoint (e.g. models/seed42_fold0_best.pth),
reconstructs the exact validation set using the same K-Fold split,
and runs evaluation using Beam Search, printing out metrics and sample mismatches.

Usage:
    python train/evaluate.py --seed 42 --fold 0
"""
import sys
import os
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

# Allow running as `python train/evaluate.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import N_FOLDS, DEVICE, MODELS_DIR, BEAM_WIDTH
from dataset import build_dataset, PlateDataset, collate_fn
from model import build_model
from metrics import (
    compute_cer, compute_wer, compute_exact_match,
    compute_char_map, plot_confusion_matrix
)
from trainer import _val_epoch, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a specific trained CRNN model.")
    parser.add_argument('--seed', type=int, required=True, help="Seed of the trained model (e.g., 42)")
    parser.add_argument('--fold', type=int, required=True, help="Fold index of the trained model (e.g., 0)")
    parser.add_argument('--beam_width', type=int, default=BEAM_WIDTH, help="Beam width for decoding")
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed
    fold = args.fold
    beam = args.beam_width

    # ── Verify Model Exists ───────────────────────────────────────────────────
    model_name = f'seed{seed}_fold{fold}_best.pth'
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        print(f"[Error] Checkpoint not found: {model_path}")
        sys.exit(1)

    print(f"=== CRNN Evaluation ===")
    print(f"Model       : {model_name}")
    print(f"Device      : {DEVICE}")
    print(f"Beam Width  : {beam}")

    # ── Reproduce the Validation Split ────────────────────────────────────────
    set_seed(42)  # Global seed for shuffling K-Fold deterministically
    samples = build_dataset()

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_splits = list(kf.split(samples))

    if fold >= len(fold_splits):
        print(f"[Error] Fold {fold} is out of bounds (N_FOLDS={N_FOLDS}).")
        sys.exit(1)

    _, val_idx = fold_splits[fold]
    val_samples = [samples[i] for i in val_idx]

    print(f"Total Val Samples: {len(val_samples)}")

    # ── Load Model ────────────────────────────────────────────────────────────
    model = build_model().to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # ── DataLoader ────────────────────────────────────────────────────────────
    val_ds = PlateDataset(val_samples, augment=False)
    
    # We must seed the dataloader workers identically
    g = torch.Generator()
    g.manual_seed(seed)
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        import numpy as np
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False,
                            collate_fn=collate_fn, num_workers=2,
                            worker_init_fn=seed_worker, generator=g)

    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    print("\nRunning inference (Beam Search)...")
    # _val_epoch uses beam_search_decode internally
    avg_loss, cer, wer, exact_acc, all_preds, all_gts = _val_epoch(
        model, val_loader, criterion, DEVICE, beam_width=beam
    )

    # ── Compute Additional Metrics ────────────────────────────────────────────
    char_map = compute_char_map(all_preds, all_gts)

    print("\n=== Results ===")
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"CER            : {cer:.4f}  (Character Error Rate)")
    print(f"WER            : {wer:.4f}  (Word/Plate Error Rate)")
    print(f"Exact Accuracy : {exact_acc:.4f}  (Fully correct plates)")
    print(f"Global mAP     : {char_map.get('mAP', 0.0):.4f}")

    # ── Show sample of errors ─────────────────────────────────────────────────
    errors = [(gt, pred) for gt, pred in zip(all_gts, all_preds) if gt != pred]
    
    print(f"\nMisclassified Plates: {len(errors)} / {len(all_gts)}")
    if errors:
        print("Sample of mismatches (Ground Truth -> Predicted):")
        for gt, pred in errors[:10]:
            print(f"  {gt:<12} -> {pred}")

    # Optionally re-plot the confusion matrices specifically for this eval
    print(f"\nGenerating standalone evaluation confusion matrix for {model_name}...")
    plot_confusion_matrix(all_preds, all_gts, tag=f'eval_seed{seed}_fold{fold}')
    
    print("Done.")


if __name__ == '__main__':
    main()
