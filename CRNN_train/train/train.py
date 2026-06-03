"""
train.py — Main entry point for CRNN licence-plate OCR training.

Usage:
    python train/train.py             # full run (100 epochs × 3 folds × 2 seeds)
    python train/train.py --smoke     # 1 epoch smoke-test
"""
import sys
import os
import argparse

# Allow running as `python train/train.py` from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from sklearn.model_selection import KFold

from config import (
    SEEDS, N_FOLDS, EPOCHS, DEVICE,
    PURE_DIR, MODELS_DIR, EVIDENCE_DIR
)
from dataset  import build_dataset, PlateDataset, collate_fn
from model    import build_model
from trainer  import run_kfold_training, set_seed, _val_epoch
from metrics  import (compute_char_map,
                      plot_confusion_matrix, save_metrics_report)
import torch.nn as nn


# ─── Arguments ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true',
                   help='Smoke test: 1 epoch, skip evidence save')
    return p.parse_args()


# ─── Save pure (untrained) model ─────────────────────────────────────────────

def save_pure_model():
    """Saves an untrained CRNN to pure_model/crnn_untrained.pth."""
    set_seed(42)
    model = build_model()
    path  = os.path.join(PURE_DIR, 'crnn_untrained.pth')
    torch.save(model.state_dict(), path)
    print(f'[train] Pure (untrained) model saved → {path}')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    smoke = args.smoke

    if smoke:
        print('[train] *** SMOKE TEST MODE (1 epoch) ***')
        import config as cfg
        cfg.EPOCHS  = 1
        cfg.PATIENCE = 9999

    print(f'[train] Device  : {DEVICE}')
    print(f'[train] Seeds   : {SEEDS}')
    print(f'[train] Folds   : {N_FOLDS}')
    print(f'[train] Epochs  : {EPOCHS}')

    # 1. Save pure model
    save_pure_model()

    # 2. Build dataset (excludes invalid plates)
    print('[train] Loading dataset …')
    samples = build_dataset()
    print(f'[train] Valid samples: {len(samples)}')
    if len(samples) == 0:
        raise RuntimeError('No valid samples found — check paths in config.py')

    # 3. K-fold training (all seeds)
    all_results = run_kfold_training(samples)

    if smoke:
        print('[train] Smoke test finished — skipping full evidence save.')
        return

    # 4. Aggregated confusion matrix across all runs
    print('\n[train] Building aggregated confusion matrix …')
    # Re-run inference on each val split with best model
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_splits = list(kf.split(samples))
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    global_preds, global_gts = [], []
    for result in all_results:
        seed = result['seed']
        fold = result['fold']
        tag  = result['tag']

        set_seed(seed)
        _, val_idx = fold_splits[fold]
        val_samples = [samples[i] for i in val_idx]
        val_ds      = PlateDataset(val_samples, augment=False)
        from torch.utils.data import DataLoader
        val_loader  = DataLoader(val_ds, batch_size=8, shuffle=False,
                                 collate_fn=collate_fn, num_workers=2)

        model_path = os.path.join(MODELS_DIR, f'{tag}_best.pth')
        ckpt  = torch.load(model_path, map_location=DEVICE)
        model = build_model().to(DEVICE)
        model.load_state_dict(ckpt['model_state'])

        _, _, _, _, preds, gts = _val_epoch(model, val_loader, criterion, DEVICE)
        global_preds.extend(preds)
        global_gts.extend(gts)

    plot_confusion_matrix(global_preds, global_gts, tag='aggregated')
    global_char_map = compute_char_map(global_preds, global_gts)

    # 5. Build and save metrics_report.json
    report = {
        'device':    str(DEVICE),
        'seeds':     SEEDS,
        'n_folds':   N_FOLDS,
        'epochs':    EPOCHS,
        'beam_width': 5,
        'total_samples': len(samples),
        'runs': [],
        'global': {
            'char_map': global_char_map,
        }
    }

    for result in all_results:
        # Exclude heavy history from JSON — store summary only
        history = result.pop('history', {})
        entry = {
            'tag':             result['tag'],
            'seed':            result['seed'],
            'fold':            result['fold'],
            'best_val_cer':    result['best_val_cer'],
            'final_val_wer':   result['final_val_wer'],
            'final_val_exact': result['final_val_exact'],
            'char_map':        result['char_map'],
            'history_summary': {
                'min_train_loss': min(history.get('train_loss', [0])),
                'min_val_cer':    min(history.get('val_cer',    [0])),
                'min_val_wer':    min(history.get('val_wer',    [0])),
                'max_val_exact':  max(history.get('val_exact',  [0])),
                'final_lr':       history['lr'][-1] if history.get('lr') else None,
                'min_lr':         min(history.get('lr', [0])),
                'max_lr':         max(history.get('lr', [0])),
                'lr_per_epoch':   history.get('lr', []),
                'train_loss_per_epoch': history.get('train_loss', []),
                'val_loss_per_epoch':   history.get('val_loss',   []),
                'val_cer_per_epoch':    history.get('val_cer',    []),
                'val_wer_per_epoch':    history.get('val_wer',    []),
                'val_exact_per_epoch':  history.get('val_exact',  []),
            },
        }
        report['runs'].append(entry)

    # Cross-run averages
    all_cers   = [r['best_val_cer']    for r in report['runs']]
    all_wers   = [r['final_val_wer']   for r in report['runs']]
    all_exacts = [r['final_val_exact'] for r in report['runs']]
    report['global']['mean_val_cer']   = float(np.mean(all_cers))
    report['global']['std_val_cer']    = float(np.std(all_cers))
    report['global']['mean_val_wer']   = float(np.mean(all_wers))
    report['global']['mean_val_exact'] = float(np.mean(all_exacts))
    report['global']['mAP']            = global_char_map.get('mAP', 0.0)

    save_metrics_report(report)

    print('\n[train] ─── DONE ───')
    print(f"  Mean Val CER  : {report['global']['mean_val_cer']:.4f}")
    print(f"  Mean Val WER  : {report['global']['mean_val_wer']:.4f}")
    print(f"  Mean Exact Acc: {report['global']['mean_val_exact']:.4f}")
    print(f"  Global mAP    : {report['global']['mAP']:.4f}")


if __name__ == '__main__':
    main()
