"""
trainer.py — K-fold training loop with full reproducibility.

Outer loop: for seed in SEEDS → for fold in range(N_FOLDS)
Each run:
  - set_seed(seed)
  - build train/val split from pre-computed fold indices
  - train EPOCHS epochs
  - save best model (lowest val CER)
  - log history (loss, CER, WER, exact accuracy, LR per epoch)
"""
import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from config import (
    SEEDS, N_FOLDS, EPOCHS, BATCH_SIZE, LR, PATIENCE,
    DEVICE, MODELS_DIR, BEAM_WIDTH
)
from dataset  import PlateDataset, collate_fn, build_dataset
from model    import build_model
from decoder  import beam_search_decode
from metrics  import (compute_cer, compute_wer, compute_exact_match,
                      compute_char_map, plot_confusion_matrix,
                      plot_training_curves)


# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─── One epoch helpers ────────────────────────────────────────────────────────

def _train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, labels, label_lens, _ in loader:
        images     = images.to(device)
        labels     = labels.to(device)
        label_lens = label_lens.to(device)

        log_probs = model(images)                 # [T, B, C]
        T, B, C   = log_probs.shape
        input_lens = torch.full((B,), T, dtype=torch.long, device=device)

        loss = criterion(log_probs, labels, input_lens, label_lens)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * B

    return total_loss / len(loader.dataset)


@torch.no_grad()
def _val_epoch(model, loader, criterion, device, beam_width=BEAM_WIDTH):
    model.eval()
    total_loss = 0.0
    all_preds, all_gts = [], []

    for images, labels, label_lens, plates in loader:
        images     = images.to(device)
        labels     = labels.to(device)
        label_lens = label_lens.to(device)

        log_probs = model(images)
        T, B, C   = log_probs.shape
        input_lens = torch.full((B,), T, dtype=torch.long, device=device)

        loss = criterion(log_probs, labels, input_lens, label_lens)
        total_loss += loss.item() * B

        preds = beam_search_decode(log_probs, beam_width=beam_width)
        all_preds.extend(preds)
        all_gts.extend(plates)

    avg_loss  = total_loss / len(loader.dataset)
    cer       = compute_cer(all_preds, all_gts)
    wer       = compute_wer(all_preds, all_gts)
    exact_acc = compute_exact_match(all_preds, all_gts)
    return avg_loss, cer, wer, exact_acc, all_preds, all_gts


# ─── Core training function ───────────────────────────────────────────────────

def train_one_run(samples, train_idx, val_idx, seed: int, fold: int) -> dict:
    """
    Train CRNN for one (seed, fold) combination.
    Returns history dict with per-epoch metrics.
    """
    tag = f'seed{seed}_fold{fold}'
    print(f'\n{"="*60}')
    print(f'[trainer] START  seed={seed}  fold={fold}  device={DEVICE}')
    print(f'{"="*60}')

    set_seed(seed)

    # ── DataLoaders ─────────────────────────────────────────────────────────
    _samples = [samples[i] for i in train_idx]
    _val_s   = [samples[i] for i in val_idx]

    train_ds = PlateDataset(_samples, augment=True)
    val_ds   = PlateDataset(_val_s,   augment=False)

    g = torch.Generator()
    g.manual_seed(seed)
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=2, pin_memory=True,
                              worker_init_fn=seed_worker,
                              generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn,
                              num_workers=2, pin_memory=True)

    # ── Model, optimiser, scheduler ─────────────────────────────────────────
    model     = build_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # CosineAnnealing so that LR is nicely tracked
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    # ── History ─────────────────────────────────────────────────────────────
    history = {
        'train_loss': [], 'val_loss': [],
        'val_cer':    [], 'val_wer': [],
        'val_exact':  [], 'lr': [],
    }

    best_cer      = float('inf')
    best_state    = None
    patience_ctr  = 0

    # ── Epoch loop ──────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        train_loss = _train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, cer, wer, exact_acc, _, _ = _val_epoch(
            model, val_loader, criterion, DEVICE)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_cer'].append(cer)
        history['val_wer'].append(wer)
        history['val_exact'].append(exact_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f'  [Epoch {epoch:3d}/{EPOCHS}]  '
                  f'TrainLoss={train_loss:.4f}  ValLoss={val_loss:.4f}  '
                  f'CER={cer:.4f}  WER={wer:.4f}  ExactAcc={exact_acc:.4f}  '
                  f'LR={current_lr:.2e}')

        # Best model tracking
        if cer < best_cer:
            best_cer   = cer
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= PATIENCE:
            print(f'  [trainer] Early stop at epoch {epoch}')
            break

    # ── Save best model ─────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, f'{tag}_best.pth')
    torch.save({'model_state': best_state,
                'seed': seed, 'fold': fold,
                'best_val_cer': best_cer}, model_path)
    print(f'[trainer] Saved best model → {model_path}  (CER={best_cer:.4f})')

    # ── Final eval on val set for evidence ──────────────────────────────────
    model.load_state_dict(best_state)
    _, _, _, _, all_preds, all_gts = _val_epoch(
        model, val_loader, criterion, DEVICE)

    # Confusion matrix + training curves
    plot_confusion_matrix(all_preds, all_gts, tag=tag)
    plot_training_curves(history, tag=tag)

    # Per-character mAP
    char_map = compute_char_map(all_preds, all_gts)

    return {
        'tag':       tag,
        'seed':      seed,
        'fold':      fold,
        'best_val_cer':   best_cer,
        'final_val_wer':  compute_wer(all_preds, all_gts),
        'final_val_exact':compute_exact_match(all_preds, all_gts),
        'char_map':  char_map,
        'history':   history,
    }


# ─── K-Fold orchestrator ─────────────────────────────────────────────────────

def run_kfold_training(samples: list) -> list[dict]:
    """
    Outer loop: seed × fold.
    Returns list of per-run result dicts.
    """
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_splits = list(kf.split(samples))   # fixed splits

    all_results = []

    for seed in SEEDS:
        for fold, (train_idx, val_idx) in enumerate(fold_splits):
            result = train_one_run(samples, train_idx, val_idx, seed, fold)
            all_results.append(result)

    # ── Aggregated confusion matrix (all seeds + folds combined) ─────────────
    print('\n[trainer] Building aggregated confusion matrix …')
    all_run_preds, all_run_gts = [], []
    for result in all_results:
        # Re-run inference quickly to collect preds (stored in history)
        # We use per-run stored data reconstructed from history
        pass  # will be handled in train.py where we accumulate preds

    return all_results
