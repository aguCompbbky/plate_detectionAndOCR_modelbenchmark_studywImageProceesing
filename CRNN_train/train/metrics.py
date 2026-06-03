"""
metrics.py — CER, WER, per-character mAP, confusion matrices, training curves.
All results are saved to evidence/.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    confusion_matrix
)

from config import EVIDENCE_DIR, ALPHABET


# ─── String-level metrics ─────────────────────────────────────────────────────

def _edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_cer(preds: list[str], gts: list[str]) -> float:
    """Character Error Rate: sum(edit_dist) / sum(len(gt))."""
    total_dist, total_len = 0, 0
    for p, g in zip(preds, gts):
        total_dist += _edit_distance(p, g)
        total_len  += len(g)
    return total_dist / max(total_len, 1)


def compute_wer(preds: list[str], gts: list[str]) -> float:
    """Word Error Rate: fraction of plates decoded incorrectly (exact match)."""
    errors = sum(1 for p, g in zip(preds, gts) if p != g)
    return errors / max(len(gts), 1)


def compute_exact_match(preds: list[str], gts: list[str]) -> float:
    """Exact sequence accuracy."""
    correct = sum(1 for p, g in zip(preds, gts) if p == g)
    return correct / max(len(gts), 1)


# ─── Character-level mAP ─────────────────────────────────────────────────────

def compute_char_map(
        all_preds: list[str],
        all_gts:   list[str],
        alphabet:  str = ALPHABET) -> dict:
    """
    Per-character average precision.
    For each character c in alphabet we treat it as binary:
      - ground-truth: each occurrence of c in the label plate
      - prediction:   each occurrence of c in the predicted plate
    (character-position aligned, zero-padded)
    Returns dict {char: AP, ..., 'mAP': mean_AP}
    """
    char_ap = {}
    for c in alphabet:
        y_true, y_score = [], []
        for pred, gt in zip(all_preds, all_gts):
            length = max(len(pred), len(gt))
            pred_p = pred.ljust(length)
            gt_p   = gt.ljust(length)
            for pp, gp in zip(pred_p, gt_p):
                y_true.append(1 if gp == c else 0)
                y_score.append(1 if pp == c else 0)

        if sum(y_true) == 0:
            char_ap[c] = float('nan')
            continue
        try:
            ap = average_precision_score(y_true, y_score, zero_division=0)
        except Exception:
            ap = float('nan')
        char_ap[c] = float(ap)

    valid_aps = [v for v in char_ap.values() if not np.isnan(v)]
    char_ap['mAP'] = float(np.mean(valid_aps)) if valid_aps else 0.0
    return char_ap


# ─── Confusion matrix ─────────────────────────────────────────────────────────

def _char_pairs(preds: list[str], gts: list[str]):
    """Yield (gt_char, pred_char) for every aligned position."""
    for pred, gt in zip(preds, gts):
        length = max(len(pred), len(gt))
        pred_p = pred.ljust(length, '_')
        gt_p   = gt.ljust(length, '_')
        for pp, gp in zip(pred_p, gt_p):
            if gp in ALPHABET:   # only real chars as target
                yield gp, pp


def _build_cm(preds: list[str], gts: list[str],
              labels: list[str]) -> np.ndarray:
    """Build confusion matrix for characters in `labels`."""
    gt_chars   = []
    pred_chars = []
    for gp, pp in _char_pairs(preds, gts):
        gt_chars.append(gp)
        # Unknown predictions → '_'
        pred_chars.append(pp if pp in labels else '_')

    extended = labels + (['_'] if '_' not in labels else [])
    cm = confusion_matrix(gt_chars, pred_chars, labels=extended)
    # Trim to only label rows (remove '_' row if present)
    n = len(labels)
    cm = cm[:n, :n]   # restrict to known chars
    return cm


def plot_confusion_matrix(
        preds: list[str], gts: list[str],
        tag: str,
        evidence_dir: str = EVIDENCE_DIR) -> None:
    """
    Saves:
      evidence/confusion_matrix_{tag}.png          (raw counts)
      evidence/confusion_matrix_normalized_{tag}.png (row-normalised)
    """
    labels = list(ALPHABET)
    cm = _build_cm(preds, gts, labels)

    _save_cm_figure(cm, labels, os.path.join(evidence_dir, f'confusion_matrix_{tag}.png'),
                    title=f'Confusion Matrix — {tag}', normalize=False)

    _save_cm_figure(cm, labels, os.path.join(evidence_dir, f'confusion_matrix_normalized_{tag}.png'),
                    title=f'Normalized Confusion Matrix — {tag}', normalize=True)


def _save_cm_figure(cm: np.ndarray, labels: list[str],
                    path: str, title: str, normalize: bool) -> None:
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        data = cm.astype(float) / row_sums
        fmt  = '.2f'
        vmin, vmax = 0.0, 1.0
    else:
        data = cm
        fmt  = 'd'
        vmin, vmax = None, None

    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                linewidths=0.3, linecolor='grey',
                vmin=vmin, vmax=vmax, ax=ax,
                annot_kws={'size': 6})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Ground Truth', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# ─── Training curves ─────────────────────────────────────────────────────────

def plot_training_curves(
        history: dict,
        tag: str,
        evidence_dir: str = EVIDENCE_DIR) -> None:
    """
    history keys: 'train_loss', 'val_loss', 'val_cer', 'val_wer',
                   'val_exact', 'lr'
    All are lists of length EPOCHS.
    Saves: evidence/training_curves_{tag}.png
    """
    keys     = ['train_loss', 'val_loss', 'val_cer', 'val_wer', 'val_exact', 'lr']
    labels   = ['Train Loss', 'Val Loss', 'Val CER', 'Val WER', 'Val Exact Acc', 'Learning Rate']
    colours  = ['#2196F3', '#F44336', '#FF9800', '#9C27B0', '#4CAF50', '#795548']
    n_plots  = len([k for k in keys if k in history and history[k]])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax_idx, (key, lbl, col) in enumerate(zip(keys, labels, colours)):
        if key not in history or not history[key]:
            axes[ax_idx].set_visible(False)
            continue
        epochs_range = range(1, len(history[key]) + 1)
        axes[ax_idx].plot(epochs_range, history[key], color=col, linewidth=1.5)
        if key == 'lr':
            axes[ax_idx].set_yscale('log')
        axes[ax_idx].set_xlabel('Epoch')
        axes[ax_idx].set_ylabel(lbl)
        axes[ax_idx].set_title(f'{lbl} — {tag}')
        axes[ax_idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(evidence_dir, f'training_curves_{tag}.png'), dpi=150)
    plt.close(fig)


# ─── Metrics report ───────────────────────────────────────────────────────────

def save_metrics_report(report: dict,
                         evidence_dir: str = EVIDENCE_DIR) -> None:
    """Save consolidated metrics_report.json."""
    path = os.path.join(evidence_dir, 'metrics_report.json')
    with open(path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'[metrics] Saved report → {path}')
