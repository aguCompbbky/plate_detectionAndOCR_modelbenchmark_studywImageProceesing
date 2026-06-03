"""
Evaluation and Evidence Generation Module.
Computes all scientific metrics and generates plots for each fold/seed run.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from dataset import CHARS, decode_indices


# ─── Metric Helpers ──────────────────────────────────────────────────────────

def levenshtein(a, b):
    """Compute Levenshtein edit distance between two strings."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    dp = np.zeros((la + 1, lb + 1), dtype=int)
    dp[:, 0] = np.arange(la + 1)
    dp[0, :] = np.arange(lb + 1)
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return dp[la, lb]


def compute_metrics(pred_plates, true_plates):
    """
    Compute all character and plate level metrics.

    Returns dict with:
        plate_accuracy, char_accuracy, cer,
        precision, recall, f1, confusion_matrix_data
    """
    n = len(true_plates)
    assert n == len(pred_plates), "Length mismatch"

    plate_correct = 0
    total_chars = 0
    correct_chars = 0
    total_edit = 0
    total_ref_len = 0

    # For per-char confusion matrix
    true_chars_all = []
    pred_chars_all = []

    for pred, true in zip(pred_plates, true_plates):
        # Plate-level
        if pred == true:
            plate_correct += 1

        # CER (Levenshtein)
        edit = levenshtein(pred, true)
        total_edit += edit
        total_ref_len += len(true)

        # Per-character (align by min length for confusion matrix)
        for pc, tc in zip(pred, true):
            if pc in CHARS and tc in CHARS:
                pred_chars_all.append(CHARS.index(pc))
                true_chars_all.append(CHARS.index(tc))

        # Char accuracy (considering aligned length)
        min_len = min(len(pred), len(true))
        for pc, tc in zip(pred, true):
            if pc == tc:
                correct_chars += 1
        total_chars += max(len(pred), len(true))

    plate_acc = plate_correct / n
    char_acc = correct_chars / total_chars if total_chars > 0 else 0.0
    cer = total_edit / total_ref_len if total_ref_len > 0 else 0.0

    # Confusion matrix
    num_chars = len(CHARS)
    if true_chars_all and pred_chars_all:
        cm = confusion_matrix(true_chars_all, pred_chars_all,
                              labels=list(range(num_chars)))
    else:
        cm = np.zeros((num_chars, num_chars), dtype=int)

    # Precision, Recall, F1 (macro, per-character)
    if true_chars_all and pred_chars_all:
        prec, rec, f1, _ = precision_recall_fscore_support(
            true_chars_all, pred_chars_all,
            labels=list(range(num_chars)),
            average='macro', zero_division=0
        )
    else:
        prec = rec = f1 = 0.0

    return {
        'plate_accuracy': float(plate_acc),
        'char_accuracy': float(char_acc),
        'cer': float(cer),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
    }


# ─── Plot Helpers ─────────────────────────────────────────────────────────────

def save_confusion_matrix(cm, seed, fold, out_dir, normalize=False):
    """Save confusion matrix plot."""
    cm_arr = np.array(cm)
    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_arr = cm_arr.astype(float) / row_sums
        fmt = '.2f'
        title = f'Normalized Confusion Matrix (Seed={seed}, Fold={fold})'
        fname = f'confusion_matrix_normalized_seed{seed}_fold{fold}.png'
        vmax = 1.0
    else:
        fmt = 'd'
        title = f'Confusion Matrix (Seed={seed}, Fold={fold})'
        fname = f'confusion_matrix_seed{seed}_fold{fold}.png'
        vmax = None

    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(cm_arr, ax=ax, annot=True, fmt=fmt,
                xticklabels=CHARS, yticklabels=CHARS,
                cmap='Blues', vmin=0, vmax=vmax,
                linewidths=0.3, linecolor='gray',
                cbar_kws={'shrink': 0.8})
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def save_training_curves(history, seed, fold, out_dir):
    """Save training/validation loss and accuracy curves."""
    epochs = list(range(1, len(history['train_loss']) + 1))

    # ── Loss curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history['train_loss'], label='Train Loss', color='steelblue', linewidth=2)
    ax.plot(epochs, history['val_loss'], label='Val Loss', color='tomato', linewidth=2)
    ax.set_title(f'Training & Validation Loss (Seed={seed}, Fold={fold})', fontsize=13)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('CTC Loss')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path_loss = os.path.join(out_dir, f'training_loss_curve_seed{seed}_fold{fold}.png')
    fig.savefig(path_loss, dpi=150)
    plt.close(fig)

    # ── Plate accuracy curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history['val_plate_acc'],
            label='Val Plate Acc', color='seagreen', linewidth=2)
    ax.plot(epochs, history['val_char_acc'],
            label='Val Char Acc', color='mediumpurple', linewidth=2, linestyle='--')
    ax.set_title(f'Validation Accuracy (Seed={seed}, Fold={fold})', fontsize=13)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    path_acc = os.path.join(out_dir, f'validation_accuracy_curve_seed{seed}_fold{fold}.png')
    fig.savefig(path_acc, dpi=150)
    plt.close(fig)

    # ── CER curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history['val_cer'],
            label='Val CER', color='darkorange', linewidth=2)
    ax.set_title(f'Character Error Rate (Seed={seed}, Fold={fold})', fontsize=13)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('CER')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path_cer = os.path.join(out_dir, f'cer_curve_seed{seed}_fold{fold}.png')
    fig.savefig(path_cer, dpi=150)
    plt.close(fig)

    # ── Precision / Recall / F1 curve
    if 'val_precision' in history:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, history['val_precision'], label='Precision', color='royalblue', linewidth=2)
        ax.plot(epochs, history['val_recall'], label='Recall', color='coral', linewidth=2)
        ax.plot(epochs, history['val_f1'], label='F1-Score', color='gold', linewidth=2, linestyle='--')
        ax.set_title(f'Precision / Recall / F1 (Seed={seed}, Fold={fold})', fontsize=13)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        path_prf = os.path.join(out_dir, f'precision_recall_f1_seed{seed}_fold{fold}.png')
        fig.savefig(path_prf, dpi=150)
        plt.close(fig)


def save_per_char_accuracy(cm, seed, fold, out_dir):
    """Save per-character accuracy bar chart."""
    cm_arr = np.array(cm)
    char_totals = cm_arr.sum(axis=1)
    char_corrects = np.diag(cm_arr)
    char_acc = np.where(char_totals > 0, char_corrects / char_totals, 0.0)

    fig, ax = plt.subplots(figsize=(16, 5))
    colors = ['seagreen' if a >= 0.9 else ('goldenrod' if a >= 0.7 else 'tomato')
              for a in char_acc]
    bars = ax.bar(CHARS, char_acc, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_title(f'Per-Character Accuracy (Seed={seed}, Fold={fold})', fontsize=13)
    ax.set_xlabel('Character')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.9, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='90% threshold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    for bar, acc in zip(bars, char_acc):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                    f'{acc:.2f}', ha='center', va='bottom', fontsize=7, rotation=90)
    plt.tight_layout()
    path = os.path.join(out_dir, f'per_character_accuracy_seed{seed}_fold{fold}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_summary_plots(all_results, summary_dir):
    """Save cross-validation summary comparison plots."""
    os.makedirs(summary_dir, exist_ok=True)

    seeds = sorted(set(r['seed'] for r in all_results))
    folds = sorted(set(r['fold'] for r in all_results))
    n_folds = len(folds)

    metrics = ['plate_accuracy', 'char_accuracy', 'cer', 'precision', 'recall', 'f1_score']
    labels = ['Plate Acc', 'Char Acc', 'CER', 'Precision', 'Recall', 'F1']
    colors = ['steelblue', 'seagreen', 'tomato', 'royalblue', 'coral', 'gold']

    # ── Per-seed per-fold comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for mi, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        ax = axes[mi]
        x = np.arange(n_folds)
        width = 0.35
        for si, seed in enumerate(seeds):
            vals = [next(r[metric] for r in all_results
                         if r['seed'] == seed and r['fold'] == f) for f in folds]
            ax.bar(x + si * width, vals, width, label=f'Seed {seed}', color=color,
                   alpha=0.7 + 0.3 * si, edgecolor='white')
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([f'Fold {f}' for f in folds])
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        if metric != 'cer':
            ax.set_ylim(0, 1)

    plt.suptitle('Cross-Validation Results: All Seeds & Folds', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(summary_dir, 'overall_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # ── CV summary mean ± std
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    means = [np.mean([r[m] for r in all_results]) for m in metrics]
    stds = [np.std([r[m] for r in all_results]) for m in metrics]
    bars = ax.bar(x, means, color=colors, alpha=0.8, edgecolor='white', capsize=5)
    ax.errorbar(x, means, yerr=stds, fmt='none', color='black', capsize=5, linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title('Cross-Validation Mean ± Std (All Seeds & Folds)', fontsize=13)
    ax.set_ylabel('Score')
    ax.grid(axis='y', alpha=0.3)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01,
                f'{m:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    path2 = os.path.join(summary_dir, 'cross_validation_summary.png')
    fig.savefig(path2, dpi=150)
    plt.close(fig)

    return path, path2


def save_metrics_report(all_results, experiment_config, summary_dir):
    """Save complete metrics_report.json."""
    os.makedirs(summary_dir, exist_ok=True)

    metrics_keys = ['plate_accuracy', 'char_accuracy', 'cer',
                    'precision', 'recall', 'f1_score']

    means = {k: float(np.mean([r[k] for r in all_results])) for k in metrics_keys}
    stds = {k: float(np.std([r[k] for r in all_results])) for k in metrics_keys}

    # Best model
    best = max(all_results, key=lambda r: r['plate_accuracy'])

    report = {
        'experiment_config': experiment_config,
        'results': all_results,
        'cross_validation_mean': means,
        'cross_validation_std': stds,
        'best_run': {
            'seed': best['seed'],
            'fold': best['fold'],
            'plate_accuracy': best['plate_accuracy'],
            'model_path': best.get('model_path', ''),
        }
    }

    path = os.path.join(summary_dir, 'metrics_report.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    best_path = os.path.join(summary_dir, 'best_model_info.json')
    with open(best_path, 'w', encoding='utf-8') as f:
        json.dump(report['best_run'], f, indent=2)

    return path
