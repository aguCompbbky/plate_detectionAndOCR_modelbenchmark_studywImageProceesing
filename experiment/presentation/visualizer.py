"""
visualizer.py — Matplotlib benchmark visualizations for pipeline comparison.

Produces:
  1. Accuracy comparison bar chart (exact match % per pipeline)
  2. CER comparison bar chart (lower is better)
  3. Detection rate bar chart
  4. Pipeline heatmap (detector × OCR for accuracy and CER)
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─── Color palette ────────────────────────────────────────────────────────────
_PALETTE = [
    '#4e79a7', '#f28e2b', '#e15759', '#76b7b2',
    '#59a14f', '#edc948', '#b07aa1', '#ff9da7',
]


def _pipeline_label(name: str) -> str:
    """Convert file stem like 'yolo_crnn' to 'YOLO + CRNN'."""
    parts = name.split('_')
    return ' + '.join(p.upper() for p in parts)


def plot_accuracy_comparison(benchmark_report: dict, output_path: str) -> None:
    """
    Bar chart of Exact Match Accuracy per pipeline.

    Args:
        benchmark_report: Output from benchmark.run_benchmark()
        output_path: Path to save the PNG figure
    """
    pipelines = list(benchmark_report.keys())
    accuracies = [benchmark_report[p]['exact_match_accuracy'] * 100 for p in pipelines]
    labels = [_pipeline_label(p) for p in pipelines]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(pipelines))]

    fig, ax = plt.subplots(figsize=(max(8, len(pipelines) * 1.4), 5))
    bars = ax.bar(labels, accuracies, color=colors, edgecolor='black', width=0.6)

    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title('Exact Match Accuracy per Pipeline', fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 110)
    ax.axhline(y=np.mean(accuracies), color='black', linestyle='--', linewidth=1.2,
               label=f'Mean: {np.mean(accuracies):.1f}%')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(labels, rotation=20, ha='right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[visualizer] Saved → {output_path}')


def plot_cer_comparison(benchmark_report: dict, output_path: str) -> None:
    """
    Bar chart of Mean CER per pipeline (lower = better).
    """
    pipelines = list(benchmark_report.keys())
    cers = [benchmark_report[p]['mean_cer'] * 100 for p in pipelines]
    labels = [_pipeline_label(p) for p in pipelines]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(pipelines))]

    fig, ax = plt.subplots(figsize=(max(8, len(pipelines) * 1.4), 5))
    bars = ax.bar(labels, cers, color=colors, edgecolor='black', width=0.6)

    for bar, val in zip(bars, cers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title('Mean Character Error Rate (CER) per Pipeline\n(lower is better)',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('CER (%)')
    ax.set_ylim(0, max(cers) * 1.3 + 5)
    ax.axhline(y=np.mean(cers), color='black', linestyle='--', linewidth=1.2,
               label=f'Mean: {np.mean(cers):.1f}%')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(labels, rotation=20, ha='right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[visualizer] Saved → {output_path}')


def plot_detection_rate(benchmark_report: dict, output_path: str) -> None:
    """
    Bar chart of detection rate (% of images where a plate was found).
    """
    pipelines = list(benchmark_report.keys())
    rates = [benchmark_report[p]['detection_rate'] * 100 for p in pipelines]
    labels = [_pipeline_label(p) for p in pipelines]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(pipelines))]

    fig, ax = plt.subplots(figsize=(max(8, len(pipelines) * 1.4), 5))
    bars = ax.bar(labels, rates, color=colors, edgecolor='black', width=0.6)

    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title('Plate Detection Rate per Pipeline', fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('Detection Rate (%)')
    ax.set_ylim(0, 115)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(labels, rotation=20, ha='right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[visualizer] Saved → {output_path}')


def plot_pipeline_matrix(benchmark_report: dict, output_path: str) -> None:
    """
    Heatmap: rows = detectors (YOLO, RCNN), cols = OCR models (CRNN, LPRNet, Paddle).
    Cell value = Exact Match Accuracy.
    """
    detectors = []
    ocr_models = []

    for name in benchmark_report.keys():
        parts = name.split('_')
        if len(parts) >= 2:
            det = parts[0].upper()
            ocr = '_'.join(parts[1:]).upper()
            if det not in detectors:
                detectors.append(det)
            if ocr not in ocr_models:
                ocr_models.append(ocr)

    if not detectors or not ocr_models:
        print('[visualizer] Not enough pipelines for heatmap (need detector_ocr naming)')
        return

    matrix_acc = np.full((len(detectors), len(ocr_models)), np.nan)
    matrix_cer = np.full((len(detectors), len(ocr_models)), np.nan)

    for name, metrics in benchmark_report.items():
        parts = name.split('_')
        if len(parts) < 2:
            continue
        det = parts[0].upper()
        ocr = '_'.join(parts[1:]).upper()
        if det in detectors and ocr in ocr_models:
            i = detectors.index(det)
            j = ocr_models.index(ocr)
            matrix_acc[i, j] = metrics['exact_match_accuracy'] * 100
            matrix_cer[i, j] = metrics['mean_cer'] * 100

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(ocr_models) * 3), max(4, len(detectors) * 1.8)))

    for ax, matrix, title, fmt in zip(
        axes,
        [matrix_acc, matrix_cer],
        ['Exact Match Accuracy (%)', 'Mean CER (%) — lower is better'],
        ['.1f', '.1f']
    ):
        masked = np.ma.masked_invalid(matrix)
        im = ax.imshow(masked, cmap='YlOrRd' if 'CER' in title else 'YlGn',
                       aspect='auto', vmin=0, vmax=100)
        ax.set_xticks(range(len(ocr_models)))
        ax.set_yticks(range(len(detectors)))
        ax.set_xticklabels(ocr_models, fontsize=11)
        ax.set_yticklabels(detectors, fontsize=11)
        ax.set_xlabel('OCR Model', fontsize=11)
        ax.set_ylabel('Detector', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(len(detectors)):
            for j in range(len(ocr_models)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:{fmt}}%', ha='center', va='center',
                            fontsize=11, fontweight='bold',
                            color='white' if val > 60 else 'black')

    fig.suptitle('Pipeline Comparison Matrix: Detector × OCR', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[visualizer] Saved → {output_path}')


def generate_all_figures(benchmark_report: dict, figures_dir: str) -> list[str]:
    """
    Generate all benchmark figures and return list of saved file paths.

    Args:
        benchmark_report: Output from benchmark.run_benchmark()
        figures_dir: Directory to save all figure files
    """
    saved = []

    p = os.path.join(figures_dir, 'accuracy_comparison.png')
    plot_accuracy_comparison(benchmark_report, p)
    saved.append(p)

    p = os.path.join(figures_dir, 'cer_comparison.png')
    plot_cer_comparison(benchmark_report, p)
    saved.append(p)

    p = os.path.join(figures_dir, 'detection_rate.png')
    plot_detection_rate(benchmark_report, p)
    saved.append(p)

    p = os.path.join(figures_dir, 'pipeline_matrix.png')
    plot_pipeline_matrix(benchmark_report, p)
    saved.append(p)

    return saved
