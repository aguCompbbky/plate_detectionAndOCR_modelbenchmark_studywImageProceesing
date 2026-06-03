"""
train — CRNN licence-plate OCR training package.

Public API:
    build_model       — construct the CRNN model
    PlateDataset      — dataset class
    build_dataset     — load valid (image, label) pairs
    collate_fn        — DataLoader collation for CTC
    beam_search_decode— beam-search CTC decoder
    greedy_decode     — greedy CTC decoder
    compute_cer       — Character Error Rate
    compute_wer       — Word Error Rate
    run_kfold_training— full k-fold training orchestrator
"""

from .model   import build_model, CRNN
from .dataset import PlateDataset, build_dataset, collate_fn
from .decoder import beam_search_decode, greedy_decode
from .metrics import (
    compute_cer, compute_wer, compute_exact_match,
    compute_char_map, plot_confusion_matrix,
    plot_training_curves, save_metrics_report,
)
from .trainer import run_kfold_training, set_seed

__all__ = [
    # model
    'build_model', 'CRNN',
    # data
    'PlateDataset', 'build_dataset', 'collate_fn',
    # decoding
    'beam_search_decode', 'greedy_decode',
    # metrics
    'compute_cer', 'compute_wer', 'compute_exact_match',
    'compute_char_map', 'plot_confusion_matrix',
    'plot_training_curves', 'save_metrics_report',
    # training
    'run_kfold_training', 'set_seed',
]
