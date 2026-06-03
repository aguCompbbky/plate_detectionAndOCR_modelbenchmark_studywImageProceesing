"""
run_pipeline.py — Main orchestrator for the license plate experiment pipeline.

Usage:
    # Run all pipelines
    python run_pipeline.py --all

    # Single pipeline
    python run_pipeline.py --detector yolo --ocr crnn

    # List available pipelines
    python run_pipeline.py --list
"""
import os
import sys
import json
import argparse
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DB_IMAGES_DIR   = BASE_DIR / 'database' / 'test_dataset' / 'images'
DB_PLATES_JSON  = BASE_DIR / 'database' / 'test_dataset' / 'plates' / 'plates.json'
CROPPED_DIR     = BASE_DIR / 'database' / 'croped_img'
RESULTS_DIR     = BASE_DIR / 'presentation' / 'results'
FIGURES_DIR     = BASE_DIR / 'presentation' / 'figures'
BENCHMARK_PATH  = BASE_DIR / 'presentation' / 'benchmark_report.json'

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CROPPED_DIR.mkdir(parents=True, exist_ok=True)

# ─── Model default paths ──────────────────────────────────────────────────────
YOLO_MODEL  = str(BASE_DIR.parent / 'yolo_train' / 'evidences' / 'seed_0' / 'train' / 'weights' / 'best.pt')
RCNN_MODEL  = str(BASE_DIR.parent / 'faster-RCNN_train' / 'models' / 'frcnn_seed0.pt')
CRNN_MODEL  = str(BASE_DIR.parent / 'CRNN_train' / 'models' / 'seed42_fold0_best.pth')
LPRNET_MODEL = str(BASE_DIR.parent / 'LPRNet_train' / 'models' / 'lprnet_seed42_fold0.pth')


# ─── Detector / OCR registry ──────────────────────────────────────────────────

def _load_detector(name: str):
    """Load detector by name. Returns None if model file not found."""
    if name == 'yolo':
        try:
            from data.plate_detection.YOLO.yolo_detector import YOLODetector
            return YOLODetector(model_path=YOLO_MODEL)
        except FileNotFoundError as e:
            print(f'[SKIP] {e}')
            return None

    elif name == 'rcnn':
        try:
            from data.plate_detection.R_CNN.rcnn_detector import RCNNDetector  # noqa
            return RCNNDetector(model_path=RCNN_MODEL)
        except FileNotFoundError as e:
            print(f'[SKIP] {e}')
            return None

    raise ValueError(f'Unknown detector: {name}')


def _load_ocr(name: str):
    """Load OCR model by name. Returns None if model file not found."""
    if name == 'crnn':
        try:
            from data.ocr.CRNN.crnn_ocr import CRNNOcr
            return CRNNOcr(model_path=CRNN_MODEL)
        except FileNotFoundError as e:
            print(f'[SKIP] {e}')
            return None

    elif name == 'lprnet':
        try:
            from data.ocr.LPRNet.lprnet_ocr import LPRNetOcr
            return LPRNetOcr(model_path=LPRNET_MODEL)
        except FileNotFoundError as e:
            print(f'[SKIP] {e}')
            return None

    elif name == 'paddle':
        try:
            from data.ocr.padde_ocr.paddle_ocr import PaddleOcrReader
            return PaddleOcrReader()
        except ImportError as e:
            print(f'[SKIP] {e}')
            return None

    raise ValueError(f'Unknown OCR model: {name}')


def _get_pipeline_preprocessing(ocr_name: str):
    """Return the correct preprocessing pipeline for the given OCR model."""
    from domain.pre_image_proccesing.pipeline_manager import (
        get_standard_pipeline, get_paddle_pipeline
    )
    if ocr_name == 'paddle':
        return get_paddle_pipeline()
    return get_standard_pipeline()


# ─── Single pipeline run ──────────────────────────────────────────────────────

def run_single_pipeline(detector_name: str, ocr_name: str) -> dict | None:
    """
    Run one (detector, OCR) pipeline over all test images.

    Returns:
        Results dict or None if detector/OCR could not be loaded
    """
    pipeline_name = f'{detector_name}_{ocr_name}'
    print(f'\n{"="*60}')
    print(f'  PIPELINE: {pipeline_name.upper()}')
    print(f'{"="*60}')

    # Load components
    detector = _load_detector(detector_name)
    if detector is None:
        return None

    ocr = _load_ocr(ocr_name)
    if ocr is None:
        return None

    preproc = _get_pipeline_preprocessing(ocr_name)

    from domain.regex_filter.turkish_plate_regex import TurkishPlateFilter
    regex_filter = TurkishPlateFilter()

    # Crop directory for this pipeline
    pipeline_crop_dir = CROPPED_DIR / pipeline_name
    pipeline_crop_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    image_files = sorted(DB_IMAGES_DIR.glob('*.jpeg')) + sorted(DB_IMAGES_DIR.glob('*.jpg')) + \
                  sorted(DB_IMAGES_DIR.glob('*.png'))

    for img_path in image_files:
        image_name = img_path.name
        stem = img_path.stem
        print(f'  [{stem}] ', end='', flush=True)

        has_detection = False
        ocr_raw = ''
        ocr_filtered = ''

        try:
            crops = detector.detect(str(img_path))

            if crops:
                has_detection = True
                # Use highest-confidence crop (first one from detector)
                crop_bgr, bbox = crops[0]

                # Save crop to database/croped_img
                crop_path = pipeline_crop_dir / f'{stem}_crop.jpg'
                import cv2
                cv2.imwrite(str(crop_path), crop_bgr)

                # Preprocessing
                processed = preproc.run(crop_bgr)

                # OCR
                ocr_raw = ocr.read(processed)

                # Regex filter
                filtered = regex_filter.apply(ocr_raw)
                ocr_filtered = filtered if filtered else ''

                print(f'raw="{ocr_raw}" → filtered="{ocr_filtered}"')
            else:
                print('no detection')

        except Exception as exc:
            print(f'ERROR: {exc}')

        results[image_name] = {
            'detected': has_detection,
            'ocr_raw': ocr_raw,
            'ocr_filtered': ocr_filtered,
        }

    # Save results JSON
    out_path = RESULTS_DIR / f'{pipeline_name}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n  Results saved → {out_path}')

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

ALL_DETECTORS = ['yolo', 'rcnn']
ALL_OCR = ['crnn', 'lprnet', 'paddle']


def main():
    parser = argparse.ArgumentParser(
        description='License Plate Experiment Pipeline'
    )
    parser.add_argument('--detector', choices=ALL_DETECTORS,
                        help='Plate detector to use')
    parser.add_argument('--ocr', choices=ALL_OCR,
                        help='OCR model to use')
    parser.add_argument('--all', action='store_true',
                        help='Run all detector × OCR combinations')
    parser.add_argument('--list', action='store_true',
                        help='List available pipeline combinations')
    parser.add_argument('--benchmark-only', action='store_true',
                        help='Skip inference, just recompute benchmark from existing results')
    args = parser.parse_args()

    if args.list:
        print('Available pipelines:')
        for d in ALL_DETECTORS:
            for o in ALL_OCR:
                print(f'  --detector {d} --ocr {o}')
        return

    # Change working directory to experiment root so relative imports work
    os.chdir(str(BASE_DIR))
    sys.path.insert(0, str(BASE_DIR))

    if not args.benchmark_only:
        if args.all:
            for det in ALL_DETECTORS:
                for ocr in ALL_OCR:
                    run_single_pipeline(det, ocr)
        elif args.detector and args.ocr:
            run_single_pipeline(args.detector, args.ocr)
        else:
            parser.print_help()
            return

    # ── Benchmark & Visualize ─────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print('  BENCHMARKING')
    print(f'{"="*60}')

    from presentation.benchmark import run_benchmark, save_benchmark
    from presentation.visualizer import generate_all_figures

    benchmark = run_benchmark(str(RESULTS_DIR), str(DB_PLATES_JSON))

    if not benchmark:
        print('[WARNING] No results to benchmark yet.')
        return

    save_benchmark(benchmark, str(BENCHMARK_PATH))

    # Print summary table
    print(f'\n{"Pipeline":<20} {"Accuracy":>9} {"CER":>8} {"DetRate":>9}')
    print('-' * 50)
    for name, m in sorted(benchmark.items()):
        print(f'{name:<20} {m["exact_match_accuracy"]*100:>8.1f}% '
              f'{m["mean_cer"]*100:>7.1f}% '
              f'{m["detection_rate"]*100:>8.1f}%')

    # Best pipeline
    best = max(benchmark.items(), key=lambda kv: kv[1]['exact_match_accuracy'])
    print(f'\n✓ Best pipeline: {best[0].upper()} '
          f'({best[1]["exact_match_accuracy"]*100:.1f}% accuracy, '
          f'{best[1]["mean_cer"]*100:.1f}% CER)')

    # Generate figures
    saved_figs = generate_all_figures(benchmark, str(FIGURES_DIR))
    print(f'\n  {len(saved_figs)} figures saved → {FIGURES_DIR}/')


if __name__ == '__main__':
    main()
