"""
benchmark.py — Compares pipeline results against ground-truth plates.

Metrics computed per pipeline:
  - Exact Match Accuracy: % of plates exactly matching ground truth
  - CER (Character Error Rate): edit distance / ground truth length
  - Detected Count: number of images where a crop was found
"""
import json
import os
import re
from pathlib import Path


def _cer(pred: str, gt: str) -> float:
    """Compute normalized CER (Levenshtein distance / len(gt))."""
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0

    # Remove spaces for CER calculation (compare character sequences)
    p = pred.replace(' ', '')
    g = gt.replace(' ', '')

    m, n = len(g), len(p)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if g[i - 1] == p[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n] / m


def run_benchmark(results_dir: str, ground_truth_path: str) -> dict:
    """
    Load all pipeline result JSON files and compare against ground truth.

    Args:
        results_dir: Directory containing per-pipeline JSON result files
        ground_truth_path: Path to plates.json (ground truth)

    Returns:
        Dict mapping pipeline_name → metrics dict
    """
    # Load ground truth
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        gt_list = json.load(f)
    gt_map = {item['image']: item['plate'] for item in gt_list}

    benchmark = {}

    # Find all result JSON files
    result_files = sorted(Path(results_dir).glob('*.json'))
    if not result_files:
        print(f"[benchmark] No result files found in {results_dir}")
        return benchmark

    for result_file in result_files:
        pipeline_name = result_file.stem  # e.g. "yolo_crnn"
        with open(result_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        total = len(gt_map)
        exact_matches = 0
        cer_total = 0.0
        detected_count = 0
        per_image = {}

        for image_name, gt_plate in gt_map.items():
            pred_info = results.get(image_name, {})
            pred_raw = pred_info.get('ocr_raw', '')
            pred_filtered = pred_info.get('ocr_filtered', '')
            detected = pred_info.get('detected', False)

            if detected:
                detected_count += 1

            # Use filtered output; fall back to raw if filter returned nothing
            pred = pred_filtered if pred_filtered else pred_raw

            is_exact = (pred.upper().replace(' ', '') == gt_plate.upper().replace(' ', ''))
            cer_val = _cer(pred.upper(), gt_plate.upper())

            if is_exact:
                exact_matches += 1

            cer_total += cer_val
            per_image[image_name] = {
                'gt': gt_plate,
                'pred_raw': pred_raw,
                'pred_filtered': pred_filtered,
                'exact_match': is_exact,
                'cer': round(cer_val, 4),
                'detected': detected,
            }

        benchmark[pipeline_name] = {
            'total_images': total,
            'detected_count': detected_count,
            'detection_rate': round(detected_count / total, 4) if total > 0 else 0.0,
            'exact_match_count': exact_matches,
            'exact_match_accuracy': round(exact_matches / total, 4) if total > 0 else 0.0,
            'mean_cer': round(cer_total / total, 4) if total > 0 else 0.0,
            'per_image': per_image,
        }

    return benchmark


def save_benchmark(metrics: dict, output_path: str) -> None:
    """Save benchmark metrics dict as JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[benchmark] Saved → {output_path}")
