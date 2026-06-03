import json
import os

BASE_DIR = "/home/muk/Masaüstü/python_scripts/plaka2/faster-RCNN_train"
REPORT_PATH = os.path.join(BASE_DIR, "metrics_report.json")

# Data retrieved from logs 
seed_42_data = {
  "epoch_metrics": [
    {"epoch": 1, "train_loss": 0.119919, "lr": 0.005, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 232.4},
    {"epoch": 2, "train_loss": 0.064936, "lr": 0.005, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 236.2},
    {"epoch": 3, "train_loss": 0.054668, "lr": 0.005, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 237.5},
    {"epoch": 4, "train_loss": 0.048782, "lr": 0.005, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 237.0},
    {"epoch": 5, "train_loss": 0.044205, "lr": 0.005, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 237.5},
    {"epoch": 6, "train_loss": 0.041521, "lr": 0.005, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 238.6},
    {"epoch": 7, "train_loss": 0.041194, "lr": 0.005, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 239.1},
    {"epoch": 8, "train_loss": 0.037299, "lr": 0.005, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 235.6},
    {"epoch": 9, "train_loss": 0.034111, "lr": 0.005, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 238.1},
    {"epoch": 10, "train_loss": 0.032178, "lr": 0.005, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 238.6},
    {"epoch": 11, "train_loss": 0.027244, "lr": 0.0005, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 238.0},
    {"epoch": 12, "train_loss": 0.026441, "lr": 0.0005, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 234.9},
    {"epoch": 13, "train_loss": 0.025721, "lr": 0.0005, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 235.9},
    {"epoch": 14, "train_loss": 0.025093, "lr": 0.0005, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 239.1},
    {"epoch": 15, "train_loss": 0.024332, "lr": 0.0005, "mAP50": 0.810929, "mAP50_95": 0.709483, "epoch_time_seconds": 271.5, "confusion_matrix": {"TP": 335, "FP": 58, "FN": 75, "precision": 0.8524, "recall": 0.8171}},
    {"epoch": 16, "train_loss": 0.023704, "lr": 0.0005, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 236.4},
    {"epoch": 17, "train_loss": 0.0234, "lr": 0.0005, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 236.8},
    {"epoch": 18, "train_loss": 0.023202, "lr": 0.0005, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 237.7},
    {"epoch": 19, "train_loss": 0.022939, "lr": 0.0005, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 241.6},
    {"epoch": 20, "train_loss": 0.022386, "lr": 0.0005, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 237.7},
    {"epoch": 21, "train_loss": 0.022006, "lr": 5e-05, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 240.8},
    {"epoch": 22, "train_loss": 0.021688, "lr": 5e-05, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 237.7},
    {"epoch": 23, "train_loss": 0.021757, "lr": 5e-05, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 238.1},
    {"epoch": 24, "train_loss": 0.021619, "lr": 5e-05, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 242.8},
    {"epoch": 25, "train_loss": 0.021581, "lr": 5e-05, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 241.1},
    {"epoch": 26, "train_loss": 0.021601, "lr": 5e-05, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 239.1},
    {"epoch": 27, "train_loss": 0.021641, "lr": 5e-05, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 239.5},
    {"epoch": 28, "train_loss": 0.021594, "lr": 5e-05, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 241.2},
    {"epoch": 29, "train_loss": 0.021531, "lr": 5e-05, "mAP50": None, "mAP50_95": None, "epoch_time_seconds": 242.4},
    {"epoch": 30, "train_loss": 0.021561, "lr": 5e-05, "mAP50": 0.810901, "mAP50_95": 0.711093, "epoch_time_seconds": 271.4, "confusion_matrix": {"TP": 336, "FP": 43, "FN": 74, "precision": 0.8865, "recall": 0.8195}}
  ],
  "final_confusion_matrix": {"TP": 336, "FP": 43, "FN": 74, "precision": 0.8865, "recall": 0.8195},
  "final_per_iou_ap": {
    "0.5": 0.810901, "0.55": 0.810901, "0.6": 0.810901, "0.65": 0.810901, "0.7": 0.810901,
    "0.75": 0.810008, "0.8": 0.800862, "0.85": 0.757489, "0.9": 0.586395, "0.95": 0.101673
  },
  "best_mAP50": 0.810929,
  "best_mAP50_95": 0.711093,
  "final_mAP50": 0.810901,
  "final_mAP50_95": 0.711093,
  "model_path": "/home/muk/Masaüstü/python_scripts/plaka2/faster-RCNN_train/models/frcnn_seed42.pt",
  "training_time_seconds": 7244.3
}

with open(REPORT_PATH, "r") as f:
    report = json.load(f)

report["seeds"]["seed_42"] = seed_42_data

# Recalculate summary
import numpy as np
computed_seeds = list(report["seeds"].keys())
report["summary"]["mean_best_mAP50"] = round(float(np.mean([report["seeds"][s].get("best_mAP50",0) for s in computed_seeds])), 6)
report["summary"]["mean_best_mAP50_95"] = round(float(np.mean([report["seeds"][s].get("best_mAP50_95",0) for s in computed_seeds])), 6)
report["summary"]["mean_final_mAP50"] = round(float(np.mean([report["seeds"][s].get("final_mAP50",0) for s in computed_seeds])), 6)
report["summary"]["mean_final_mAP50_95"] = round(float(np.mean([report["seeds"][s].get("final_mAP50_95",0) for s in computed_seeds])), 6)

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)
print("Seed 42 restored.")
