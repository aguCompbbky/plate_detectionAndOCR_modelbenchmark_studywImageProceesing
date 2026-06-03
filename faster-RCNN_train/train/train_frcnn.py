#!/usr/bin/env python3
"""
Faster R-CNN Training Script for License Plate Detection
=========================================================
- Model: fasterrcnn_resnet50_fpn (pretrained)
- 3 seeds (0, 42, 123), 30 epochs each, batch_size=8
- 80/20 train/val random split
- Collects: mAP@50, mAP@50:95, confusion matrix, LR, loss per epoch
- Outputs: metrics_report.json + trained model .pt files
- Saves confusion matrix plots and training curves to evidences/
"""

import os
import sys
import json
import random
import time
import glob
import warnings
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader, Subset
from PIL import Image
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
import torchvision.transforms.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS     = 30
IMG_SIZE   = 640
BATCH_SIZE = 8
SEEDS      = [0, 42, 123]  # User requested 123
LR         = 0.005
MOMENTUM   = 0.9
WEIGHT_DECAY = 0.0005
STEP_SIZE  = 10
GAMMA      = 0.1
NUM_CLASSES = 2  # background + plate
VAL_SPLIT  = 0.20
EVAL_INTERVAL = 15  # evaluate every N epochs (and always on last epoch)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA_DIR   = os.path.join(BASE_DIR, "dataset", "images")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PURE_MODEL_DIR = os.path.join(BASE_DIR, "pure_model")
EVIDENCES_DIR  = os.path.join(BASE_DIR, "evidences")
REPORT_PATH    = os.path.join(BASE_DIR, "metrics_report.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PURE_MODEL_DIR, exist_ok=True)
os.makedirs(EVIDENCES_DIR, exist_ok=True)


# ─── Dataset ────────────────────────────────────────────────────────────────────
class PlateDataset(torch.utils.data.Dataset):
    """License plate dataset – reads co-located image + YOLO .txt label pairs."""

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, data_dir, img_size=640):
        self.img_size = img_size
        self.samples = []  # list of (img_path, label_path)

        # Collect all image files that have a corresponding .txt
        for fname in sorted(os.listdir(data_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in self.IMG_EXTS:
                continue
            base = os.path.splitext(fname)[0]
            label_path = os.path.join(data_dir, base + ".txt")
            if os.path.isfile(label_path):
                img_path = os.path.join(data_dir, fname)
                self.samples.append((img_path, label_path))

        print(f"  Dataset loaded: {len(self.samples)} image-label pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        # Load & resize image
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img = img.resize((self.img_size, self.img_size))
        img_tensor = F.to_tensor(img)  # (3, H, W) float32 [0,1]
 
        # Parse YOLO labels → absolute boxes
        boxes, labels = [], []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                # Convert normalized YOLO → absolute pixel coords (on resized image)
                abs_xc = xc * self.img_size
                abs_yc = yc * self.img_size
                abs_w  = w  * self.img_size
                abs_h  = h  * self.img_size
                x1 = abs_xc - abs_w / 2
                y1 = abs_yc - abs_h / 2
                x2 = abs_xc + abs_w / 2
                y2 = abs_yc + abs_h / 2
                # Clamp
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(self.img_size, x2)
                y2 = min(self.img_size, y2)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls_id + 1)  # shift: 0→1 (background=0 in FRCNN)

        if len(boxes) == 0:
            # Empty annotation fallback
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ─── Model ───────────────────────────────────────────────────────────────────────
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ─── Evaluation helpers (VECTORIZED) ────────────────────────────────────────────
def compute_ap(precisions, recalls):
    """Compute AP using 101-point interpolation (COCO style)."""
    recall_levels = np.linspace(0, 1, 101)
    ap = 0
    for r_level in recall_levels:
        precs_at_r = [p for p, r in zip(precisions, recalls) if r >= r_level]
        if precs_at_r:
            ap += max(precs_at_r)
    return ap / 101


def evaluate_model(model, data_loader, device, iou_thresholds=None, score_threshold=0.3):
    """
    Evaluate model on data_loader using VECTORIZED IoU computation.
    Returns mAP@50, mAP@50:95, and confusion matrix (TP, FP, FN) at IoU=0.5.
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)  # 0.50 .. 0.95

    model.eval()

    # Collect all detections for AP calculation
    # For each detection: (iou_threshold, score, is_tp)
    all_detections = {round(t, 2): [] for t in iou_thresholds}
    total_gt = 0

    # For confusion matrix at IoU=0.5
    cm_tp, cm_fp, cm_fn = 0, 0, 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                gt_boxes = target["boxes"]
                num_gt = len(gt_boxes)
                total_gt += num_gt

                pred_boxes = output["boxes"].cpu()
                pred_scores = output["scores"].cpu()

                # Filter by score threshold
                keep = pred_scores >= score_threshold
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]

                # Sort by score descending
                order = torch.argsort(pred_scores, descending=True)
                pred_boxes = pred_boxes[order]
                pred_scores = pred_scores[order]

                num_pred = len(pred_boxes)

                if num_gt == 0:
                    cm_fp += num_pred
                    for iou_thr in iou_thresholds:
                        key = round(iou_thr, 2)
                        for ps in pred_scores:
                            all_detections[key].append((ps.item(), False))
                    continue

                if num_pred == 0:
                    cm_fn += num_gt
                    continue

                # VECTORIZED IoU computation
                iou_matrix = box_iou(pred_boxes, gt_boxes).numpy()  # (num_pred, num_gt)

                # --- Confusion matrix at IoU=0.5 ---
                matched_gt_50 = set()
                img_tp = 0
                for pi in range(num_pred):
                    if len(matched_gt_50) == num_gt:
                        break
                    best_gi = -1
                    best_iou = 0
                    for gi in range(num_gt):
                        if gi in matched_gt_50:
                            continue
                        if iou_matrix[pi, gi] > best_iou:
                            best_iou = iou_matrix[pi, gi]
                            best_gi = gi
                    if best_iou >= 0.5 and best_gi >= 0:
                        matched_gt_50.add(best_gi)
                        img_tp += 1

                cm_tp += img_tp
                cm_fp += num_pred - img_tp
                cm_fn += num_gt - img_tp

                # --- Per-IoU-threshold matching for AP ---
                for iou_thr in iou_thresholds:
                    key = round(iou_thr, 2)
                    matched = set()
                    for pi in range(num_pred):
                        best_gi = -1
                        best_iou = 0
                        for gi in range(num_gt):
                            if gi in matched:
                                continue
                            if iou_matrix[pi, gi] > best_iou:
                                best_iou = iou_matrix[pi, gi]
                                best_gi = gi
                        is_tp = best_iou >= iou_thr and best_gi >= 0 and best_gi not in matched
                        if is_tp:
                            matched.add(best_gi)
                        all_detections[key].append((pred_scores[pi].item(), is_tp))

    # Compute AP per IoU threshold
    aps = {}
    for iou_thr in iou_thresholds:
        key = round(iou_thr, 2)
        dets = all_detections[key]
        dets.sort(key=lambda x: -x[0])  # sort by score descending
        tp_cum, fp_cum = 0, 0
        precisions, recalls = [], []
        for score, is_tp in dets:
            if is_tp:
                tp_cum += 1
            else:
                fp_cum += 1
            precision = tp_cum / (tp_cum + fp_cum) if (tp_cum + fp_cum) > 0 else 0
            recall = tp_cum / total_gt if total_gt > 0 else 0
            precisions.append(precision)
            recalls.append(recall)
        ap = compute_ap(precisions, recalls) if dets else 0.0
        aps[key] = ap

    mAP50 = aps.get(0.5, 0.0)
    mAP50_95 = float(np.mean(list(aps.values()))) if aps else 0.0

    confusion_matrix = {
        "TP": int(cm_tp),
        "FP": int(cm_fp),
        "FN": int(cm_fn),
        "precision": round(cm_tp / (cm_tp + cm_fp), 4) if (cm_tp + cm_fp) > 0 else 0.0,
        "recall": round(cm_tp / (cm_tp + cm_fn), 4) if (cm_tp + cm_fn) > 0 else 0.0,
    }

    return mAP50, mAP50_95, confusion_matrix, aps


# ─── Visualization helpers ──────────────────────────────────────────────────────
def save_confusion_matrix(cm, save_path, seed, epoch_label="Final", normalize=False):
    """Save a confusion matrix visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # 2x2 matrix: rows=Predicted, cols=Actual
    # [TP, FP]
    # [FN, TN(N/A)]
    matrix = np.array([
        [cm["TP"], cm["FP"]],
        [cm["FN"], 0]
    ], dtype=float)

    if normalize:
        total = cm["TP"] + cm["FP"] + cm["FN"]
        if total > 0:
            matrix = matrix / total

    labels_text = [["TP", "FP"], ["FN", "TN\n(N/A)"]]

    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Positive (GT)", "Negative (GT)"])
    ax.set_yticklabels(["Positive (Pred)", "Negative (Pred)"])

    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            if i == 1 and j == 1:
                text = f"{labels_text[i][j]}\n0"
            else:
                text = f"{labels_text[i][j]}\n{val:.3f}" if normalize else f"{labels_text[i][j]}\n{int(val)}"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if val > (matrix.max() / 2) else "black")

    title_prefix = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"
    ax.set_title(f"{title_prefix} — Seed {seed} ({epoch_label})\n"
                 f"Precision: {cm['precision']:.4f} | Recall: {cm['recall']:.4f}",
                 fontsize=11)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(epoch_metrics, save_dir, seed):
    """Save loss, mAP, and LR curves."""
    epochs = [m["epoch"] for m in epoch_metrics]
    losses = [m["train_loss"] for m in epoch_metrics]
    lrs = [m["lr"] for m in epoch_metrics]

    # Filter only epochs with evaluation data
    eval_epochs = [m["epoch"] for m in epoch_metrics if m.get("mAP50") is not None]
    map50s = [m["mAP50"] for m in epoch_metrics if m.get("mAP50") is not None]
    map50_95s = [m["mAP50_95"] for m in epoch_metrics if m.get("mAP50_95") is not None]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(epochs, losses, "r-o", markersize=3, linewidth=1.5)
    axes[0, 0].set_title("Training Loss", fontsize=12)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    # mAP@50
    axes[0, 1].plot(eval_epochs, map50s, "b-o", markersize=5, linewidth=1.5)
    axes[0, 1].set_title("mAP@50", fontsize=12)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("mAP@50")
    axes[0, 1].grid(True, alpha=0.3)

    # mAP@50:95
    axes[1, 0].plot(eval_epochs, map50_95s, "g-o", markersize=5, linewidth=1.5)
    axes[1, 0].set_title("mAP@50:95", fontsize=12)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("mAP@50:95")
    axes[1, 0].grid(True, alpha=0.3)

    # LR
    axes[1, 1].plot(epochs, lrs, "m-o", markersize=3, linewidth=1.5)
    axes[1, 1].set_title("Learning Rate", fontsize=12)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("LR")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Training Curves — Seed {seed}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"training_curves_seed{seed}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_per_iou_ap_chart(per_iou_aps, save_dir, seed):
    """Save per-IoU AP bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    thresholds = sorted(per_iou_aps.keys())
    values = [per_iou_aps[t] for t in thresholds]
    bars = ax.bar([f"{t:.2f}" for t in thresholds], values, color="steelblue", edgecolor="black")
    ax.set_title(f"AP per IoU Threshold — Seed {seed}", fontsize=13)
    ax.set_xlabel("IoU Threshold")
    ax.set_ylabel("AP")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"per_iou_ap_seed{seed}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Training ───────────────────────────────────────────────────────────────────
def train_one_seed(seed, dataset, device):
    """Train a single Faster R-CNN model with the given seed."""
    print(f"\n{'='*70}")
    print(f"  SEED = {seed}")
    print(f"{'='*70}")

    # Create seed evidence directory
    seed_evidence_dir = os.path.join(EVIDENCES_DIR, f"seed_{seed}")
    os.makedirs(seed_evidence_dir, exist_ok=True)

    # Seed everything
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 80/20 split
    n = len(dataset)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(n * (1 - VAL_SPLIT))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_ds = Subset(dataset, train_indices)
    val_ds   = Subset(dataset, val_indices)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # Model
    model = get_model(NUM_CLASSES).to(device)

    # Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    epoch_metrics = []
    best_map50 = 0.0
    best_map50_95 = 0.0

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # ── Train ──
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False)
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            num_batches += 1
            
            pbar.set_postfix({"loss": f"{losses.item():.4f}"})

        avg_loss = epoch_loss / max(num_batches, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # ── Evaluate (only every EVAL_INTERVAL epochs and on last epoch) ──
        do_eval = (epoch % EVAL_INTERVAL == 0) or (epoch == EPOCHS)

        if do_eval:
            mAP50, mAP50_95, cm, per_iou_aps = evaluate_model(
                model, val_loader, device
            )
            best_map50 = max(best_map50, mAP50)
            best_map50_95 = max(best_map50_95, mAP50_95)
        else:
            mAP50, mAP50_95, cm = 0.0, 0.0, None

        epoch_elapsed = time.time() - epoch_start

        epoch_data = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 6),
            "lr": current_lr,
            "mAP50": round(mAP50, 6) if do_eval else None,
            "mAP50_95": round(mAP50_95, 6) if do_eval else None,
            "epoch_time_seconds": round(epoch_elapsed, 1),
        }
        if do_eval:
            epoch_data["confusion_matrix"] = cm
        epoch_metrics.append(epoch_data)

        if do_eval:
            print(f"  Epoch {epoch:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
                  f"LR: {current_lr:.6f} | mAP@50: {mAP50:.4f} | "
                  f"mAP@50:95: {mAP50_95:.4f} | "
                  f"P: {cm['precision']:.3f} R: {cm['recall']:.3f} | "
                  f"Time: {epoch_elapsed:.0f}s")
        else:
            print(f"  Epoch {epoch:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
                  f"LR: {current_lr:.6f} | Time: {epoch_elapsed:.0f}s")

    # Final evaluation
    final_mAP50, final_mAP50_95, final_cm, final_per_iou = evaluate_model(
        model, val_loader, device
    )

    # ── Save evidences ──
    # Confusion matrix
    save_confusion_matrix(final_cm, os.path.join(seed_evidence_dir, f"confusion_matrix_seed{seed}.png"),
                          seed, epoch_label="Final")
    save_confusion_matrix(final_cm, os.path.join(seed_evidence_dir, f"confusion_matrix_normalized_seed{seed}.png"),
                          seed, epoch_label="Final", normalize=True)
    # Training curves
    save_training_curves(epoch_metrics, seed_evidence_dir, seed)
    # Per-IoU AP chart
    save_per_iou_ap_chart(final_per_iou, seed_evidence_dir, seed)

    print(f"  Evidences saved → {seed_evidence_dir}/")

    # Save model to models/ directory
    save_path = os.path.join(MODELS_DIR, f"frcnn_seed{seed}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"  Model saved → {save_path}")

    return {
        "epoch_metrics": epoch_metrics,
        "final_confusion_matrix": final_cm,
        "final_per_iou_ap": {str(k): round(v, 6) for k, v in final_per_iou.items()},
        "best_mAP50": round(best_map50, 6),
        "best_mAP50_95": round(best_map50_95, 6),
        "final_mAP50": round(final_mAP50, 6),
        "final_mAP50_95": round(final_mAP50_95, 6),
        "model_path": save_path,
    }


# ─── Main ────────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Save pure (pretrained, not fine-tuned) model to pure_model/
    pure_save_path = os.path.join(PURE_MODEL_DIR, "frcnn_pure.pt")
    if not os.path.exists(pure_save_path):
        print("Saving pure (pretrained, not fine-tuned) model...")
        pure_model = get_model(NUM_CLASSES)
        torch.save(pure_model.state_dict(), pure_save_path)
        print(f"  Pure model saved → {pure_save_path}")
        del pure_model
    else:
        print(f"  Pure model already exists → {pure_save_path}")

    # Load dataset once
    dataset = PlateDataset(DATA_DIR, img_size=IMG_SIZE)

    # Merge with existing report if present
    if os.path.exists(REPORT_PATH):
        try:
            with open(REPORT_PATH, "r", encoding="utf-8") as f:
                report = json.load(f)
            print(f"  Loaded existing report from {REPORT_PATH}")
            # Update hyperparameters just in case
            report["hyperparameters"]["epochs"] = EPOCHS
            report["hyperparameters"]["img_size"] = IMG_SIZE
        except Exception as e:
            print(f"  Could not load existing report: {e}")
            report = {
                "hyperparameters": {
                    "epochs": EPOCHS,
                    "img_size": IMG_SIZE,
                    "batch_size": BATCH_SIZE,
                    "seeds": SEEDS,
                    "optimizer": "SGD",
                    "lr": LR,
                    "momentum": MOMENTUM,
                    "weight_decay": WEIGHT_DECAY,
                    "scheduler": f"StepLR(step_size={STEP_SIZE}, gamma={GAMMA})",
                    "model": "fasterrcnn_resnet50_fpn",
                    "num_classes": NUM_CLASSES,
                    "train_val_split": f"{int((1-VAL_SPLIT)*100)}/{int(VAL_SPLIT*100)}",
                    "total_images": len(dataset),
                },
                "seeds": {},
            }
    else:
        report = {
            "hyperparameters": {
                "epochs": EPOCHS,
                "img_size": IMG_SIZE,
                "batch_size": BATCH_SIZE,
                "seeds": SEEDS,
                "optimizer": "SGD",
                "lr": LR,
                "momentum": MOMENTUM,
                "weight_decay": WEIGHT_DECAY,
                "scheduler": f"StepLR(step_size={STEP_SIZE}, gamma={GAMMA})",
                "model": "fasterrcnn_resnet50_fpn",
                "num_classes": NUM_CLASSES,
                "train_val_split": f"{int((1-VAL_SPLIT)*100)}/{int(VAL_SPLIT*100)}",
                "total_images": len(dataset),
            },
            "seeds": {},
        }

    total_start = time.time()

    for seed in SEEDS:
        seed_start = time.time()
        seed_results = train_one_seed(seed, dataset, device)
        seed_elapsed = time.time() - seed_start
        seed_results["training_time_seconds"] = round(seed_elapsed, 1)
        report["seeds"][f"seed_{seed}"] = seed_results

        # Save intermediate report
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"  Intermediate report saved → {REPORT_PATH}")

    total_elapsed = time.time() - total_start

    # Summary across seeds
    # Summary across all computed seeds
    computed_seeds = list(report["seeds"].keys())
    all_best50 = [report["seeds"][s].get("best_mAP50", 0) for s in computed_seeds]
    all_best50_95 = [report["seeds"][s].get("best_mAP50_95", 0) for s in computed_seeds]
    all_final50 = [report["seeds"][s].get("final_mAP50", 0) for s in computed_seeds]
    all_final50_95 = [report["seeds"][s].get("final_mAP50_95", 0) for s in computed_seeds]

    report["summary"] = {
        "total_training_time_seconds": round(total_elapsed, 1),
        "mean_best_mAP50": round(float(np.mean(all_best50)), 6),
        "std_best_mAP50": round(float(np.std(all_best50)), 6),
        "mean_best_mAP50_95": round(float(np.mean(all_best50_95)), 6),
        "std_best_mAP50_95": round(float(np.std(all_best50_95)), 6),
        "mean_final_mAP50": round(float(np.mean(all_final50)), 6),
        "std_final_mAP50": round(float(np.std(all_final50)), 6),
        "mean_final_mAP50_95": round(float(np.mean(all_final50_95)), 6),
        "std_final_mAP50_95": round(float(np.std(all_final50_95)), 6),
    }

    # ── Summary comparison chart ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(computed_seeds, all_final50, color=["#4e79a7", "#f28e2b", "#e15759"][:len(computed_seeds)],
                edgecolor="black")
    axes[0].axhline(y=float(np.mean(all_final50)), color="black", linestyle="--",
                    label=f"Mean: {np.mean(all_final50):.4f}")
    axes[0].set_title("Final mAP@50 per Seed", fontsize=13)
    axes[0].set_ylabel("mAP@50")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(computed_seeds, all_final50_95, color=["#4e79a7", "#f28e2b", "#e15759"][:len(computed_seeds)],
                edgecolor="black")
    axes[1].axhline(y=float(np.mean(all_final50_95)), color="black", linestyle="--",
                    label=f"Mean: {np.mean(all_final50_95):.4f}")
    axes[1].set_title("Final mAP@50:95 per Seed", fontsize=13)
    axes[1].set_ylabel("mAP@50:95")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Seed Comparison — Faster R-CNN", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(EVIDENCES_DIR, "seed_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save final report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"  ALL DONE — Total time: {total_elapsed/60:.1f} min")
    print(f"  Report    → {REPORT_PATH}")
    print(f"  Models    → {MODELS_DIR}/")
    print(f"  Evidences → {EVIDENCES_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
