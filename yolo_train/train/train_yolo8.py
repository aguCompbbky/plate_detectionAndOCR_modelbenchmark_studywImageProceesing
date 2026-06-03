"""
YOLOv8 Plaka Tespiti - Çoklu Seed Eğitim Scripti
====================================================
Hyperparametreler:
    - Epochs  : 30
    - IMG_SIZE : 640
    - BATCH    : 8
    - Seeds    : [0, 42, 123]

Dataset Bölme Stratejisi:
    - %80 Train / %20 Val — rastgele, TEK SEFERLIK (SPLIT_SEED=42)
    - Her seed eğitimi aynı train/val setini kullanır

Çıktılar:
    - Her seed için model: yolo{seed}.pt
    - Akademik metrik raporu: evidences/metrics_report.json
    - CSV özet: evidences/all_seeds_summary.csv
"""

import os
import sys
import json
import random
import shutil
import csv
import glob
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PROJE YOLLARI
# ─────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent          # yolo_train/
DATASET_DIR = BASE_DIR / "dataset" / "images"                 # görüntü + etiket
PURE_MODEL  = BASE_DIR / "pure_model" / "yolov8n.pt"
EVIDENCE_DIR = BASE_DIR / "evidences"
TRAIN_DIR   = BASE_DIR / "train"

# ─────────────────────────────────────────────
# HYPERPARAMETRELER
# ─────────────────────────────────────────────
EPOCHS     = 30
IMG_SIZE   = 640
BATCH_SIZE = 8
SEEDS      = [0, 42, 123]
VAL_SPLIT  = 0.20    # %20 validation
SPLIT_SEED = 42      # dataset bölme için sabit seed (seed döngüsünden bağımsız)
NC         = 1       # sınıf sayısı (plaka)
CLASS_NAMES = ["license_plate"]

# ─────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────

def set_seed(seed: int):
    """Tüm rassal sayı üreticilerini sabitler."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def collect_valid_samples(dataset_dir: Path):
    """
    dataset/images/ içindeki TÜM görüntüleri toplar.
    Kabul edilen formatlar: jpg, jpeg, png, webp
    - Etiket dosyası yoksa → boş .txt oluşturulur (background örneği)
    - Etiket dosyası boşsa → olduğu gibi bırakılır (background örneği)
    """
    exts = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    all_images = []
    for ext in exts:
        all_images.extend(sorted(dataset_dir.glob(ext)))

    valid = []
    for img in all_images:
        txt = img.with_suffix(".txt")
        if not txt.exists():
            txt.touch()   # boş etiket dosyası oluştur → background
        valid.append(img)

    n_empty = sum(1 for img in valid if img.with_suffix(".txt").stat().st_size == 0)
    print(f"[INFO] Toplam örnek: {len(valid)} "
          f"({n_empty} boş etiket = background, "
          f"{len(valid)-n_empty} annotasyonlu)")
    return valid


def build_global_split(samples: list, evidence_dir: Path):
    """
    Dataset'i BİR KERE %80 train / %20 val olarak böler.
    Bölme SPLIT_SEED ile sabitlenir; seed döngüsünden bağımsızdır.
    Tüm dosyalar kopyalanır:
        evidence_dir/dataset_split/
            images/train/
            images/val/
            labels/train/
            labels/val/
    data.yaml evidence_dir/dataset_split/ içine yazılır.
    Zaten bölünmüşse yeniden kopyalamaz.
    """
    split_dir = evidence_dir / "dataset_split"
    data_yaml = split_dir / "data.yaml"

    if data_yaml.exists():
        print(f"[INFO] Mevcut dataset split kullanılıyor → {split_dir}")
        # Sayıları dosyadan oku
        n_train = len(list((split_dir / "images" / "train").glob("*")))
        n_val   = len(list((split_dir / "images" / "val").glob("*")))
        return str(data_yaml), n_train, n_val

    # Rastgele karıştır (sabit seed)
    random.seed(SPLIT_SEED)
    shuffled = list(samples)
    random.shuffle(shuffled)

    n_val   = max(1, int(len(shuffled) * VAL_SPLIT))
    n_train = len(shuffled) - n_val
    train_imgs = shuffled[:n_train]
    val_imgs   = shuffled[n_train:]

    print(f"[INFO] Dataset bölündü (SPLIT_SEED={SPLIT_SEED}): "
          f"Train={n_train}, Val={n_val} (toplam={len(shuffled)})")

    for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
        img_dst = split_dir / "images" / split_name
        lbl_dst = split_dir / "labels" / split_name
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for img_path in split_imgs:
            shutil.copy2(img_path, img_dst / img_path.name)
            lbl_path = img_path.with_suffix(".txt")
            shutil.copy2(lbl_path, lbl_dst / lbl_path.name)

    # data.yaml
    yaml_content = (
        f"path: {split_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"nc: {NC}\n"
        f"names: {CLASS_NAMES}\n"
    )
    data_yaml.write_text(yaml_content)
    print(f"[OK] data.yaml → {data_yaml}")
    return str(data_yaml), n_train, n_val


def parse_results_csv(results_csv: Path) -> dict:
    """
    YOLO'nun results.csv dosyasından son epoch metriklerini okur.
    Dönen dict anahtarları normalize edilmiş (strip + lower) sütun isimleridir.
    """
    if not results_csv.exists():
        return {}

    with open(results_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {}

    # Son epoch satırı
    last = {k.strip(): v.strip() for k, v in rows[-1].items()}

    def safe_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    metrics = {
        # Detection metrics
        "mAP50":                safe_float(last.get("metrics/mAP50(B)")),
        "mAP50_95":             safe_float(last.get("metrics/mAP50-95(B)")),
        "precision":            safe_float(last.get("metrics/precision(B)")),
        "recall":               safe_float(last.get("metrics/recall(B)")),
        "f1":                   safe_float(last.get("metrics/f1(B)")),
        "accuracy":             safe_float(last.get("metrics/accuracy(B)")),
        
        # Box loss
        "train_box_loss":       safe_float(last.get("train/box_loss")),
        "train_cls_loss":       safe_float(last.get("train/cls_loss")),
        "train_dfl_loss":       safe_float(last.get("train/dfl_loss")),
        "val_box_loss":         safe_float(last.get("val/box_loss")),
        "val_cls_loss":         safe_float(last.get("val/cls_loss")),
        "val_dfl_loss":         safe_float(last.get("val/dfl_loss")),
        # Learning Rate
        "lr_pg0":               safe_float(last.get("lr/pg0")),
        "lr_pg1":               safe_float(last.get("lr/pg1")),
        "lr_pg2":               safe_float(last.get("lr/pg2")),
        # Epoch
        "last_epoch":           safe_float(last.get("                  epoch")) or safe_float(last.get("epoch")),
    }
    return metrics


def parse_all_epochs_csv(results_csv: Path) -> list:
    """Tüm epoch'ların metriklerini liste olarak döner (learning rate eğrisi için)."""
    if not results_csv.exists():
        return []
    with open(results_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            clean = {k.strip(): v.strip() for k, v in row.items()}
            rows.append(clean)
    return rows


def extract_confusion_matrix_info(run_save_dir: Path) -> dict:
    """
    YOLO confusion_matrix.csv (normalized) veya confusion_matrix dosyasını
    okuyarak TP/FP/FN/TN değerlerini çıkarır.
    """
    cm_files = list(run_save_dir.glob("confusion_matrix*.csv"))
    if not cm_files:
        # PNG olarak üretilmiş olabilir, yoksa boş döndür
        cm_png = list(run_save_dir.glob("confusion_matrix*"))
        return {"confusion_matrix_file": str(cm_png[0]) if cm_png else "not_found"}

    # İlk csv'yi oku
    cm_path = cm_files[0]
    try:
        with open(cm_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
        return {
            "confusion_matrix_csv": str(cm_path),
            "confusion_matrix_rows": rows,
        }
    except Exception:
        return {"confusion_matrix_file": str(cm_path)}


def compute_f1(precision, recall):
    if precision is None or recall is None:
        return None
    denom = precision + recall
    if denom == 0.0:
        return 0.0
    return 2 * precision * recall / denom


# ─────────────────────────────────────────────
# ANA EĞİTİM DÖNGÜSÜ
# ─────────────────────────────────────────────

def main():
    EVIDENCE_DIR.mkdir(exist_ok=True)
    print("=" * 60)
    print("YOLOv8 Çoklu Seed Eğitimi Başlıyor")
    print(f"Epochs={EPOCHS}, IMG={IMG_SIZE}, Batch={BATCH_SIZE}")
    print(f"Seeds: {SEEDS}")
    print("=" * 60)

    # Ultralytics import (kurulu değilse erken çık)
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[HATA] ultralytics kurulu değil. Lütfen: pip install ultralytics")
        sys.exit(1)

    # Geçerli örnekleri topla
    samples = collect_valid_samples(DATASET_DIR)
    if len(samples) < 10:
        print("[HATA] Yeterli geçerli örnek bulunamadı!")
        sys.exit(1)

    # ── DATASET BÖLME (tek seferlik, seed döngüsünden önce) ────────────
    data_yaml_path, n_train, n_val = build_global_split(samples, EVIDENCE_DIR)

    all_seed_results = {}
    summary_rows     = []

    for seed in SEEDS:
        print(f"\n{'─'*60}")
        print(f"SEED: {seed}  →  Model: yolo{seed}.pt")
        print(f"{'─'*60}")

        # Her seed için ayrı çalışma dizini (sadece model çıktıları için)
        run_dir = EVIDENCE_DIR / f"seed_{seed}"
        run_dir.mkdir(exist_ok=True)

        # Seed sabitlenmesi
        set_seed(seed)

        # Modeli yükle
        model = YOLO(str(PURE_MODEL))

        # Eğitim
        start_time = datetime.now()
        results = model.train(
            data      = data_yaml_path,
            epochs    = EPOCHS,
            imgsz     = IMG_SIZE,
            batch     = BATCH_SIZE,
            seed      = seed,
            project   = str(run_dir),
            name      = "train",
            exist_ok  = True,
            verbose   = True,
            # Optimize edici & öğrenme oranı (varsayılan SGD + cosine LR)
            optimizer = "SGD",
            lr0       = 0.01,
            lrf       = 0.01,
            momentum  = 0.937,
            weight_decay = 0.0005,
            warmup_epochs = 3,
            warmup_momentum = 0.8,
            # Augmentation
            hsv_h     = 0.015,
            hsv_s     = 0.7,
            hsv_v     = 0.4,
            degrees   = 0.0,
            translate = 0.1,
            scale     = 0.5,
            fliplr    = 0.5,
            mosaic    = 1.0,
            # Kayıt
            save      = True,
            save_period = -1,   # sadece best + last
            plots     = True,
        )

        end_time = datetime.now()
        elapsed  = (end_time - start_time).total_seconds()

        # ── Eğitilen modeli yolo{seed}.pt olarak kaydet ──────────────────
        yolo_train_dir = run_dir / "train" / "weights"
        best_weight    = yolo_train_dir / "best.pt"
        dest_model     = TRAIN_DIR / f"yolo{seed}.pt"

        if best_weight.exists():
            shutil.copy2(best_weight, dest_model)
            print(f"[OK] Model kaydedildi → {dest_model}")
        else:
            # last.pt fallback
            last_weight = yolo_train_dir / "last.pt"
            if last_weight.exists():
                shutil.copy2(last_weight, dest_model)
                print(f"[WARN] best.pt bulunamadı, last.pt kullanıldı → {dest_model}")

        # ── results.csv'den metrikleri oku ──────────────────────────────
        results_csv  = run_dir / "train" / "results.csv"
        epoch_data   = parse_all_epochs_csv(results_csv)
        last_metrics = parse_results_csv(results_csv)

        # F1 hesapla
        last_metrics["f1_score"] = compute_f1(
            last_metrics.get("precision"),
            last_metrics.get("recall")
        )

        # Confusion matrix bilgisi
        cm_info = extract_confusion_matrix_info(run_dir / "train")

        # ── Epoch bazlı LR ve loss serileri ─────────────────────────────
        lr_series, mAP_series, box_loss_series = [], [], []
        for row in epoch_data:
            def rf(key): 
                try: return float(row.get(key, 0) or 0)
                except: return 0.0
            lr_series.append(rf("lr/pg0"))
            mAP_series.append(rf("metrics/mAP50(B)"))
            box_loss_series.append(rf("val/box_loss"))

        # ── Seed sonuç paketi ────────────────────────────────────────────
        seed_result = {
            "seed"         : seed,
            "model_path"   : str(dest_model),
            "n_train"      : n_train,
            "n_val"        : n_val,
            "epochs"       : EPOCHS,
            "img_size"     : IMG_SIZE,
            "batch_size"   : BATCH_SIZE,
            "training_time_seconds": round(elapsed, 2),
            "training_time_human"  : str(end_time - start_time),
            "hyperparameters": {
                "optimizer"       : "SGD",
                "lr0"             : 0.01,
                "lrf"             : 0.01,
                "momentum"        : 0.937,
                "weight_decay"    : 0.0005,
                "warmup_epochs"   : 3,
                "warmup_momentum" : 0.8,
            },
            "metrics": {
                # ── Detection ──
                "mAP50"             : last_metrics.get("mAP50"),
                "mAP50_95"          : last_metrics.get("mAP50_95"),
                "precision"         : last_metrics.get("precision"),
                "recall"            : last_metrics.get("recall"),
                "f1_score"          : last_metrics.get("f1_score"),
                # ── Losses ──
                "train_box_loss"    : last_metrics.get("train_box_loss"),
                "train_cls_loss"    : last_metrics.get("train_cls_loss"),
                "train_dfl_loss"    : last_metrics.get("train_dfl_loss"),
                "val_box_loss"      : last_metrics.get("val_box_loss"),
                "val_cls_loss"      : last_metrics.get("val_cls_loss"),
                "val_dfl_loss"      : last_metrics.get("val_dfl_loss"),
                # ── Learning Rate (son epoch) ──
                "lr_final_pg0"      : last_metrics.get("lr_pg0"),
                "lr_final_pg1"      : last_metrics.get("lr_pg1"),
                "lr_final_pg2"      : last_metrics.get("lr_pg2"),
                # ── Zaman serisi (tüm epoch) ──
                "lr_per_epoch"      : lr_series,
                "mAP50_per_epoch"   : mAP_series,
                "val_box_loss_per_epoch": box_loss_series,
            },
            "confusion_matrix": cm_info,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "timestamp": datetime.now().isoformat(),
        }

        all_seed_results[f"seed_{seed}"] = seed_result

        # CSV özet satırı
        summary_rows.append({
            "seed"            : seed,
            "n_train"         : n_train,
            "n_val"           : n_val,
            "mAP50"           : last_metrics.get("mAP50"),
            "mAP50_95"        : last_metrics.get("mAP50_95"),
            "precision"       : last_metrics.get("precision"),
            "recall"          : last_metrics.get("recall"),
            "f1_score"        : last_metrics.get("f1_score"),
            "train_box_loss"  : last_metrics.get("train_box_loss"),
            "val_box_loss"    : last_metrics.get("val_box_loss"),
            "train_cls_loss"  : last_metrics.get("train_cls_loss"),
            "val_cls_loss"    : last_metrics.get("val_cls_loss"),
            "train_dfl_loss"  : last_metrics.get("train_dfl_loss"),
            "val_dfl_loss"    : last_metrics.get("val_dfl_loss"),
            "lr_final"        : last_metrics.get("lr_pg0"),
            "training_time_s" : round(elapsed, 2),
            "model_path"      : str(dest_model),
        })

        print(f"\n[SEED {seed}] Sonuçlar:")
        print(f"  mAP50      : {last_metrics.get('mAP50')}")
        print(f"  mAP50:95   : {last_metrics.get('mAP50_95')}")
        print(f"  Precision  : {last_metrics.get('precision')}")
        print(f"  Recall     : {last_metrics.get('recall')}")
        print(f"  F1 Score   : {last_metrics.get('f1_score')}")
        print(f"  LR (son)   : {last_metrics.get('lr_pg0')}")
        print(f"  Süre       : {elapsed:.1f}s")

    # ────────────────────────────────────────────────────────────────────
    # ÖZET İSTATİSTİKLER (tüm seedler üzerinde)
    # ────────────────────────────────────────────────────────────────────
    def mean_std(key):
        vals = [r["metrics"].get(key) for r in all_seed_results.values()]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None, None
        arr = np.array(vals)
        return float(np.mean(arr)), float(np.std(arr))

    summary_stats = {}
    for metric_key in ["mAP50", "mAP50_95", "precision", "recall", "f1_score",
                        "train_box_loss", "val_box_loss",
                        "train_cls_loss", "val_cls_loss",
                        "train_dfl_loss", "val_dfl_loss"]:
        mu, sigma = mean_std(metric_key)
        summary_stats[metric_key] = {"mean": mu, "std": sigma}

    # ────────────────────────────────────────────────────────────────────
    # SONUÇLARI KAYDET
    # ────────────────────────────────────────────────────────────────────

    # 1) Kapsamlı JSON raporu
    report = {
        "experiment": {
            "description"        : "YOLOv8n Plaka Tespiti - Çoklu Seed",
            "model_architecture" : "YOLOv8n",
            "base_weights"       : str(PURE_MODEL),
            "dataset_path"       : str(DATASET_DIR),
            "total_samples"      : len(samples),
            "val_split"          : VAL_SPLIT,
            "epochs"             : EPOCHS,
            "img_size"           : IMG_SIZE,
            "batch_size"         : BATCH_SIZE,
            "seeds"              : SEEDS,
            "nc"                 : NC,
            "class_names"        : CLASS_NAMES,
            "generated_at"       : datetime.now().isoformat(),
        },
        "per_seed_results": all_seed_results,
        "summary_statistics": summary_stats,
    }

    report_path = EVIDENCE_DIR / "metrics_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] JSON raporu kaydedildi → {report_path}")

    # 2) CSV özet
    csv_path = EVIDENCE_DIR / "all_seeds_summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[OK] CSV özet kaydedildi → {csv_path}")

    # 3) İnsan okunabilir özet
    print("\n" + "=" * 60)
    print("ÖZET İSTATİSTİKLER (Tüm Seedler Ortalaması ± Std)")
    print("=" * 60)
    for k, v in summary_stats.items():
        mu, sigma = v["mean"], v["std"]
        if mu is not None:
            print(f"  {k:<25} : {mu:.4f} ± {sigma:.4f}")

    print("\n[TAMAMLANDI] Tüm eğitimler bitti!")
    print(f"Sonuçlar: {EVIDENCE_DIR}")


if __name__ == "__main__":
    main()
