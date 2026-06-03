"""
YOLOv8 Plaka Tespiti - Test Scripti
=====================================
- test_car2 klasöründeki görüntüleri 3 farklı seed modeli ile test eder
- Tespit edilen plakaları çerçeveleyip test_results/ klasörüne kaydeder
- Terminale detaylı istatistik loglar
"""

import os
import sys
from pathlib import Path

import cv2

# ─────────────────────────────────────────────
# YOLLAR
# ─────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent          # yolo_train/
TEST_IMG_DIR = BASE_DIR / "test_car2"
MODELS_DIR   = BASE_DIR / "train"
RESULTS_DIR  = BASE_DIR / "test_results"

# ─────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45
IMG_SIZE       = 640

# Çerçeve renkleri (BGR) — her model için farklı renk
MODEL_COLORS = {
    "yolo0":   (0, 255, 0),     # yeşil
    "yolo42":  (255, 165, 0),   # turuncu
    "yolo123": (0, 0, 255),     # kırmızı
}

BOX_THICKNESS = 2
FONT_SCALE    = 0.6
FONT          = cv2.FONT_HERSHEY_SIMPLEX


def get_image_files(img_dir: Path) -> list:
    """Desteklenen formatlardaki tüm görüntü dosyalarını toplar."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in exts])
    return files


def draw_detections(img, results, color, model_name):
    """
    YOLO sonuçlarını görüntü üzerine çizer.
    Dönen: tespit sayısı
    """
    count = 0
    boxes = results[0].boxes
    for box in boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Çerçeve
        cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # Etiket
        label = f"{model_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 4), FONT, FONT_SCALE,
                    (255, 255, 255), 1, cv2.LINE_AA)
        count += 1

    return count


def main():
    # Ultralytics import
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[HATA] ultralytics kurulu değil!")
        sys.exit(1)

    # Test görüntülerini topla
    if not TEST_IMG_DIR.exists():
        print(f"[HATA] Test klasörü bulunamadı: {TEST_IMG_DIR}")
        sys.exit(1)

    images = get_image_files(TEST_IMG_DIR)
    if not images:
        print(f"[HATA] Test klasöründe görüntü bulunamadı: {TEST_IMG_DIR}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"YOLOv8 Test Scripti")
    print(f"Test görüntüleri : {TEST_IMG_DIR}")
    print(f"Toplam görüntü   : {len(images)}")
    print(f"Confidence       : {CONF_THRESHOLD}")
    print(f"{'='*60}\n")

    # Modelleri bul ve yükle
    model_files = sorted(MODELS_DIR.glob("yolo*.pt"))
    if not model_files:
        print(f"[HATA] Model dosyası bulunamadı: {MODELS_DIR}/yolo*.pt")
        sys.exit(1)

    models = {}
    for mf in model_files:
        name = mf.stem  # yolo0, yolo42, yolo123
        models[name] = YOLO(str(mf))
        print(f"[OK] Model yüklendi: {name} → {mf}")

    print()

    # Her model için istatistikler
    stats = {name: {"detected": 0, "total_boxes": 0, "images_with_detection": 0}
             for name in models}

    # Her model için ayrı çıktı klasörü
    for name in models:
        out_dir = RESULTS_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)

    # ─── TEST DÖNGÜSÜ ──────────────────────────────────────────
    for img_path in images:
        img_name = img_path.name

        for model_name, model in models.items():
            # Görüntüyü oku
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [WARN] Okunamadı: {img_name}")
                continue

            # Tahmin
            results = model.predict(
                source=str(img_path),
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=IMG_SIZE,
                verbose=False,
            )

            # Çerçeve çiz
            color = MODEL_COLORS.get(model_name, (0, 255, 0))
            n_det = draw_detections(img, results, color, model_name)

            # İstatistik güncelle
            stats[model_name]["total_boxes"] += n_det
            if n_det > 0:
                stats[model_name]["images_with_detection"] += 1

            # Kaydet
            out_path = RESULTS_DIR / model_name / img_name
            cv2.imwrite(str(out_path), img)

    # ─── SONUÇ TABLOSU ─────────────────────────────────────────
    total_imgs = len(images)

    print(f"\n{'='*60}")
    print(f"{'TEST SONUÇLARI':^60}")
    print(f"{'='*60}")
    print(f"{'Model':<12} {'Tespit/Toplam':<18} {'Oran':<10} {'Toplam Kutu':<12}")
    print(f"{'-'*60}")

    for model_name in sorted(stats.keys()):
        s = stats[model_name]
        detected = s["images_with_detection"]
        total_b  = s["total_boxes"]
        ratio    = f"{detected}/{total_imgs}"
        pct      = f"%{100*detected/total_imgs:.1f}"
        print(f"{model_name:<12} {ratio:<18} {pct:<10} {total_b:<12}")

    print(f"{'-'*60}")
    print(f"Sonuçlar kaydedildi → {RESULTS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
