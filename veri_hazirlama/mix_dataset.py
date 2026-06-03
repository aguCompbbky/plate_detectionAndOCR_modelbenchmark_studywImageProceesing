"""
Dataset Karıştırma Scripti
==========================
dataset/ ve random_background/ verilerini MIXEDDATASETklasörüne karıştırır.
- dataset/images/ içindeki tüm .jpg + .txt dosyaları mixeddataset/images/ klasörüne kopyalanır
- dataset/label/ içindeki tüm .txt dosyaları mixeddataset/label/ klasörüne kopyalanır
- random_background/ içindeki resimler mixeddataset/images/ klasörüne kopyalanır
- Her random_background resmi için BOŞ label (.txt) dosyası oluşturulur

Kullanım: python mix_dataset.py
"""

import os
import shutil

# Yollar
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_IMAGES_DIR  = os.path.join(BASE_DIR, "dataset", "images")
DATASET_LABEL_DIR   = os.path.join(BASE_DIR, "dataset", "label")
RANDOM_BG_DIR       = os.path.join(BASE_DIR, "random_background")
OUTPUT_IMAGES_DIR   = os.path.join(BASE_DIR, "mixeddataset", "images")
OUTPUT_LABEL_DIR    = os.path.join(BASE_DIR, "mixeddataset", "label")

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif')


def get_max_image_id(images_dir):
    """Klasördeki en büyük sayısal .jpg/.txt ID'yi döndür."""
    max_id = 0
    for fname in os.listdir(images_dir):
        base, ext = os.path.splitext(fname)
        if ext.lower() in IMAGE_EXTENSIONS:
            try:
                num = int(base)
                if num > max_id:
                    max_id = num
            except ValueError:
                pass
    return max_id


def main():
    print("=" * 65)
    print("  Dataset + Random Background → mixeddataset")
    print("=" * 65)

    # Hedef klasörleri oluştur
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR,  exist_ok=True)
    print(f"[OK] mixeddataset/images/ ve mixeddataset/label/ klasörleri hazır.\n")

    # ── 1. ADIM: dataset/images/ → mixeddataset/images/ ──────────────────────
    print("[1/3] dataset/images/ klasörü kopyalanıyor...")
    img_copied = 0
    for fname in os.listdir(DATASET_IMAGES_DIR):
        src = os.path.join(DATASET_IMAGES_DIR, fname)
        dst = os.path.join(OUTPUT_IMAGES_DIR, fname)
        shutil.copy2(src, dst)
        img_copied += 1
    print(f"      {img_copied} dosya kopyalandı.\n")

    # ── 2. ADIM: dataset/label/ → mixeddataset/label/ ────────────────────────
    print("[2/3] dataset/label/ klasörü kopyalanıyor...")
    lbl_copied = 0
    for fname in os.listdir(DATASET_LABEL_DIR):
        src = os.path.join(DATASET_LABEL_DIR, fname)
        dst = os.path.join(OUTPUT_LABEL_DIR, fname)
        shutil.copy2(src, dst)
        lbl_copied += 1
    print(f"      {lbl_copied} dosya kopyalandı.\n")

    # ── 3. ADIM: random_background → mixeddataset/images/ (boş label ile) ────
    print("[3/3] random_background resimleri ekleniyor (boş label ile)...")

    max_id = get_max_image_id(OUTPUT_IMAGES_DIR)
    print(f"      Mevcut en büyük ID: {max_id}")

    bg_images = sorted([
        f for f in os.listdir(RANDOM_BG_DIR)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ])
    print(f"      {len(bg_images)} random_background resmi bulundu.")

    added = 0
    for fname in bg_images:
        new_id = max_id + added + 1
        ext    = os.path.splitext(fname)[1].lower()

        # Resmi kopyala → mixeddataset/images/
        src_img = os.path.join(RANDOM_BG_DIR, fname)
        dst_img = os.path.join(OUTPUT_IMAGES_DIR, f"{new_id}{ext}")
        shutil.copy2(src_img, dst_img)

        # Boş label → mixeddataset/images/ (images klasöründe de txt bekleniyor)
        dst_lbl_img = os.path.join(OUTPUT_IMAGES_DIR, f"{new_id}.txt")
        open(dst_lbl_img, 'w').close()

        # Boş label → mixeddataset/label/
        dst_lbl_dir = os.path.join(OUTPUT_LABEL_DIR, f"{new_id}.txt")
        open(dst_lbl_dir, 'w').close()

        print(f"      [{new_id}] {fname}")
        added += 1

    print()
    print("=" * 65)
    print(f"[TAMAMLANDI]")
    print(f"  dataset/images/  dosya sayısı      : {img_copied}")
    print(f"  dataset/label/   dosya sayısı      : {lbl_copied}")
    print(f"  random_background eklenen resim    : {added}")
    print(f"  Yeni ID aralığı                    : {max_id+1} → {max_id+added}")
    total_images = img_copied // 2 + added   # her çift: resim + txt
    print(f"  mixeddataset/images/ toplam dosya  : {img_copied + added*2}")
    print(f"  mixeddataset/label/  toplam dosya  : {lbl_copied + added}")
    print("=" * 65)


if __name__ == "__main__":
    main()
