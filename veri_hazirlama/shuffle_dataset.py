"""
mixeddataset Shuffle Scripti
============================
mixeddataset/images/ içindeki tüm resim+label çiftlerini rastgele karıştırır
ve 1'den başlayarak yeniden numaralandırır.
mixeddataset/label/ de buna göre güncellenir.

Kullanım: python shuffle_dataset.py
"""

import os
import random
import shutil

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR        = os.path.join(BASE_DIR, "mixeddataset", "images")
LABEL_DIR         = os.path.join(BASE_DIR, "mixeddataset", "label")
TMP_IMAGES_DIR    = os.path.join(BASE_DIR, "mixeddataset", "_tmp_images")
TMP_LABEL_DIR     = os.path.join(BASE_DIR, "mixeddataset", "_tmp_label")

IMAGE_EXTENSIONS  = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif')

random.seed(42)   # tekrarlanabilirlik için; istersen kaldır

def collect_pairs(images_dir):
    """images_dir içinden (resim_dosyası, label_dosyası) çiftlerini topla."""
    pairs = []
    for fname in os.listdir(images_dir):
        base, ext = os.path.splitext(fname)
        if ext.lower() in IMAGE_EXTENSIONS:
            lbl = os.path.join(images_dir, base + ".txt")
            img = os.path.join(images_dir, fname)
            pairs.append((img, lbl if os.path.exists(lbl) else None))
    return pairs

def main():
    print("=" * 65)
    print("  mixeddataset karıştırma (shuffle) başlıyor...")
    print("=" * 65)

    # Çiftleri topla
    pairs = collect_pairs(IMAGES_DIR)
    print(f"[BİLGİ] Toplam {len(pairs)} resim+label çifti bulundu.")

    # Rastgele karıştır
    random.shuffle(pairs)
    print(f"[BİLGİ] Karıştırma tamamlandı.")

    # Geçici klasörler
    os.makedirs(TMP_IMAGES_DIR, exist_ok=True)
    os.makedirs(TMP_LABEL_DIR,  exist_ok=True)

    print(f"[1/3] Yeni sırayla geçici klasöre yazılıyor...")
    for new_id, (img_path, lbl_path) in enumerate(pairs, start=1):
        ext = os.path.splitext(img_path)[1].lower()

        # Resim → tmp
        shutil.move(img_path, os.path.join(TMP_IMAGES_DIR, f"{new_id}{ext}"))

        # Label → tmp/images
        if lbl_path and os.path.exists(lbl_path):
            shutil.move(lbl_path, os.path.join(TMP_IMAGES_DIR, f"{new_id}.txt"))
        else:
            open(os.path.join(TMP_IMAGES_DIR, f"{new_id}.txt"), 'w').close()

        # Label → tmp/label
        src_lbl_in_label_dir = os.path.join(LABEL_DIR, os.path.basename(img_path).rsplit('.', 1)[0] + ".txt")
        if os.path.exists(src_lbl_in_label_dir):
            shutil.move(src_lbl_in_label_dir, os.path.join(TMP_LABEL_DIR, f"{new_id}.txt"))
        else:
            open(os.path.join(TMP_LABEL_DIR, f"{new_id}.txt"), 'w').close()

    print(f"[2/3] Eski klasörlerdeki kalan dosyalar temizleniyor...")
    # Kalan eski dosyaları temizle (etiket vb.)
    for f in os.listdir(IMAGES_DIR):
        os.remove(os.path.join(IMAGES_DIR, f))
    for f in os.listdir(LABEL_DIR):
        os.remove(os.path.join(LABEL_DIR, f))

    print(f"[3/3] Geçici klasörden ana klasöre taşınıyor...")
    for f in os.listdir(TMP_IMAGES_DIR):
        shutil.move(os.path.join(TMP_IMAGES_DIR, f), os.path.join(IMAGES_DIR, f))
    for f in os.listdir(TMP_LABEL_DIR):
        shutil.move(os.path.join(TMP_LABEL_DIR, f), os.path.join(LABEL_DIR, f))

    # Geçici klasörleri sil
    shutil.rmtree(TMP_IMAGES_DIR)
    shutil.rmtree(TMP_LABEL_DIR)

    print()
    print("=" * 65)
    print(f"[TAMAMLANDI] {len(pairs)} çift rastgele karıştırıldı ve 1-{len(pairs)} arası yeniden numaralandırıldı.")
    print("=" * 65)

if __name__ == "__main__":
    main()
