import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


# ============================================================
# 1. Incarcare
# ============================================================

def load_raw_images(raw_path="data/raw", img_size=(48, 48)):
    """
    Încarcă imaginile brute din data/raw/
    Fiecare folder din raw/ reprezintă o clasă (0,1,2,...).
    Returnează două array-uri: images și labels.
    """

    images = []
    labels = []

    print("[INFO] Încarc imaginile din:", raw_path)

    for class_folder in os.listdir(raw_path):
        class_path = os.path.join(raw_path, class_folder)

        if not os.path.isdir(class_path):
            continue  # ignoră fișierele care nu sunt foldere/clase

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(img_size)
                images.append(np.array(img))
                labels.append(int(class_folder))
            except Exception as e:
                print(f"[WARN] Nu pot încărca {img_path}: {e}")

    images = np.array(images)
    labels = np.array(labels)

    print(f"[INFO] Încărcat {len(images)} imagini.")
    return images, labels



# ============================================================
# 2. Normalizare  (0-255 → 0-1)
# ============================================================

def normalize(images):
    """
    Normalizează pixelii imaginilor între 0 și 1.
    """
    images = images.astype("float32") / 255.0
    print("[INFO] Normalizare completă.")
    return images



# ============================================================
# 3. Împărțirea datasetului în train/val/test
# ============================================================

def split_data(images, labels):
    """
    Împarte datele în:
    - 80% train
    - 10% validation
    - 10% test
    Folosește stratificare pentru păstrarea proporțiilor claselor.
    """

    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels,
        test_size=0.20,
        random_state=42,
        stratify=labels
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print("[INFO] Împărțire completă:")
    print(" Train:", len(X_train))
    print(" Validation:", len(X_val))
    print(" Test:", len(X_test))

    return X_train, X_val, X_test, y_train, y_val, y_test



# ============================================================
# 4. Salvarea imaginilor procesate în foldere
# ============================================================

def save_dataset(images, labels, out_dir):
    """
    Salvează imaginile în foldere separate:
    out_dir/clasa/img_xxx.png
    """

    for img, label in zip(images, labels):
        class_folder = os.path.join(out_dir, str(label))
        os.makedirs(class_folder, exist_ok=True)

        img_id = len(os.listdir(class_folder))
        img_path = os.path.join(class_folder, f"img_{img_id}.png")

        Image.fromarray((img * 255).astype("uint8")).save(img_path)

    print(f"[INFO] Salvate imagini în {out_dir}")



# ============================================================
# 5. MAIN – rulează toată preprocesarea
# ============================================================

if __name__ == "__main__":
    raw_images, raw_labels = load_raw_images("data/raw")
    raw_images = normalize(raw_images)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(raw_images, raw_labels)

    save_dataset(X_train, y_train, "data/train")
    save_dataset(X_val, y_val, "data/validation")
    save_dataset(X_test, y_test, "data/test")

    print("\n[INFO] Preprocesare finalizată cu succes!")
