import os
import shutil
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import yaml
from datetime import datetime
from ultralytics import YOLO

# --- CONFIG ---
PROJECT_ROOT = Path("C:/Users/91986/OneDrive/Documents/CVAPROJECT")
IMG_DIR = PROJECT_ROOT / "dataset/semantic_drone_dataset/original_images"
MASK_DIR = PROJECT_ROOT / "labels"
YOLO_DIR = PROJECT_ROOT / "yolo_dataset"
YOLO_IMAGES = YOLO_DIR / "images"
YOLO_LABELS = YOLO_DIR / "labels"
TRAIN_SPLIT = 0.8
EPOCHS = 30
IMG_SIZE = 640
BATCH_SIZE = 4

GREEN_CLASSES = ["tree", "grass", "vegetation"]
CLASS_NAMES = [
    "unlabeled", "paved-area", "dirt", "grass", "gravel", "water", "rocks", "pool",
    "vegetation", "roof", "wall", "window", "door", "fence", "fence-pole", "person",
    "dog", "car", "bicycle", "tree", "bald-tree", "ar-marker", "obstacle", "conflicting"
]
GREEN_CLASS_IDS = [CLASS_NAMES.index(cls) for cls in GREEN_CLASSES]

# --- SAFE REMOVE ---
def force_remove_dir(path):
    if path.exists():
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                try:
                    os.remove(Path(root) / name)
                except PermissionError:
                    pass
            for name in dirs:
                try:
                    os.rmdir(Path(root) / name)
                except PermissionError:
                    pass
        try:
            shutil.rmtree(path)
        except Exception:
            pass

force_remove_dir(YOLO_DIR)
YOLO_IMAGES.mkdir(parents=True, exist_ok=True)
YOLO_LABELS.mkdir(parents=True, exist_ok=True)

# --- COPY IMAGES & CONVERT MASKS ---
image_files = list(IMG_DIR.glob("*.jpg"))
num_train = int(len(image_files) * TRAIN_SPLIT)
label_counts = {cls_id: 0 for cls_id in GREEN_CLASS_IDS}

for idx, img_path in enumerate(tqdm(image_files, desc="Preparing YOLO dataset")):
    label_name = img_path.stem + ".png"
    mask_path = MASK_DIR / label_name
    if not mask_path.exists():
        continue

    split = "train" if idx < num_train else "val"
    (YOLO_IMAGES / split).mkdir(parents=True, exist_ok=True)
    (YOLO_LABELS / split).mkdir(parents=True, exist_ok=True)

    new_img_path = YOLO_IMAGES / split / img_path.name
    shutil.copy(img_path, new_img_path)

    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask, dtype=np.uint8)
    
    h, w = mask_np.shape
    objects = []

    for class_id in np.unique(mask_np):
        if class_id not in GREEN_CLASS_IDS:
            continue
        binary_mask = (mask_np == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if contour.shape[0] >= 6 and area > 50:
                norm_contour = contour.squeeze().astype(float)
                norm_contour[:, 0] /= w
                norm_contour[:, 1] /= h
                flat = norm_contour.flatten()
                coords = " ".join(map(str, flat))
                line = f"{class_id} {coords}"
                objects.append(line)
                label_counts[class_id] += 1

    if objects:
        yolo_label_path = YOLO_LABELS / split / (img_path.stem + ".txt")
        with open(yolo_label_path, "w") as f:
            for obj in objects:
                f.write(obj + "\n")
    else:
        new_img_path.unlink()  # Remove image with no green object

# --- LOG LABEL COUNTS ---
print("\n📊 Green class label counts:")
for class_id in GREEN_CLASS_IDS:
    print(f"  - {CLASS_NAMES[class_id]}: {label_counts[class_id]} objects")

# --- WRITE data.yaml ---
DATA_YAML_PATH = PROJECT_ROOT / "data.yaml"
data_yaml = {
    "path": str(YOLO_DIR),
    "train": "images/train",
    "val": "images/val",
    "nc": len(CLASS_NAMES),
    "names": CLASS_NAMES,
    "green_class_ids": GREEN_CLASS_IDS  # Optional metadata
}
with open(DATA_YAML_PATH, "w") as f:
    yaml.dump(data_yaml, f)

# --- TRAIN YOLOv8 SEGMENTATION ---
print("\n🚀 Starting training...")
model = YOLO("yolov8n-seg.pt")
model.train(data=str(DATA_YAML_PATH), epochs=18, imgsz=416, batch=8, cache=True)

# --- SAVE MODEL ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = PROJECT_ROOT / f"yolo_{timestamp}.pt"
model.save(str(output_path.resolve()))

print(f"\n✅ Training complete. Model saved as '{output_path.name}'")
