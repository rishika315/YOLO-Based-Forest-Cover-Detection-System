import os
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Paths
PROJECT_ROOT = Path("C:/Users/91986/OneDrive/Documents/CVAPROJECT")
CSV_PATH = PROJECT_ROOT / "class_dict_seg.csv"
RGB_MASK_DIR = PROJECT_ROOT / "RGB_color_image_masks" / "RGB_color_image_masks"
LABEL_MASK_DIR = PROJECT_ROOT / "labels"
LABEL_MASK_DIR.mkdir(exist_ok=True)

# Load RGB to class ID map
df = pd.read_csv(CSV_PATH)
rgb_to_id = {tuple(row[1:]): idx for idx, row in df.iterrows()}

# Vectorized mask conversion
def rgb_to_mask(mask_array):
    out = np.zeros((mask_array.shape[0], mask_array.shape[1]), dtype=np.uint8)
    for rgb, class_id in rgb_to_id.items():
        match = np.all(mask_array == rgb, axis=-1)
        out[match] = class_id
    return out

def convert_and_save(img_path):
    rgb_mask = Image.open(img_path).convert("RGB")
    rgb_np = np.array(rgb_mask)
    class_mask = rgb_to_mask(rgb_np)
    output_path = LABEL_MASK_DIR / img_path.name
    Image.fromarray(class_mask).save(output_path)

# Process in parallel for speed
mask_files = list(RGB_MASK_DIR.glob("*.png"))
with ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(convert_and_save, mask_files), total=len(mask_files)))
