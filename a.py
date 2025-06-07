from ultralytics import YOLO
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🌿 Green Cover Analyzer", layout="centered")

MODEL_PATH = "yolov8n-seg.pt"

# Class names (make sure these match the classes in your model)
CLASS_NAMES = [
    "unlabeled", "paved-area", "dirt", "grass", "gravel", "water", "rocks", "pool",
    "vegetation", "roof", "wall", "window", "door", "fence", "fence-pole", "person",
    "dog", "car", "bicycle", "tree", "bald-tree", "ar-marker", "obstacle", "conflicting"
]

# Green classes we want to focus on
green_class_ids = [CLASS_NAMES.index(cls) for cls in ["tree", "grass", "vegetation"]]

# Load the trained model
model = YOLO(MODEL_PATH)

# ---------------- HELPERS ----------------
def run_inference(image):
    results = model(image)[0]
    if results.masks is None or results.masks.data is None:
        return None, None
    
    # Get the final mask from the result
    final_mask = results.masks.data.argmax(0).cpu().numpy()

    # Display detected class IDs for debugging
    unique_ids = np.unique(final_mask)
    st.write("📊 Detected class IDs in this image:", unique_ids.tolist())  # Live display
    print("📊 DEBUG: Detected class IDs:", unique_ids)
    
    return final_mask, results.orig_img

def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id in np.unique(mask):
        if class_id in green_class_ids:
            color = (0, 255, 0)  # Green for green classes
        else:
            color = (0, 0, 0)  # Black for non-green areas
        color_mask[mask == class_id] = color

    return color_mask

def overlay_mask(image, mask_color, alpha=0.5):
    if image.shape[:2] != mask_color.shape[:2]:
        mask_color = cv2.resize(mask_color, (image.shape[1], image.shape[0]))
    return cv2.addWeighted(image, 1 - alpha, mask_color, alpha, 0)

def calculate_green_score_from_mask(mask):
    total = mask.size
    green_pixels = np.isin(mask, green_class_ids).sum()
    score = (green_pixels / total) * 100
    return round(score, 2)

# ---------------- UI ----------------
st.title("🌳 Green Score Visualizer from Drone Image")

uploaded_img = st.file_uploader("📸 Upload a drone-style image (.jpg/.png)", type=["jpg", "png"])

if uploaded_img is not None:
    with st.spinner("Running segmentation..."):
        image = Image.open(uploaded_img).convert("RGB")
        img_np = np.array(image)

        final_mask, original_img = run_inference(img_np)

        if final_mask is not None:
            # Calculate green score
            score = calculate_green_score_from_mask(final_mask)
            green_mask = mask_to_color(final_mask)
            overlayed = overlay_mask(img_np, green_mask)

            st.image(image, caption="📷 Uploaded Image", use_column_width=True)
            st.image(green_mask, caption="🌿 Detected Green Areas", use_column_width=True)
            st.image(overlayed, caption="🌀 Overlayed with Green Highlights", use_column_width=True)
            st.metric(label="🟢 Green Score", value=f"{score}%")
        else:
            st.error("❌ Segmentation failed. Model didn't return a valid mask.")
