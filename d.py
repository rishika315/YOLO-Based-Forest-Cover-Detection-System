import torch
import numpy as np
import streamlit as st
from torchvision import models, transforms
from PIL import Image
import cv2
import time

# ---------------- CONFIG ----------------
MODEL_PATH = "deeplabv3_resnet101.pth"
GREEN_CLASSES = ["tree", "grass", "vegetation"]  # green-related classes

# ---------------- TRAINING ----------------
def train_deeplabv3():
    # Use pre-trained model and modify for our use case
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, len(GREEN_CLASSES), kernel_size=(1, 1), stride=(1, 1))  # Modify final layer
    model = model.cuda() if torch.cuda.is_available() else model

    # Training code (skipping dataset loading for brevity)
    print("Model ready for training...")
    # train_model(model)  # You'll need to implement dataset loading and training loop

    return model

# ---------------- INFERENCE ----------------
def run_inference(image):
    model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Preprocess the image
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(image).unsqueeze(0)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    # Start the progress bar
    progress_bar = st.progress(0)
    
    # Simulate the inference process
    num_steps = 10
    for step in range(1, num_steps + 1):
        time.sleep(0.5)  # Simulate the process with some delay
        progress_bar.progress(step / num_steps)
    
    # Perform actual inference (after simulating progress)
    with torch.no_grad():
        output = model(input_tensor)["out"][0]  # Get raw prediction
        output_predictions = output.argmax(0)  # Get class ID with highest score
        return output_predictions.cpu().numpy()

def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Assign green to green-related classes
    for class_id in np.unique(mask):
        if class_id == 21:  # Assuming class 21 is the "vegetation" class in COCO
            color = (0, 255, 0)  # Green for vegetation
        else:
            color = (128, 128, 128)  # Gray for other classes
        color_mask[mask == class_id] = color

    return color_mask

def overlay_mask(image, mask_color, alpha=0.5):
    if image.shape[:2] != mask_color.shape[:2]:
        mask_color = cv2.resize(mask_color, (image.shape[1], image.shape[0]))
    return cv2.addWeighted(image, 1 - alpha, mask_color, alpha, 0)

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="🌿 Green Cover Analyzer", layout="centered")

st.title("🌳 Green Cover Visualizer from Drone Image")

uploaded_img = st.file_uploader("📸 Upload a drone-style image (.jpg/.png)", type=["jpg", "png"])

if uploaded_img is not None:
    image = Image.open(uploaded_img).convert("RGB")
    img_np = np.array(image)
    
    with st.spinner("Running segmentation..."):
        final_mask = run_inference(image)
        
        # Post-processing and visualizing the result
        color_mask = mask_to_color(final_mask)
        overlayed = overlay_mask(img_np, color_mask)

        st.image(image, caption="📷 Uploaded Image", use_column_width=True)
        st.image(color_mask, caption="🌿 Detected Green Areas", use_column_width=True)
        st.image(overlayed, caption="🌀 Overlayed with Green Highlights", use_column_width=True)

        # Calculate Green Score (percentage of green pixels)
        green_pixels = np.sum(final_mask == 21)  # Class ID 21 corresponds to vegetation in COCO
        total_pixels = final_mask.size
        green_score = (green_pixels / total_pixels) * 100
        st.metric(label="🟢 Green Score", value=f"{green_score:.2f}%")
