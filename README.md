# 🌲 Green Forest Analyzer Using YOLO and RGB Masking

## Overview

**Green Forest Analyzer** is a deep learning-powered system designed to detect, segment, and quantify green cover in satellite and aerial imagery using computer vision techniques. By leveraging **YOLO (You Only Look Once)** object detection and custom **RGB mask datasets**, the project enables accurate forest area mapping and monitoring. The tool is especially useful for **environmental surveillance**, **deforestation tracking**, and **ecological research**.

---

## 🌍 Problem Statement

Deforestation, illegal logging, and urban expansion are significantly reducing green cover globally. Traditional remote sensing methods are either manual or require expensive processing. This project presents a **scalable, real-time, AI-driven alternative** that can:

* Detect green forest areas from RGB satellite images.
* Track canopy cover changes over time.
* Generate visual and data-based forest health reports.

---

## 🎯 Objectives

* Detect and localize forest areas in satellite/aerial imagery.
* Generate binary/colored masks showing green cover regions.
* Quantify total area covered by forest.
* Visualize changes across time using pre/post satellite shots (optional extension).
* Enable integration with Google Earth Pro for spatial referencing.

---

## 🧠 Technologies & Tools

| Component                       | Description                                                            |
| ------------------------------- | ---------------------------------------------------------------------- |
| **YOLOv5**                      | Real-time object detection architecture, trained on green cover images |
| **RGB Masks**                   | Color segmentation masks to identify dense foliage areas               |
| **LabelImg**                    | Used for manual annotation of forest regions                           |
| **OpenCV**                      | For preprocessing, filtering, and overlaying masks                     |
| **NumPy / Matplotlib**          | Data computation and visualization of mask overlays                    |
| **Google Earth Pro (optional)** | For spatial verification and result overlay with KML exports           |
| **Python**                      | Core development language                                              |
| **PyTorch**                     | Framework used for training the YOLO model                             |

---

## 🗂️ Dataset

### 📦 Inputs

* High-resolution satellite images (.jpg/.png) sourced from Google Earth or open remote sensing datasets.
* RGB mask images indicating forested vs. non-forested regions.

### 🏷️ Labels

* Format: Pascal VOC or YOLO Darknet
* Classes: `GreenCover` (1 class model)
* Images + Annotations (Train/Test split with augmentation)

### 🌐 Sources

* Custom collected via Google Earth Pro
* Open datasets (e.g., Sentinel-2, LandSat, Kaggle green cover datasets)

---

## 🚀 How It Works

### 1. **Data Annotation**

* Manual annotation of forested areas using **LabelImg**.
* RGB masks generated or edited for ground truth comparison.

### 2. **Model Training**

* YOLOv5 trained on annotated green cover images.
* Loss minimization via bounding box accuracy and IOU.

### 3. **Inference**

* Real-time detection on new satellite images.
* Overlay of predicted forest regions with bounding boxes or masks.

### 4. **Mask Overlay & Analysis**

* Binary masks generated to quantify green area.
* Area (%) computed pixel-wise over total frame.
* Optionally mapped via georeferencing with Google Earth Pro.

---

## 📊 Output & Visualization

* Bounding boxes or shaded regions highlighting green cover.
* Overlay of predicted vs. actual forest masks.
* Area statistics (in % or hectares if georeferenced).
* Optional export of KML file for spatial visualization in Earth viewers.

---

## 🔄 Potential Extensions

* 🌱 **Temporal Change Detection** using pre- and post-disaster or seasonal satellite shots.
* 🛰️ **NDVI Integration** for more accurate vegetation classification.
* 📈 **Dashboard Integration** with Streamlit or Dash for live monitoring.
* 📍 **Geo-referenced Reporting** using Shapefiles or KML exports.

---

## 💡 Applications

* Environmental Impact Assessment (EIA)
* Government Forest Monitoring Agencies
* NGO Conservation Initiatives
* Urban Development Planning
* Agriculture and Plantation Health Analysis

---

## 🛠 Installation

```bash
git clone https://github.com/yourusername/green-forest-analyzer
cd green-forest-analyzer
pip install -r requirements.txt
```

Make sure to also install:

* Python 3.8+
* PyTorch
* OpenCV
* YOLOv5 (submodule or dependency)
* Google Earth Pro (optional)

---

## 🧪 Usage

```bash
# Detect green cover in test image
python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source data/test.jpg
```

```python
# Post-processing mask overlay
from utils.mask_analysis import overlay_mask, calculate_area
```

---

## 📁 Project Structure

```
green-forest-analyzer/
│
├── data/
│   ├── images/
│   ├── masks/
│   └── annotations/
│
├── models/
│   └── yolov5/
│
├── src/
│   ├── training/
│   ├── detection/
│   └── visualization/
│
├── utils/
│   └── mask_analysis.py
│
└── README.md
```

---

## 📜 License

This repository is proprietary and all rights are reserved. No usage, modification, or distribution is allowed without permission.


