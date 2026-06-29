```markdown
<div align="center">

# 🪖 AI-Based Helmet Detection System using YOLOv8

### Real-Time Intelligent Motorcycle Helmet Detection using Deep Learning

<p align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)

![YOLOv8](https://img.shields.io/badge/YOLO-v8-red)

![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-orange?logo=pytorch)

![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-green?logo=opencv)

![CUDA](https://img.shields.io/badge/CUDA-11.8-success?logo=nvidia)

![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red?logo=streamlit)

![License](https://img.shields.io/badge/License-Educational-lightgrey)

</p>

---

### 🚀 Intelligent Traffic Surveillance using Artificial Intelligence

*A high-performance real-time helmet detection system developed using **YOLOv8**, **PyTorch**, **OpenCV**, and **BoT-SORT Object Tracking** to automatically identify motorcycle riders wearing or not wearing safety helmets.*

---

<img src="docs/banner.png" width="100%">

> *(Replace the above image with your project banner or screenshots.)*

---

## 👩‍💻 Author

# **Rudrayani Adichwal**

**Electronics & Telecommunication Engineering**

Artificial Intelligence • Computer Vision • Deep Learning • Embedded Systems

⭐ If you like this project, consider giving it a **Star**.

</div>

---

# 📑 Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Workflow](#workflow)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training the Model](#training-the-model)
- [Running Detection](#running-detection)
- [Streamlit Web Interface](#streamlit-web-interface)
- [Detection Classes](#detection-classes)
- [Performance](#performance)
- [Turbo Fine-Tuning](#turbo-fine-tuning)
- [Applications](#applications)
- [Future Scope](#future-scope)
- [Troubleshooting](#troubleshooting)
- [Frequently Asked Questions](#frequently-asked-questions)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Author](#author)

---

# 📌 Project Overview

Road safety has become one of the most critical challenges worldwide. According to international road safety studies, motorcycle riders are significantly more vulnerable to fatal accidents compared to four-wheel vehicle occupants.

One of the primary reasons for severe injuries during motorcycle accidents is the **failure to wear a protective helmet**.

Traditional monitoring methods rely heavily on traffic police or surveillance personnel who manually inspect riders. These approaches are:

- Time-consuming
- Expensive
- Labour intensive
- Difficult to monitor continuously
- Prone to human errors
- Inefficient in crowded traffic

Artificial Intelligence and Deep Learning provide a scalable solution capable of automatically detecting helmet violations in real time.

The **AI-Based Helmet Detection System** is an intelligent computer vision application that utilizes **YOLOv8**, one of the latest state-of-the-art object detection models, to identify motorcycle riders wearing helmets and those violating helmet safety rules.

The system processes:

- Images
- Videos
- CCTV recordings
- Live webcam streams

and generates accurate predictions with bounding boxes, confidence scores, and real-time statistics.

To further improve robustness, object tracking algorithms such as **BoT-SORT** and **ByteTrack** are integrated to assign persistent IDs to riders, preventing duplicate counting across consecutive frames.

This project demonstrates how modern deep learning techniques can contribute toward **smart city infrastructure**, **intelligent traffic management systems**, and **road safety automation**.

---

# ❗ Problem Statement

Manual helmet compliance monitoring faces several limitations:

- Human fatigue during long surveillance periods
- Difficulty monitoring thousands of vehicles simultaneously
- High manpower requirements
- Inconsistent violation reporting
- Reduced accuracy during nighttime or crowded traffic

An automated AI-based solution is therefore required to:

- Detect helmet violations instantly
- Reduce human intervention
- Improve road safety
- Generate real-time analytics
- Assist traffic authorities in enforcing helmet regulations

---

# 🎯 Objectives

The primary objectives of this project are:

- Develop a real-time helmet detection system.
- Detect riders wearing helmets with high confidence.
- Detect riders without helmets accurately.
- Support image, video, and webcam inputs.
- Enable object tracking using BoT-SORT or ByteTrack.
- Display live statistics.
- Provide a user-friendly Streamlit interface.
- Achieve high detection accuracy with real-time performance.
- Reduce false positives through transfer learning and fine-tuning.
- Build a scalable system suitable for smart city deployment.

---

# ✨ Key Features

## 🟢 Real-Time Helmet Detection

Detect motorcycle riders wearing helmets in real time using YOLOv8 object detection.

---

## 🔴 No Helmet Detection

Automatically identifies riders violating helmet safety regulations and highlights them using red bounding boxes.

---

## 📷 Image Detection

Supports multiple image formats:

- JPG
- JPEG
- PNG
- BMP
- TIFF

---

## 🎥 Video Detection

Supports:

- MP4
- AVI
- MOV
- MKV
- MPEG

Frame-by-frame processing ensures smooth visualization and accurate detection throughout the video.

---

## 📹 Live Webcam Detection

Supports live inference from:

- Laptop Camera
- USB Webcam
- CCTV Camera
- IP Camera

---

## 👥 Multi-Object Detection

Capable of detecting multiple riders simultaneously, even in dense traffic conditions.

---

## 🛰️ Object Tracking

Integrated with:

- BoT-SORT
- ByteTrack

to assign unique tracking IDs for every detected rider.

Benefits include:

- Prevent duplicate counting
- Persistent identity tracking
- Better occlusion handling
- Smooth object transitions

---

## 📊 Live Statistics

Displays:

- Total Riders
- Helmet Count
- No Helmet Count
- Detection Confidence
- FPS
- Processing Time
- Active Tracking IDs

---

## ⚡ GPU Acceleration

Automatically detects CUDA-supported NVIDIA GPUs for significantly faster inference while also supporting CPU execution.

---
```
````markdown
# 🏗️ System Architecture

The Helmet Detection System follows a modular pipeline that combines image preprocessing, deep learning inference, object tracking, visualization, and user interaction.

```text
                    ┌──────────────────────────┐
                    │     Input Source         │
                    │──────────────────────────│
                    │  Image │ Video │ Webcam  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Frame Acquisition Module │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Image Preprocessing    │
                    │──────────────────────────│
                    │ Resize                   │
                    │ Normalize                │
                    │ Convert Color Space      │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │     YOLOv8 Detection      │
                    │──────────────────────────│
                    │ Helmet Detection         │
                    │ No Helmet Detection      │
                    │ Confidence Score         │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Object Tracking Module   │
                    │──────────────────────────│
                    │ BoT-SORT / ByteTrack     │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Statistics Generation    │
                    │──────────────────────────│
                    │ Helmet Count             │
                    │ No Helmet Count          │
                    │ FPS                      │
                    │ Tracking IDs             │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Visualization Module     │
                    │──────────────────────────│
                    │ Bounding Boxes           │
                    │ Labels                   │
                    │ Confidence               │
                    │ Statistics               │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Save / Display Results   │
                    └──────────────────────────┘
```

---

# 🔄 Project Workflow

The complete execution workflow consists of the following stages:

### Step 1 — Input Acquisition

The application accepts input from:

- Static Images
- Recorded Videos
- Live Webcam Feed
- CCTV Camera
- IP Camera

↓

### Step 2 — Image Preprocessing

Every frame undergoes preprocessing before being passed to the neural network.

Operations include:

- Image resizing
- Pixel normalization
- Tensor conversion
- RGB conversion

↓

### Step 3 — YOLOv8 Inference

The pretrained YOLOv8 model detects objects.

Each detected object contains:

- Bounding Box
- Class Label
- Confidence Score

↓

### Step 4 — Helmet Classification

Objects are classified into:

- Helmet
- No Helmet

↓

### Step 5 — Object Tracking

BoT-SORT / ByteTrack assigns a unique ID to every rider.

Example:

```
Helmet #12

Helmet #13

No Helmet #14
```

↓

### Step 6 — Statistics Generation

The application continuously updates

- Total Riders
- Helmet Count
- No Helmet Count
- FPS
- Detection Time

↓

### Step 7 — Visualization

Bounding boxes are drawn using OpenCV.

Helmet

🟩 Green

No Helmet

🟥 Red

↓

### Step 8 — Output

Results can be

- Displayed
- Saved
- Exported

---

# 📁 Project Structure

```
AI-Based-Helmet-Detection-System/
│
├── config/
│   │
│   └── data.yaml
│
├── data/
│   ├── train/
│   ├── valid/
│   ├── test/
│   ├── sample/
│   └── dataset_info.txt
│
├── docs/
│   ├── banner.png
│   ├── architecture.png
│   ├── demo.gif
│   └── screenshots/
│
├── models/
│   ├── best.pt
│   ├── last.pt
│   └── yolov8m.pt
│
├── output/
│   ├── images/
│   ├── videos/
│   ├── snapshots/
│   └── logs/
│
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── detector.py
│   ├── detect.py
│   ├── dataset_manager.py
│   ├── train.py
│   ├── utils.py
│   └── tracker.py
│
├── requirements.txt
├── run.bat
├── README.md
├── LICENSE
└── .gitignore
```

---

# 📂 Folder Description

## config/

Contains all configuration files.

Example:

- Dataset paths
- Class names
- Training configuration

---

## data/

Stores the complete dataset.

Contains

- Training Images
- Validation Images
- Test Images

---

## docs/

Stores documentation assets.

Examples

- Architecture diagrams
- GIF demos
- Screenshots
- Banner images

---

## models/

Stores trained YOLO model weights.

Examples

```
best.pt

last.pt

yolov8m.pt
```

---

## output/

Automatically generated results.

Contains

- Annotated Images
- Annotated Videos
- Snapshots
- Logs

---

## src/

Main source code.

Contains

- Detection Engine
- Training Pipeline
- Streamlit UI
- Utility Functions
- Dataset Manager

---

# 💻 Technology Stack

| Category | Technology |
|------------|------------|
| Programming Language | Python 3.13 |
| AI Model | YOLOv8 Medium |
| Framework | PyTorch |
| Computer Vision | OpenCV |
| Tracking | BoT-SORT |
| Alternative Tracking | ByteTrack |
| Dataset | Roboflow / Kaggle |
| UI | Streamlit |
| Visualization | OpenCV |
| GPU Support | CUDA 11.8 |
| IDE | Visual Studio Code |

---

# 🖥️ Hardware Requirements

Minimum

- Intel Core i5
- 8 GB RAM
- Integrated Graphics
- Windows 10

Recommended

- Intel Core i7 / Ryzen 7
- NVIDIA RTX 3050 or above
- CUDA 11.8
- 16 GB RAM
- SSD Storage

---

# 💾 Software Requirements

- Python 3.13+
- pip
- Git
- VS Code
- CUDA Toolkit (Optional)
- NVIDIA Driver (Optional)

---

# ⚙️ Installation

## Step 1 — Clone Repository

```bash
git clone https://github.com/Ad1thh/AI-Based-Helmet-Detection-System-using-YOLO---Competition.git

cd AI-Based-Helmet-Detection-System-using-YOLO---Competition
```

---

## Step 2 — Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

---

### Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

---

### macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Step 3 — Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 5 — Verify Installation

```bash
python -c "import torch; print(torch.__version__)"
```

If installed correctly, PyTorch version will be displayed.

---

## Step 6 — Verify CUDA (Optional)

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Output

```
True
```

indicates GPU acceleration is available.

---

## Step 7 — Run the Project

```bash
python -m src.detect --webcam
```

If the webcam opens with live helmet detection, the installation is successful.

---
````
---

# 📊 Detection Classes

| Class | Description | Color |
|--------|-------------|-------|
| Helmet | Rider wearing a safety helmet | 🟢 Green |
| No Helmet | Rider not wearing a safety helmet | 🔴 Red |

---

# 📈 Performance

- ⚡ Real-time inference using YOLOv8
- 🎯 High helmet detection accuracy
- 🛰️ Object tracking with BoT-SORT / ByteTrack
- 🚀 GPU acceleration with CUDA support
- 💻 Compatible with both CPU and GPU

---

# 🎯 Applications

- Smart Traffic Monitoring
- Traffic Law Enforcement
- Smart Cities
- Highway Surveillance
- CCTV Analytics
- Industrial Safety Monitoring
- Campus Security

---

# 🚀 Future Scope

- Automatic Number Plate Recognition (ANPR)
- Triple Riding Detection
- Mobile Phone Usage Detection
- Traffic Rule Violation Detection
- TensorRT Optimization
- Jetson Nano Deployment
- Cloud Dashboard
- Mobile Application Support

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push your branch
5. Open a Pull Request

---

# 📜 License

This project is developed for **educational, research, and competition purposes**.

---

# 🙏 Acknowledgements

Special thanks to the open-source community and the developers of:

- Ultralytics YOLOv8
- PyTorch
- OpenCV
- Streamlit
- Roboflow
- ByteTrack
- BoT-SORT

---

# 👩‍💻 Author

**Rudrayani Adichwal*

**Skills**

- Artificial Intelligence
- Computer Vision
- Deep Learning
- Python
- PyTorch
- OpenCV
- YOLOv8

⭐ **If you found this project helpful, don't forget to give it a Star!**