# 🪖 AI-Based Helmet Detection System using YOLOv8

> **Real-Time Motorcycle Helmet Detection using YOLOv8, OpenCV, PyTorch, and BoT-SORT**

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

## 📌 Overview

The **AI-Based Helmet Detection System** is a real-time computer vision application that detects whether motorcycle riders are wearing helmets using **YOLOv8**. It supports **images, videos, webcams, CCTV, and IP cameras**, making it suitable for intelligent traffic monitoring and smart city applications.

The system also integrates **BoT-SORT / ByteTrack** object tracking to maintain unique rider IDs and avoid duplicate counting.

---

## ✨ Features

- 🪖 Real-time Helmet & No Helmet Detection
- 📷 Image, Video & Webcam Support
- 🎥 CCTV & IP Camera Compatibility
- 👥 Multi-Rider Detection
- 🛰️ BoT-SORT / ByteTrack Object Tracking
- 📊 Live Statistics (Helmet Count, No Helmet Count, FPS)
- ⚡ CUDA GPU Acceleration
- 🌐 Streamlit Web Interface
- 💾 Save Detection Results

---

## 🏗️ System Workflow

```text
Input (Image / Video / Webcam)
            │
            ▼
   Image Preprocessing
            │
            ▼
     YOLOv8 Detection
            │
            ▼
 Helmet / No Helmet Classification
            │
            ▼
 Object Tracking (BoT-SORT)
            │
            ▼
 Statistics & Visualization
            │
            ▼
      Output / Save Results
```

---

## 📁 Project Structure

```text
AI-Based-Helmet-Detection-System/
│
├── config/
│   └── data.yaml
│
├── data/
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── sample/
│
├── docs/
│   ├── banner.png
│   ├── architecture.png
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
│   └── logs/
│
├── src/
│   ├── app.py
│   ├── detect.py
│   ├── detector.py
│   ├── tracker.py
│   ├── train.py
│   └── utils.py
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.13 |
| Model | YOLOv8 |
| Framework | PyTorch |
| Computer Vision | OpenCV |
| Tracking | BoT-SORT / ByteTrack |
| UI | Streamlit |
| GPU | CUDA 11.8 |

---

## 💻 Requirements

### Hardware

- Intel Core i5 (Minimum)
- 8 GB RAM
- NVIDIA GPU (Recommended)
- Windows / Linux / macOS

### Software

- Python 3.13+
- Git
- pip
- OpenCV
- PyTorch

---

# ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/Ad1thh/AI-Based-Helmet-Detection-System-using-YOLO---Competition.git

cd AI-Based-Helmet-Detection-System-using-YOLO---Competition
```

### Create Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Verify PyTorch

```bash
python -c "import torch; print(torch.__version__)"
```

### Check CUDA (Optional)

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

# 🚀 Usage

### Webcam Detection

```bash
python -m src.detect --webcam
```

### Image Detection

```bash
python -m src.detect --source path/to/image.jpg
```

### Video Detection

```bash
python -m src.detect --source path/to/video.mp4
```

### Train Model

```bash
python -m src.train
```

### Launch Streamlit App

```bash
streamlit run src/app.py
```

---

## 📊 Detection Classes

| Class | Color |
|--------|-------|
| 🟢 Helmet | Green |
| 🔴 No Helmet | Red |

---

## 🎯 Applications

- 🚦 Smart Traffic Monitoring
- 🚔 Traffic Law Enforcement
- 🏙️ Smart City Surveillance
- 🎥 CCTV Analytics
- 🛣️ Highway Monitoring
- 🏭 Industrial Safety
- 🏫 Campus Security

---

## 🚀 Future Improvements

- Automatic Number Plate Recognition (ANPR)
- Triple Riding Detection
- Mobile Phone Detection
- Traffic Rule Violation Detection
- TensorRT Optimization
- Jetson Nano Deployment
- Cloud Dashboard
- Mobile Application

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Open a Pull Request

---

## 📜 License

This project is intended for **educational, research, and competition purposes**.

---

## 🙏 Acknowledgements

Special thanks to the open-source community and the developers of:

- Ultralytics YOLOv8
- PyTorch
- OpenCV
- Streamlit
- Roboflow
- BoT-SORT
- ByteTrack

---

## 👩‍💻 Author

### **Rudrayani Adichwal**

**Skills**

- Artificial Intelligence
- Computer Vision
- Deep Learning
- Python
- PyTorch
- OpenCV
- YOLOv8

---

<p align="center">
⭐ If you found this project helpful, consider giving it a <b>Star</b> on GitHub!
</p>
