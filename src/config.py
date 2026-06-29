"""
Configuration Module
====================
Central configuration for the Helmet Detection System.
Contains all constants, paths, model settings, and detection parameters.
"""

import os
import torch

# ============================================================
# Project Paths
# ============================================================
# Root directory of the project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory paths
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")

# Dataset configuration file
DATA_YAML = os.path.join(CONFIG_DIR, "data.yaml")

# ============================================================
# Model Configuration
# ============================================================
# Pre-trained YOLO model to use as a base for fine-tuning
BASE_MODEL = "yolov8n.pt"  # YOLOv8 Nano (fast, good for real-time)

# Path to the trained/fine-tuned model weights
TRAINED_MODEL_PATH = os.path.join(MODELS_DIR, "best.pt")

# Fallback: use the base model if no trained model exists
def get_model_path():
    """
    Returns the best available model path.
    Prefers trained model, falls back to base YOLOv8 model.
    """
    if os.path.exists(TRAINED_MODEL_PATH):
        return TRAINED_MODEL_PATH
    return BASE_MODEL

# ============================================================
# Detection Parameters
# ============================================================
# Minimum confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.5

# IoU (Intersection over Union) threshold for Non-Max Suppression
IOU_THRESHOLD = 0.45

# Input image size for YOLO inference
INPUT_SIZE = 640

# ============================================================
# Class Configuration
# ============================================================
# Class names matching data.yaml
CLASS_NAMES = {
    0: "Helmet",
    1: "No Helmet"
}

# Colors for bounding boxes (BGR format for OpenCV)
CLASS_COLORS = {
    0: (0, 200, 0),       # Green for Helmet
    1: (0, 0, 220),       # Red for No Helmet
}

# Label background colors (slightly darker for readability)
LABEL_BG_COLORS = {
    0: (0, 160, 0),       # Dark green
    1: (0, 0, 180),       # Dark red
}

# Number of classes
NUM_CLASSES = 2

# ============================================================
# Device Configuration
# ============================================================
def get_device():
    """
    Auto-detect the best available compute device.
    Returns 'cuda' if NVIDIA GPU is available, else 'cpu'.
    """
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[INFO] Using GPU: {gpu_name}")
    else:
        device = "cpu"
        print("[INFO] Using CPU (no GPU detected)")
    return device

DEVICE = get_device()

# ============================================================
# Training Hyperparameters
# ============================================================
TRAIN_EPOCHS = 50
TRAIN_BATCH_SIZE = 16
TRAIN_IMG_SIZE = 640
TRAIN_PATIENCE = 10        # Early stopping patience
TRAIN_WORKERS = 4          # DataLoader workers

# ============================================================
# Display Settings
# ============================================================
# Font settings for OpenCV text rendering
FONT_SCALE = 0.6
FONT_THICKNESS = 2
BOX_THICKNESS = 2

# Window name for OpenCV display
WINDOW_NAME = "Helmet Detection - YOLOv8"

# Maximum display resolution (resize if larger)
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720
