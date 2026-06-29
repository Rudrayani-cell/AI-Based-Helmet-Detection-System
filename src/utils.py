"""
Utility Functions
=================
Helper functions for drawing bounding boxes, resizing frames,
logging setup, and other common operations used across the project.
"""

import cv2
import logging
import os
import sys
import numpy as np
from datetime import datetime

from src.config import (
    CLASS_NAMES, CLASS_COLORS, LABEL_BG_COLORS,
    FONT_SCALE, FONT_THICKNESS, BOX_THICKNESS,
    MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT, OUTPUT_DIR
)


def setup_logger(name="helmet_detection", log_file=None):
    """
    Set up a structured logger with console and optional file output.

    Args:
        name (str): Logger name.
        log_file (str): Optional path to a log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(funcName)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def get_color(class_id):
    """
    Get the bounding box color for a given class ID.

    Args:
        class_id (int): The class index (0=Helmet, 1=No Helmet).

    Returns:
        tuple: BGR color tuple for OpenCV.
    """
    return CLASS_COLORS.get(class_id, (255, 255, 255))


def get_label(class_id, confidence, track_id=None):
    """
    Generate the display label for a detection.

    Args:
        class_id (int): The class index.
        confidence (float): Detection confidence score (0-1).
        track_id (int, optional): Unique ID assigned by the tracker.

    Returns:
        str: Formatted label string, e.g., "Helmet #5 95.2%"
    """
    class_name = CLASS_NAMES.get(class_id, "Unknown")
    
    if track_id is not None:
        return f"{class_name} #{int(track_id)} {confidence * 100:.1f}%"
    return f"{class_name} {confidence * 100:.1f}%"


def draw_detections(frame, detections):
    """
    Draw bounding boxes and labels on a frame for all detections.

    Each detection is rendered with:
    - A colored bounding box (green=Helmet, red=No Helmet)
    - A label with class name and confidence score (and track ID if available)
    - A semi-transparent label background for readability

    Args:
        frame (np.ndarray): The image/frame to annotate (modified in-place).
        detections (list): List of detection dicts with keys:
            - 'bbox': [x1, y1, x2, y2] pixel coordinates
            - 'confidence': float (0-1)
            - 'class_id': int (0 or 1)
            - 'track_id': int (optional) assigned tracker ID.

    Returns:
        np.ndarray: The annotated frame.
    """
    for det in detections:
        class_id = det['class_id']
        # Box color and labeling logic
        x1, y1, x2, y2 = [int(c) for c in det['bbox']]
        confidence = det['confidence']
        track_id = det.get('track_id', None)

        # Get colors
        box_color = get_color(class_id)
        label_bg = LABEL_BG_COLORS.get(class_id, (50, 50, 50))

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, BOX_THICKNESS)

        # Prepare label text
        label = get_label(class_id, confidence, track_id)

        # Calculate label size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, FONT_SCALE, FONT_THICKNESS
        )

        # Label background rectangle (above the bounding box)
        label_y1 = max(y1 - text_h - 10, 0)
        label_y2 = y1
        label_x2 = min(x1 + text_w + 10, frame.shape[1])

        # Draw filled rectangle for label background
        cv2.rectangle(
            frame, (x1, label_y1), (label_x2, label_y2),
            label_bg, -1  # Filled
        )

        # Draw label text
        cv2.putText(
            frame, label, (x1 + 5, y1 - 5),
            font, FONT_SCALE, (255, 255, 255),
            FONT_THICKNESS, cv2.LINE_AA
        )

    return frame


def draw_stats_overlay(frame, detections):
    """
    Draw a statistics overlay on the top-left corner of the frame.
    Shows the count of helmets and no-helmets detected.

    Args:
        frame (np.ndarray): The image/frame to annotate.
        detections (list): List of detection dicts.

    Returns:
        np.ndarray: The annotated frame.
    """
    helmet_count = sum(1 for d in detections if d['class_id'] == 0)
    no_helmet_count = sum(1 for d in detections if d['class_id'] == 1)
    total = len(detections)

    # Semi-transparent overlay background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, 110), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Stats text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Total Detections: {total}", (20, 35),
                font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Helmet: {helmet_count}", (20, 60),
                font, 0.55, (0, 200, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"No Helmet: {no_helmet_count}", (20, 85),
                font, 0.55, (0, 0, 220), 1, cv2.LINE_AA)

    return frame


def resize_frame(frame, max_width=MAX_DISPLAY_WIDTH, max_height=MAX_DISPLAY_HEIGHT):
    """
    Resize a frame to fit within maximum display dimensions while
    maintaining aspect ratio.

    Args:
        frame (np.ndarray): Input image/frame.
        max_width (int): Maximum display width.
        max_height (int): Maximum display height.

    Returns:
        np.ndarray: Resized frame (or original if already within bounds).
    """
    h, w = frame.shape[:2]

    if w <= max_width and h <= max_height:
        return frame

    # Calculate scale factor
    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def save_output(frame, source_name="detection", output_dir=OUTPUT_DIR):
    """
    Save an annotated frame to the output directory.

    Args:
        frame (np.ndarray): Annotated image to save.
        source_name (str): Base name for the output file.
        output_dir (str): Directory to save the output.

    Returns:
        str: Path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{source_name}_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)

    cv2.imwrite(filepath, frame)
    return filepath


def calculate_fps(prev_time):
    """
    Calculate frames per second securely using time.time().
    """
    import time
    current_time = time.time()
    time_elapsed = current_time - prev_time
    fps = 1.0 / time_elapsed if time_elapsed > 0 else 0.0
    return fps, current_time


def draw_fps(frame, fps):
    """
    Draw FPS counter on the bottom-left of the frame.

    Args:
        frame (np.ndarray): Image/frame to annotate.
        fps (float): Current FPS value.

    Returns:
        np.ndarray: Annotated frame.
    """
    h = frame.shape[0]
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255),
        2, cv2.LINE_AA
    )
    return frame
