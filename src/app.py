"""
Streamlit Demo Application
==========================
Interactive web-based demo for the Helmet Detection System.
Provides a user-friendly interface for uploading images, running
detection, and viewing results — ideal for live demonstrations.

Usage:
    streamlit run src/app.py
"""

import streamlit as st
import cv2
import numpy as np
import os
import sys
import tempfile
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import HelmetDetector
from src.config import CLASS_NAMES, CLASS_COLORS, CONFIDENCE_THRESHOLD


# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Helmet Detection System - YOLOv8",
    page_icon="⛑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom Styling
# ============================================================
st.markdown("""
<style>
    /* Main header styling */
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-top: 0;
    }

    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #333;
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    .stat-label {
        color: #aaa;
        font-size: 0.9rem;
        margin: 0;
    }

    /* Detection badges */
    .helmet-badge {
        background-color: #00c853;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 2px;
    }
    .no-helmet-badge {
        background-color: #ff1744;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 2px;
    }

    /* Sidebar styling */
    .sidebar-info {
        background: #1a1a2e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
        margin-bottom: 10px;
    }

    /* Hide Streamlit menu & footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State Initialization
# ============================================================
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'detections' not in st.session_state:
    st.session_state.detections = []


@st.cache_resource
def load_detector(confidence):
    """Load and cache the YOLO detector model."""
    return HelmetDetector(confidence=confidence)


def process_uploaded_image(uploaded_file, detector):
    """
    Process an uploaded image through the helmet detector.

    Args:
        uploaded_file: Streamlit uploaded file object.
        detector: HelmetDetector instance.

    Returns:
        tuple: (original_image, annotated_image, detections)
    """
    # Read image from upload
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Store original
    original = frame.copy()

    # Run detection
    detections = detector.detect_frame(frame)

    # Draw results
    from src.utils import draw_detections, draw_stats_overlay
    annotated = draw_detections(frame.copy(), detections)

    # Convert BGR to RGB for Streamlit display
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return original_rgb, annotated_rgb, detections


def process_video_file(uploaded_video, detector):
    """
    Process an uploaded video through the helmet detector.

    Args:
        uploaded_video: Streamlit uploaded video file object.
        detector: HelmetDetector instance.

    Returns:
        str: Path to the processed video, or None.
    """
    # Save uploaded video to a temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error("Could not open video file.")
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    progress_bar = st.progress(0, text="Processing video...")
    frame_placeholder = st.empty()
    frame_count = 0

    from src.utils import draw_detections, draw_stats_overlay

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        detections = detector.detect_frame(frame)
        annotated = draw_detections(frame, detections)
        annotated = draw_stats_overlay(annotated, detections)

        writer.write(annotated)

        # Update progress
        progress = frame_count / total_frames
        progress_bar.progress(progress, text=f"Processing frame {frame_count}/{total_frames}")

        # Show every 10th frame in the UI
        if frame_count % 10 == 0:
            display_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(display_rgb, channels="RGB", use_container_width=True)

    cap.release()
    writer.release()
    progress_bar.empty()
    frame_placeholder.empty()

    # Clean up temp input
    os.unlink(tfile.name)

    return output_path


# ============================================================
# Main Application
# ============================================================
def main():
    """Main Streamlit application."""

    # ---- Header ----
    st.markdown('<h1 class="main-title">⛑️ Helmet Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Powered by YOLOv8 • Real-Time AI Safety Detection</p>', unsafe_allow_html=True)
    st.markdown("---")

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        # Confidence threshold
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.1, max_value=1.0,
            value=CONFIDENCE_THRESHOLD, step=0.05,
            help="Minimum confidence score to display a detection"
        )

        st.markdown("---")

        # Input mode selection
        st.markdown("## 📥 Input Source")
        input_mode = st.radio(
            "Choose input mode:",
            ["📷 Upload Image", "🎥 Upload Video", "📹 Webcam (Live)"],
            index=0
        )

        st.markdown("---")

        # Info box
        st.markdown("""
        <div class="sidebar-info">
            <h4>📋 About</h4>
            <p style="font-size: 0.85rem; color: #ccc;">
                AI-powered system to detect whether individuals
                are wearing safety helmets. Built with YOLOv8,
                OpenCV, and PyTorch.
            </p>
            <p style="font-size: 0.8rem; color: #888;">
                <b>Classes:</b><br>
                🟢 Helmet<br>
                🔴 No Helmet
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Controls info
        st.markdown("""
        <div class="sidebar-info">
            <h4>🎮 Controls</h4>
            <p style="font-size: 0.85rem; color: #ccc;">
                <b>Image Mode:</b> Upload JPG/PNG<br>
                <b>Video Mode:</b> Upload MP4/AVI<br>
                <b>Webcam:</b> Toggle live detection
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ---- Load Model ----
    with st.spinner("Loading YOLOv8 model..."):
        detector = load_detector(confidence)
        detector.confidence = confidence  # Update if slider changes

    # ---- Image Mode ----
    if input_mode == "📷 Upload Image":
        st.markdown("### 📷 Image Detection")
        st.info("Upload an image to detect helmets. Supported formats: JPG, JPEG, PNG, BMP, WEBP")

        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="image_uploader"
        )

        if uploaded_file is not None:
            with st.spinner("🔍 Running detection..."):
                original, annotated, detections = process_uploaded_image(uploaded_file, detector)

            # Display results side by side
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Original Image")
                st.image(original, channels="RGB", use_container_width=True)
            with col2:
                st.markdown("#### Detection Results")
                st.image(annotated, channels="RGB", use_container_width=True)

            # Stats section
            st.markdown("---")
            st.markdown("### 📊 Detection Statistics")

            helmet_count = sum(1 for d in detections if d['class_id'] == 0)
            no_helmet_count = sum(1 for d in detections if d['class_id'] == 1)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Detections", len(detections))
            with col2:
                st.metric("✅ Helmet", helmet_count)
            with col3:
                st.metric("❌ No Helmet", no_helmet_count)

            # Detection details table
            if detections:
                st.markdown("### 📋 Detection Details")
                for i, det in enumerate(detections, 1):
                    badge_class = "helmet-badge" if det['class_id'] == 0 else "no-helmet-badge"
                    badge_text = det['class_name']
                    conf = det['confidence'] * 100
                    st.markdown(
                        f"**#{i}** "
                        f'<span class="{badge_class}">{badge_text}</span> '
                        f"— Confidence: **{conf:.1f}%**",
                        unsafe_allow_html=True
                    )

    # ---- Video Mode ----
    elif input_mode == "🎥 Upload Video":
        st.markdown("### 🎥 Video Detection")
        st.info("Upload a video file to process frame-by-frame. Supported formats: MP4, AVI, MOV")

        uploaded_video = st.file_uploader(
            "Choose a video...",
            type=["mp4", "avi", "mov"],
            key="video_uploader"
        )

        if uploaded_video is not None:
            st.markdown("#### Processing Video...")
            output_path = process_video_file(uploaded_video, detector)

            if output_path and os.path.exists(output_path):
                st.success("✅ Video processing complete!")
                st.video(output_path)

                # Download button
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=f.read(),
                        file_name="helmet_detection_output.mp4",
                        mime="video/mp4"
                    )

    # ---- Webcam Mode ----
    elif input_mode == "📹 Webcam (Live)":
        st.markdown("### 📹 Live Webcam Detection")
        st.warning("⚠️ For the best webcam experience, use the CLI mode instead:\n\n`python -m src.detect --webcam`")

        st.markdown("""
        The Streamlit webcam captures individual snapshots for processing.
        For smooth real-time video, use the **CLI mode** which uses OpenCV's
        native video capture for the best performance.

        **CLI Command:**
        ```bash
        python -m src.detect --webcam
        ```
        """)

        # Streamlit camera input for snapshot-based detection
        camera_input = st.camera_input("Take a photo for detection")

        if camera_input is not None:
            with st.spinner("🔍 Running detection..."):
                original, annotated, detections = process_uploaded_image(camera_input, detector)

            st.image(annotated, channels="RGB", use_container_width=True)

            # Stats
            helmet_count = sum(1 for d in detections if d['class_id'] == 0)
            no_helmet_count = sum(1 for d in detections if d['class_id'] == 1)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total", len(detections))
            with col2:
                st.metric("✅ Helmet", helmet_count)
            with col3:
                st.metric("❌ No Helmet", no_helmet_count)


if __name__ == "__main__":
    main()
