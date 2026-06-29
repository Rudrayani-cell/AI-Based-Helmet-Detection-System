"""
Helmet Detector Module
======================
Core detection engine that wraps the YOLOv8 model from the Ultralytics library.
Provides methods for detecting helmets in images, video files, and webcam streams.

Usage:
    from src.detector import HelmetDetector
    detector = HelmetDetector()
    detections = detector.detect_frame(frame)
"""

import cv2
import numpy as np
import os
import time

from ultralytics import YOLO

from src.config import (
    get_model_path, CONFIDENCE_THRESHOLD, IOU_THRESHOLD,
    INPUT_SIZE, CLASS_NAMES, DEVICE, WINDOW_NAME
)
from src.utils import (
    draw_detections, draw_stats_overlay, draw_fps,
    resize_frame, save_output, calculate_fps, setup_logger
)


# Module logger
logger = setup_logger(__name__)


class HelmetDetector:
    """
    Helmet Detection Engine using YOLOv8.

    This class encapsulates the YOLO model and provides a clean API
    for running helmet detection on various input sources.

    Attributes:
        model (YOLO): The loaded YOLO model instance.
        confidence (float): Minimum confidence threshold for detections.
        iou_threshold (float): IoU threshold for NMS.
        device (str): Compute device ('cuda' or 'cpu').
    """

    def __init__(self, model_path=None, confidence=CONFIDENCE_THRESHOLD,
                 iou_threshold=IOU_THRESHOLD, device=DEVICE, half=False):
        """
        Initialize the Helmet Detector.

        Args:
            model_path (str, optional): Path to YOLO model weights.
                Defaults to the best available model.
            confidence (float): Minimum confidence threshold (0-1).
            iou_threshold (float): IoU threshold for NMS (0-1).
            device (str): Device to run inference on ('cuda' or 'cpu').
            half (bool): Use half-precision (FP16) for faster GPU inference.
        """
        self.model_path = model_path or get_model_path()
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device
        self.half = half
        
        # Failsafe: CPU inference deadlocks on Windows if half-precision (FP16) is forced
        if self.device == "cpu" and self.half:
            logger.warning("Emergency Failsafe: CPU detected. Disabling FP16 (--half) to prevent freezing.")
            self.half = False
        self.model = None

        self._load_model()

    def _load_model(self):
        """
        Load the YOLO model from the specified path.

        Raises:
            FileNotFoundError: If model path doesn't exist and isn't a
                standard ultralytics model name.
        """
        logger.info(f"Loading YOLO model from: {self.model_path}")
        logger.info(f"Device: {self.device} (FP16: {self.half})")

        try:
            self.model = YOLO(self.model_path)
            
            # Note: ultralytics applies half() automatically during inference
            # if the device supports it, but setting it explicitly is safe.
            if self.half and self.device == 'cuda':
                logger.info("Enabling half-precision (FP16)")
                
            logger.info("Model loaded successfully!")

            # Log model info
            if hasattr(self.model, 'names'):
                logger.info(f"Model classes: {self.model.names}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def detect_frame(self, frame, track=False, tracker="botsort.yaml"):
        """
        Run helmet detection on a single frame/image.

        Args:
            frame (np.ndarray): Input image in BGR format (OpenCV).
            track (bool): Whether to enable real-time object tracking.
            tracker (str): Tracker configuration file (botsort.yaml or bytetrack.yaml).

        Returns:
            list: List of detection dictionaries, each containing:
                - 'bbox': [x1, y1, x2, y2] bounding box coordinates
                - 'confidence': float detection confidence
                - 'class_id': int class index (0=Helmet, 1=No Helmet)
                - 'class_name': str class label
                - 'track_id': int tracker ID (if track=True)
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame received, skipping detection.")
            return []

        # Run inference or tracking
        if track:
            results = self.model.track(
                frame,
                persist=True,
                tracker=tracker,
                conf=self.confidence,
                iou=self.iou_threshold,
                imgsz=INPUT_SIZE,
                device=self.device,
                half=self.half,
                verbose=False
            )
        else:
            results = self.model(
                frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                imgsz=INPUT_SIZE,
                device=self.device,
                half=self.half,
                verbose=False
            )

        # Parse results into a clean format
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Extract confidence, class, and optional ID
                conf = float(box.conf[0].cpu().numpy())
                
                # MANUALLY ENFORCE CONFIDENCE (Trackers sometimes ignore it to persist tracks)
                if conf < self.confidence:
                    continue
                    
                cls_id = int(box.cls[0].cpu().numpy())
                
                track_id = None
                if track and box.id is not None:
                    track_id = int(box.id[0].cpu().numpy())

                # Map class name
                cls_name = CLASS_NAMES.get(cls_id, self.model.names.get(cls_id, "Unknown"))

                det_info = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': cls_name
                }
                
                if track_id is not None:
                    det_info['track_id'] = track_id
                    
                detections.append(det_info)

        return detections

    def detect_image(self, image_path, show=True, save=True):
        """
        Run detection on a single image file.

        Args:
            image_path (str): Path to the input image.
            show (bool): Whether to display the annotated image.
            save (bool): Whether to save the annotated output.

        Returns:
            tuple: (annotated_frame, detections_list)
        """
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info(f"Processing image: {image_path}")

        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"Failed to read image: {image_path}")
            raise ValueError(f"Could not read image: {image_path}")

        # Run detection
        detections = self.detect_frame(frame)

        # Draw results
        annotated = draw_detections(frame.copy(), detections)
        annotated = draw_stats_overlay(annotated, detections)

        # Log results
        helmet_count = sum(1 for d in detections if d['class_id'] == 0)
        no_helmet_count = sum(1 for d in detections if d['class_id'] == 1)
        logger.info(f"Detected: {helmet_count} Helmet(s), {no_helmet_count} No Helmet(s)")

        # Save output
        if save:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = save_output(annotated, source_name=base_name)
            logger.info(f"Output saved to: {output_path}")

        # Display
        if show:
            display_frame = resize_frame(annotated)
            cv2.imshow(WINDOW_NAME, display_frame)
            logger.info("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return annotated, detections

    def detect_video(self, video_path, show=True, save=True, track=False, tracker="botsort.yaml"):
        """
        Run detection on a video file frame-by-frame.

        Args:
            video_path (str): Path to the input video file.
            show (bool): Whether to display the annotated video.
            save (bool): Whether to save the annotated output video.
            track (bool): Whether to enable real-time object tracking.
            tracker (str): Tracker configuration file.

        Returns:
            str: Path to the saved output video (if save=True).
        """
        if not os.path.exists(video_path):
            logger.error(f"Video not found: {video_path}")
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(f"Processing video: {video_path}")

        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")

        # Setup video writer for saving
        output_path = None
        writer = None
        if save:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(
                os.path.dirname(save_output.__defaults__[1]) if save_output.__defaults__ else "output",
                f"{base_name}_detected.mp4"
            )
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else "output", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        prev_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Run detection
                detections = self.detect_frame(frame, track=track, tracker=tracker)

                # Draw results
                annotated = draw_detections(frame, detections)
                annotated = draw_stats_overlay(annotated, detections)

                # Calculate and draw FPS
                current_fps, prev_time = calculate_fps(prev_time)
                annotated = draw_fps(annotated, current_fps)

                # Save frame
                if writer:
                    writer.write(annotated)

                # Display
                if show:
                    display_frame = resize_frame(annotated)
                    cv2.imshow(WINDOW_NAME, display_frame)

                    # Press 'q' to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User pressed 'q' — stopping video.")
                        break

                # Progress logging every 100 frames
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count}/{total_frames} frames")

        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()

        logger.info(f"Video processing complete. Processed {frame_count} frames.")
        if output_path:
            logger.info(f"Output saved to: {output_path}")

        return output_path

    def detect_webcam(self, camera_id=0, save=False, track=False, tracker="botsort.yaml"):
        """
        Run real-time helmet detection on a webcam feed.

        Args:
            camera_id (int): Camera device index (default 0).
            save (bool): Whether to save a snapshot on 's' key press.
            track (bool): Whether to enable real-time object tracking.
            tracker (str): Tracker configuration file.
        """
        logger.info(f"Starting webcam detection (camera {camera_id})...")
        logger.info("Controls: 'q' = quit, 's' = save snapshot")

        # Use DirectShow backend on Windows to prevent standard OpenCV freezing/0-FPS bugs
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            logger.warning(f"Failed to open with DSHOW, falling back to default API.")
            cap = cv2.VideoCapture(camera_id)
            
        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_id}")
            raise RuntimeError(f"Could not open camera {camera_id}")

        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        prev_time = time.time()

        try:
            while cap.isOpened():
                print("[DEBUG] Grabbing frame from webcam...")
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to grab frame from webcam.")
                    break

                print("[DEBUG] Frame grabbed. Running detection...")
                # Run detection
                detections = self.detect_frame(frame, track=track, tracker=tracker)
                print(f"[DEBUG] Detection finished. Found {len(detections)} objects.")

                # Draw results
                annotated = draw_detections(frame, detections)
                annotated = draw_stats_overlay(annotated, detections)

                # Calculate and draw FPS
                current_fps, prev_time = calculate_fps(prev_time)
                annotated = draw_fps(annotated, current_fps)

                # Display
                display_frame = resize_frame(annotated)
                cv2.imshow(WINDOW_NAME, display_frame)

                # Key handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("User pressed 'q' — stopping webcam.")
                    break
                elif key == ord('s'):
                    snapshot_path = save_output(annotated, source_name="webcam_snapshot")
                    logger.info(f"Snapshot saved to: {snapshot_path}")

        finally:
            cap.release()
            cv2.destroyAllWindows()

        logger.info("Webcam detection stopped.")
