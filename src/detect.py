"""
CLI Detection Script
====================
Main command-line entry point for running helmet detection.
Supports three modes: image, video file, and real-time webcam.

Usage:
    # Detect on an image
    python -m src.detect --image path/to/image.jpg

    # Detect on a video
    python -m src.detect --video path/to/video.mp4

    # Detect using webcam
    python -m src.detect --webcam

    # With custom confidence threshold
    python -m src.detect --webcam --confidence 0.6

    # Use a specific model
    python -m src.detect --image photo.jpg --model models/best.pt
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import HelmetDetector
from src.utils import setup_logger
from src.config import CONFIDENCE_THRESHOLD, TRAINED_MODEL_PATH


logger = setup_logger("detect")


def parse_arguments():
    """
    Parse command-line arguments for the detection script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="AI-Based Helmet Detection System using YOLOv8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.detect --image data/sample/test.jpg
  python -m src.detect --video data/sample/test.mp4
  python -m src.detect --webcam
  python -m src.detect --webcam --confidence 0.6
  python -m src.detect --image photo.jpg --model models/best.pt --no-save
        """
    )

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", type=str, metavar="PATH",
        help="Path to an input image file for detection"
    )
    input_group.add_argument(
        "--video", type=str, metavar="PATH",
        help="Path to an input video file for detection"
    )
    input_group.add_argument(
        "--webcam", action="store_true",
        help="Use webcam for real-time detection"
    )

    # Model settings
    parser.add_argument(
        "--model", type=str, default=None, metavar="PATH",
        help="Path to YOLO model weights (default: auto-detect best available)"
    )
    parser.add_argument(
        "--confidence", type=float, default=CONFIDENCE_THRESHOLD,
        help=f"Minimum confidence threshold (default: {CONFIDENCE_THRESHOLD})"
    )
    parser.add_argument(
        "--half", action="store_true",
        help="Use FP16 (Half Precision) for faster GPU inference"
    )

    # Tracking settings
    parser.add_argument(
        "--track", action="store_true",
        help="Enable object tracking (BoT-SORT or ByteTrack)"
    )
    parser.add_argument(
        "--tracker", type=str, default="botsort.yaml", choices=["botsort.yaml", "bytetrack.yaml"],
        help="Tracker method to use (default: botsort.yaml)"
    )

    # Output settings
    parser.add_argument(
        "--no-show", action="store_true",
        help="Don't display the detection window (headless mode)"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save the annotated output"
    )

    # Camera settings
    parser.add_argument(
        "--camera-id", type=int, default=0,
        help="Camera device ID for webcam mode (default: 0)"
    )

    return parser.parse_args()


def main():
    """
    Main entry point for the helmet detection CLI.
    """
    args = parse_arguments()

    # Banner
    print("=" * 60)
    print("  AI-Based Helmet Detection System using YOLOv8")
    print("=" * 60)
    print()

    try:
        # Initialize detector
        detector = HelmetDetector(
            model_path=args.model,
            confidence=args.confidence,
            half=args.half
        )

        # Run detection based on input mode
        if args.image:
            logger.info(f"Mode: Image Detection (Tracking: {args.track})")
            detector.detect_image(
                image_path=args.image,
                show=not args.no_show,
                save=not args.no_save,
                # Note: detect_image doesn't inherently need tracking, but we pass it anyway
                # (You would need to update HelmetDetector.detect_image to accept track=args.track if you wanted it)
            )

        elif args.video:
            logger.info(f"Mode: Video Detection (Tracking: {args.track}, Tracker: {args.tracker})")
            detector.detect_video(
                video_path=args.video,
                show=not args.no_show,
                save=not args.no_save,
                track=args.track,
                tracker=args.tracker
            )

        elif args.webcam:
            logger.info(f"Mode: Webcam Real-Time Detection (Tracking: {args.track})")
            detector.detect_webcam(
                camera_id=args.camera_id,
                save=not args.no_save,
                track=args.track,
                tracker=args.tracker
            )

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Detection interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

    print()
    print("Detection complete. Thank you!")


if __name__ == "__main__":
    main()
