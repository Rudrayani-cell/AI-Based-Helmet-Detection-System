"""
High-Performance Model Export Script
====================================
Exports trained YOLOv8 PyTorch models (.pt) to optimized formats for
high-throughput or low-latency inference in production (e.g., ONNX, TensorRT).

Usage:
    # Export to ONNX (good for CPU and cross-platform)
    python export_model.py --format onnx --model models/best.pt

    # Export to TensorRT for NVIDIA GPUs (requires Linux/TensorRT installed)
    python export_model.py --format engine --model models/best.pt --half
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
from src.config import get_model_path, INPUT_SIZE
from src.utils import setup_logger

logger = setup_logger("export")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 models to high-performance formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available formats:
  onnx      : ONNX format (Best for general use)
  engine    : TensorRT (Best for NVIDIA GPUs, requires TensorRT)
  openvino  : OpenVINO (Best for Intel CPUs)
  tflite    : TensorFlow Lite (Best for Mobile/Edge)
        """
    )
    
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to YOLO .pt model (Default: auto-select best.pt)"
    )
    parser.add_argument(
        "--format", type=str, required=True,
        choices=['onnx', 'engine', 'openvino', 'tflite', 'torchscript', 'coreml'],
        help="Target export format"
    )
    parser.add_argument(
        "--img-size", type=int, default=INPUT_SIZE,
        help=f"Image size for export (default: {INPUT_SIZE})"
    )
    parser.add_argument(
        "--half", action="store_true",
        help="Export in FP16 (Half Precision) for faster GPU inference"
    )
    parser.add_argument(
        "--dynamic", action="store_true",
        help="Enable dynamic dimensions (ONNX/TensorRT only)"
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    model_path = args.model or get_model_path()

    print("=" * 60)
    print("  HELMET DETECTION - MODEL EXPORT")
    print("=" * 60)
    print()

    if not os.path.exists(model_path):
        logger.error(f"Cannot find model: {model_path}")
        logger.error("Please train a model first or check the path.")
        sys.exit(1)

    logger.info(f"Loading base model: {model_path}")
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        sys.exit(1)

    logger.info(f"Exporting to format: {args.format.upper()}")
    logger.info(f"Target image size:   {args.img_size}")
    logger.info(f"Half precision:      {bool(args.half)}")
    logger.info(f"Dynamic shapes:      {bool(args.dynamic)}")
    print("-" * 60)

    try:
        exported_path = model.export(
            format=args.format,
            imgsz=args.img_size,
            half=args.half,
            dynamic=args.dynamic,
            simplify=True if args.format == 'onnx' else False,
            workspace=4  # standard workspace for TensorRT
        )
        print("-" * 60)
        logger.info(f"Export successful!")
        logger.info(f"New high-performance model saved to: {exported_path}")
        print()
        
        # Give usage instructions
        ext = args.format
        if args.format == 'engine': ext = 'engine'
        elif args.format == 'openvino': ext = '_openvino_model'
        elif args.format == 'onnx': ext = 'onnx'
        
        print("To use this model for fast inference, run:")
        print(f"  python -m src.detect --webcam --model {exported_path}")
        if args.half:
            print(f"  (Make sure to add --half if the export was FP16)")
            
    except Exception as e:
        logger.error(f"Export failed: {e}")
        logger.info("\nNote: Some formats like 'engine' (TensorRT) require specific NVIDIA libraries to be installed.")
        logger.info("If TensorRT fails, try exporting to 'onnx' instead, which is widely supported.")
        sys.exit(1)


if __name__ == "__main__":
    main()
