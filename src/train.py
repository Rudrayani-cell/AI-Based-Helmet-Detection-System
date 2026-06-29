"""
Training Script
===============
Fine-tune a YOLOv8 model on a custom helmet detection dataset.
Supports configurable epochs, batch size, image size, and device selection.

Usage:
    # Train with default settings
    python -m src.train

    # Train with custom parameters
    python -m src.train --epochs 100 --batch 32 --img-size 640

    # Resume training from a checkpoint
    python -m src.train --resume
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO

from src.config import (
    BASE_MODEL, DATA_YAML, MODELS_DIR, PROJECT_ROOT,
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_IMG_SIZE,
    TRAIN_PATIENCE, TRAIN_WORKERS, DEVICE
)
from src.utils import setup_logger


logger = setup_logger("train")


def parse_arguments():
    """
    Parse command-line arguments for the training script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model for Helmet Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.train
  python -m src.train --epochs 100 --batch 32
  python -m src.train --model yolov8s.pt --epochs 80
  python -m src.train --resume
        """
    )

    parser.add_argument(
        "--model", type=str, default=BASE_MODEL,
        help=f"Base YOLO model to fine-tune (default: {BASE_MODEL})"
    )
    parser.add_argument(
        "--data", type=str, default=DATA_YAML,
        help="Path to dataset configuration YAML file"
    )
    parser.add_argument(
        "--epochs", type=int, default=TRAIN_EPOCHS,
        help=f"Number of training epochs (default: {TRAIN_EPOCHS})"
    )
    parser.add_argument(
        "--batch", type=int, default=TRAIN_BATCH_SIZE,
        help=f"Batch size (default: {TRAIN_BATCH_SIZE})"
    )
    parser.add_argument(
        "--img-size", type=int, default=TRAIN_IMG_SIZE,
        help=f"Input image size (default: {TRAIN_IMG_SIZE})"
    )
    parser.add_argument(
        "--patience", type=int, default=TRAIN_PATIENCE,
        help=f"Early stopping patience (default: {TRAIN_PATIENCE})"
    )
    parser.add_argument(
        "--workers", type=int, default=TRAIN_WORKERS,
        help=f"Number of data loader workers (default: {TRAIN_WORKERS})"
    )
    parser.add_argument(
        "--device", type=str, default=DEVICE,
        help="Device to train on: 'cuda' or 'cpu'"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from the last checkpoint"
    )
    parser.add_argument(
        "--name", type=str, default="helmet_detection",
        help="Experiment name for saving results"
    )

    return parser.parse_args()


def train(args):
    """
    Execute model training with the given arguments.

    Args:
        args (argparse.Namespace): Training arguments.
    """
    print("=" * 60)
    print("  Helmet Detection - YOLOv8 Training")
    print("=" * 60)
    print()

    # Validate dataset config exists
    if not os.path.exists(args.data):
        logger.error(f"Dataset configuration not found: {args.data}")
        logger.info("Please run the dataset manager first:")
        logger.info("  python -m src.dataset_manager")
        sys.exit(1)

    # Log training configuration
    logger.info("Training Configuration:")
    logger.info(f"  Base Model:    {args.model}")
    logger.info(f"  Dataset:       {args.data}")
    logger.info(f"  Epochs:        {args.epochs}")
    logger.info(f"  Batch Size:    {args.batch}")
    logger.info(f"  Image Size:    {args.img_size}")
    logger.info(f"  Patience:      {args.patience}")
    logger.info(f"  Device:        {args.device}")
    logger.info(f"  Workers:       {args.workers}")
    logger.info(f"  Experiment:    {args.name}")
    print()

    # Load the base model
    logger.info(f"Loading base model: {args.model}")
    model = YOLO(args.model)

    # Start training
    logger.info("Starting training...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.img_size,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
        name=args.name,
        project=os.path.join(PROJECT_ROOT, "runs"),
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        seed=42,
        deterministic=True,
        save=True,
        save_period=-1,         # Save only best and last
        plots=True,             # Generate training plots
        resume=args.resume,
    )

    # Copy best weights to models directory
    best_weights_src = os.path.join(
        PROJECT_ROOT, "runs", args.name, "weights", "best.pt"
    )
    best_weights_dst = os.path.join(MODELS_DIR, "best.pt")

    if os.path.exists(best_weights_src):
        import shutil
        os.makedirs(MODELS_DIR, exist_ok=True)
        shutil.copy2(best_weights_src, best_weights_dst)
        logger.info(f"Best weights copied to: {best_weights_dst}")
    else:
        logger.warning("Best weights file not found in training output.")

    # Log final results
    print()
    print("=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    logger.info("Training finished successfully.")
    logger.info(f"Results saved to: runs/{args.name}/")
    logger.info(f"Best model saved to: {best_weights_dst}")
    print()
    logger.info("Next steps:")
    logger.info("  1. Test the model: python -m src.detect --image <path>")
    logger.info("  2. Run webcam demo: python -m src.detect --webcam")
    logger.info("  3. Launch Streamlit: streamlit run src/app.py")


def main():
    args = parse_arguments()
    train(args)


if __name__ == "__main__":
    main()
