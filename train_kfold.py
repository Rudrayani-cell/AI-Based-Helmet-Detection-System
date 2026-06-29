"""
K-Fold Cross Validation Training Script
========================================
Train YOLOv8 with K-Fold Cross Validation for more reliable accuracy
estimates and better generalization on the helmet detection dataset.

This script:
  1. Combines all images/labels from train + val into a single pool
  2. Splits them into K folds
  3. Trains a separate model for each fold
  4. Reports per-fold and averaged metrics (mAP, precision, recall)
  5. Picks and saves the best fold's model as the final model

Usage:
    # 5-fold cross validation (default)
    python train_kfold.py

    # Custom fold count and epochs
    python train_kfold.py --folds 5 --epochs 50

    # Use a larger base model for better accuracy
    python train_kfold.py --model yolov8s.pt --epochs 80
"""

import os
import sys
import shutil
import random
import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO
from src.config import (
    BASE_MODEL, MODELS_DIR, DEVICE,
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_IMG_SIZE,
    TRAIN_PATIENCE, TRAIN_WORKERS
)
from src.utils import setup_logger

logger = setup_logger("kfold_train")

# Directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
KFOLD_DIR = os.path.join(PROJECT_ROOT, "kfold_splits")


def collect_all_samples(data_dir):
    """
    Collect all image-label pairs from train + val directories.
    Returns a list of (image_path, label_path) tuples.

    Args:
        data_dir: Root data directory containing images/ and labels/.

    Returns:
        list of tuples: [(image_path, label_path), ...]
    """
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    samples = []

    # Collect from both train and val splits
    for split in ['train', 'val']:
        img_dir = os.path.join(data_dir, "images", split)
        lbl_dir = os.path.join(data_dir, "labels", split)

        if not os.path.exists(img_dir):
            continue

        for img_file in sorted(os.listdir(img_dir)):
            ext = os.path.splitext(img_file)[1].lower()
            if ext not in img_extensions:
                continue

            img_path = os.path.join(img_dir, img_file)
            lbl_file = os.path.splitext(img_file)[0] + '.txt'
            lbl_path = os.path.join(lbl_dir, lbl_file)

            # Only include if both image and label exist
            if os.path.exists(lbl_path):
                samples.append((img_path, lbl_path))

    return samples


def create_kfold_splits(samples, k=5, seed=42):
    """
    Split samples into K folds using stratified-like random splitting.

    Args:
        samples: List of (image_path, label_path) tuples.
        k: Number of folds.
        seed: Random seed for reproducibility.

    Returns:
        list of lists: K lists, each containing (img_path, lbl_path) tuples.
    """
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)

    # Distribute evenly into K folds
    folds = [[] for _ in range(k)]
    for i, sample in enumerate(shuffled):
        folds[i % k].append(sample)

    return folds


def setup_fold_data(folds, fold_idx, kfold_dir):
    """
    Set up the dataset directory structure for a specific fold.
    Uses symlinks (or copies) to avoid duplicating data.

    Args:
        folds: List of K sample lists.
        fold_idx: Current fold index (used as validation set).
        kfold_dir: Base directory for K-fold temp data.

    Returns:
        str: Path to the fold-specific data.yaml file.
    """
    fold_dir = os.path.join(kfold_dir, f"fold_{fold_idx + 1}")

    # Clean up any previous fold data (retry for Windows file locks)
    if os.path.exists(fold_dir):
        import time
        for attempt in range(3):
            try:
                shutil.rmtree(fold_dir)
                break
            except PermissionError:
                if attempt < 2:
                    time.sleep(2)
                else:
                    # Force delete on Windows
                    os.system(f'rmdir /S /Q "{fold_dir}"')
                    time.sleep(1)

    # Create directories
    train_img_dir = os.path.join(fold_dir, "images", "train")
    train_lbl_dir = os.path.join(fold_dir, "labels", "train")
    val_img_dir = os.path.join(fold_dir, "images", "val")
    val_lbl_dir = os.path.join(fold_dir, "labels", "val")

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    # Validation fold = fold_idx, Training = all other folds
    for i, fold in enumerate(folds):
        if i == fold_idx:
            # This is the validation fold
            for img_path, lbl_path in fold:
                shutil.copy2(img_path, os.path.join(val_img_dir, os.path.basename(img_path)))
                shutil.copy2(lbl_path, os.path.join(val_lbl_dir, os.path.basename(lbl_path)))
        else:
            # This is a training fold
            for img_path, lbl_path in fold:
                shutil.copy2(img_path, os.path.join(train_img_dir, os.path.basename(img_path)))
                shutil.copy2(lbl_path, os.path.join(train_lbl_dir, os.path.basename(lbl_path)))

    # Create fold-specific data.yaml
    fold_yaml_path = os.path.join(fold_dir, "data.yaml")
    fold_config = {
        'path': os.path.abspath(fold_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 2,
        'names': {0: 'Helmet', 1: 'No Helmet'}
    }
    with open(fold_yaml_path, 'w') as f:
        yaml.dump(fold_config, f, default_flow_style=False, sort_keys=False)

    return fold_yaml_path


def train_fold(fold_idx, k, fold_yaml, args):
    """
    Train a single fold of the K-Fold cross validation.

    Args:
        fold_idx: Current fold index (0-based).
        k: Total number of folds.
        fold_yaml: Path to this fold's data.yaml.
        args: Training arguments.

    Returns:
        dict: Metrics for this fold (mAP50, mAP50-95, precision, recall).
    """
    fold_num = fold_idx + 1
    logger.info(f"")
    logger.info(f"{'=' * 60}")
    logger.info(f"  FOLD {fold_num}/{k}")
    logger.info(f"{'=' * 60}")

    # Load fresh base model for each fold
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=fold_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.img_size,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
        name=f"fold_{fold_num}",
        project=os.path.join(PROJECT_ROOT, "runs", "kfold"),
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        seed=42 + fold_idx,
        save=True,
        plots=True,
    )

    # Extract metrics from training results
    metrics = {}
    try:
        # Validate using the best model from this fold
        best_weights = os.path.join(
            PROJECT_ROOT, "runs", "kfold", f"fold_{fold_num}", "weights", "best.pt"
        )
        if os.path.exists(best_weights):
            val_model = YOLO(best_weights)
            val_results = val_model.val(data=fold_yaml, imgsz=args.img_size, device=args.device)

            metrics = {
                'fold': fold_num,
                'mAP50': float(val_results.box.map50),
                'mAP50_95': float(val_results.box.map),
                'precision': float(val_results.box.mp),
                'recall': float(val_results.box.mr),
                'best_weights': best_weights,
            }
        else:
            logger.warning(f"Best weights not found for fold {fold_num}")
            metrics = {
                'fold': fold_num,
                'mAP50': 0.0, 'mAP50_95': 0.0,
                'precision': 0.0, 'recall': 0.0,
                'best_weights': None,
            }
    except Exception as e:
        logger.error(f"Error extracting metrics for fold {fold_num}: {e}")
        metrics = {
            'fold': fold_num,
            'mAP50': 0.0, 'mAP50_95': 0.0,
            'precision': 0.0, 'recall': 0.0,
            'best_weights': None,
        }

    logger.info(f"  Fold {fold_num} Results:")
    logger.info(f"    mAP@50:    {metrics['mAP50']:.4f}")
    logger.info(f"    mAP@50-95: {metrics['mAP50_95']:.4f}")
    logger.info(f"    Precision: {metrics['precision']:.4f}")
    logger.info(f"    Recall:    {metrics['recall']:.4f}")

    return metrics


def summarize_results(all_metrics, output_dir):
    """
    Compute and display averaged K-Fold results with standard deviations.
    Save the best model as the final model.

    Args:
        all_metrics: List of per-fold metric dicts.
        output_dir: Directory to save the summary.
    """
    print()
    print("=" * 65)
    print("  K-FOLD CROSS VALIDATION RESULTS")
    print("=" * 65)
    print()

    # Per-fold table
    print(f"  {'Fold':>4}  |  {'mAP@50':>8}  |  {'mAP@50-95':>10}  |  {'Precision':>9}  |  {'Recall':>7}")
    print(f"  {'-' * 4}  |  {'-' * 8}  |  {'-' * 10}  |  {'-' * 9}  |  {'-' * 7}")

    for m in all_metrics:
        print(f"  {m['fold']:>4}  |  {m['mAP50']:>8.4f}  |  {m['mAP50_95']:>10.4f}  |  {m['precision']:>9.4f}  |  {m['recall']:>7.4f}")

    # Averaged results
    map50_values = [m['mAP50'] for m in all_metrics]
    map50_95_values = [m['mAP50_95'] for m in all_metrics]
    prec_values = [m['precision'] for m in all_metrics]
    rec_values = [m['recall'] for m in all_metrics]

    print(f"  {'-' * 55}")
    print(f"  {'AVG':>4}  |  {np.mean(map50_values):>8.4f}  |  {np.mean(map50_95_values):>10.4f}  |  {np.mean(prec_values):>9.4f}  |  {np.mean(rec_values):>7.4f}")
    print(f"  {'STD':>4}  |  {np.std(map50_values):>8.4f}  |  {np.std(map50_95_values):>10.4f}  |  {np.std(prec_values):>9.4f}  |  {np.std(rec_values):>7.4f}")
    print()

    # Summary
    print(f"  Final mAP@50:    {np.mean(map50_values)*100:.1f}% (+/- {np.std(map50_values)*100:.1f}%)")
    print(f"  Final mAP@50-95: {np.mean(map50_95_values)*100:.1f}% (+/- {np.std(map50_95_values)*100:.1f}%)")
    print(f"  Final Precision: {np.mean(prec_values)*100:.1f}% (+/- {np.std(prec_values)*100:.1f}%)")
    print(f"  Final Recall:    {np.mean(rec_values)*100:.1f}% (+/- {np.std(rec_values)*100:.1f}%)")
    print()

    # Find and save best fold model
    best_fold = max(all_metrics, key=lambda m: m['mAP50'])
    print(f"  Best Fold: {best_fold['fold']} (mAP@50 = {best_fold['mAP50']:.4f})")

    if best_fold.get('best_weights') and os.path.exists(best_fold['best_weights']):
        dst = os.path.join(MODELS_DIR, "best.pt")
        os.makedirs(MODELS_DIR, exist_ok=True)
        shutil.copy2(best_fold['best_weights'], dst)
        print(f"  Best model saved to: {dst}")

    # Save results to JSON
    results_file = os.path.join(output_dir, "kfold_results.json")
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'num_folds': len(all_metrics),
        'per_fold': all_metrics,
        'averaged': {
            'mAP50': {'mean': float(np.mean(map50_values)), 'std': float(np.std(map50_values))},
            'mAP50_95': {'mean': float(np.mean(map50_95_values)), 'std': float(np.std(map50_95_values))},
            'precision': {'mean': float(np.mean(prec_values)), 'std': float(np.std(prec_values))},
            'recall': {'mean': float(np.mean(rec_values)), 'std': float(np.std(rec_values))},
        },
        'best_fold': best_fold['fold'],
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"  Results saved to: {results_file}")
    print()

    # Summary text file (easy to include in reports/presentations)
    summary_file = os.path.join(output_dir, "kfold_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("K-Fold Cross Validation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of Folds: {len(all_metrics)}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"mAP@50:    {np.mean(map50_values)*100:.1f}% +/- {np.std(map50_values)*100:.1f}%\n")
        f.write(f"mAP@50-95: {np.mean(map50_95_values)*100:.1f}% +/- {np.std(map50_95_values)*100:.1f}%\n")
        f.write(f"Precision: {np.mean(prec_values)*100:.1f}% +/- {np.std(prec_values)*100:.1f}%\n")
        f.write(f"Recall:    {np.mean(rec_values)*100:.1f}% +/- {np.std(rec_values)*100:.1f}%\n\n")
        f.write(f"Best Fold: {best_fold['fold']} (mAP@50 = {best_fold['mAP50']*100:.1f}%)\n")
    print(f"  Summary saved to: {summary_file}")
    print()

    return best_fold


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="K-Fold Cross Validation Training for Helmet Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_kfold.py                          # 5-fold, 50 epochs
  python train_kfold.py --folds 5 --epochs 50    # Same as above
  python train_kfold.py --folds 3 --epochs 30    # Quick 3-fold
  python train_kfold.py --model yolov8s.pt       # Better base model
        """
    )

    parser.add_argument("--folds", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--model", type=str, default=BASE_MODEL, help=f"Base YOLO model (default: {BASE_MODEL})")
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS, help=f"Epochs per fold (default: {TRAIN_EPOCHS})")
    parser.add_argument("--batch", type=int, default=TRAIN_BATCH_SIZE, help=f"Batch size (default: {TRAIN_BATCH_SIZE})")
    parser.add_argument("--img-size", type=int, default=TRAIN_IMG_SIZE, help=f"Image size (default: {TRAIN_IMG_SIZE})")
    parser.add_argument("--patience", type=int, default=TRAIN_PATIENCE, help=f"Early stopping (default: {TRAIN_PATIENCE})")
    parser.add_argument("--workers", type=int, default=TRAIN_WORKERS, help=f"Workers (default: {TRAIN_WORKERS})")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device (default: auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--keep-splits", action="store_true", help="Keep fold split directories after training")

    return parser.parse_args()


def main():
    args = parse_arguments()

    print()
    print("=" * 65)
    print("  HELMET DETECTION - K-FOLD CROSS VALIDATION TRAINING")
    print("=" * 65)
    print()

    # Step 1: Collect all samples
    logger.info("Step 1: Collecting all image-label pairs...")

    samples = collect_all_samples(DATA_DIR)
    logger.info(f"  Found {len(samples)} image-label pairs")

    if len(samples) < args.folds * 10:
        logger.error(f"Not enough samples ({len(samples)}) for {args.folds}-fold CV.")
        logger.error("Need at least 10 samples per fold.")
        sys.exit(1)

    # Step 2: Create K-fold splits
    logger.info(f"Step 2: Creating {args.folds}-fold splits (seed={args.seed})...")

    folds = create_kfold_splits(samples, k=args.folds, seed=args.seed)
    for i, fold in enumerate(folds):
        logger.info(f"  Fold {i+1}: {len(fold)} samples")

    # Step 3: Train each fold
    logger.info(f"Step 3: Training {args.folds} folds, {args.epochs} epochs each...")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Total epochs: {args.folds * args.epochs}")
    print()

    all_metrics = []
    for fold_idx in range(args.folds):
        # Set up fold data
        fold_yaml = setup_fold_data(folds, fold_idx, KFOLD_DIR)

        # Count train/val images for this fold
        fold_dir = os.path.join(KFOLD_DIR, f"fold_{fold_idx + 1}")
        n_train = len(os.listdir(os.path.join(fold_dir, "images", "train")))
        n_val = len(os.listdir(os.path.join(fold_dir, "images", "val")))
        logger.info(f"  Fold {fold_idx + 1}: {n_train} train, {n_val} val images")

        # Train
        metrics = train_fold(fold_idx, args.folds, fold_yaml, args)
        all_metrics.append(metrics)

    # Step 4: Summarize results
    logger.info("Step 4: Summarizing results...")
    output_dir = os.path.join(PROJECT_ROOT, "runs", "kfold")
    best_fold = summarize_results(all_metrics, output_dir)

    # Step 5: Cleanup fold split directories (optional)
    if not args.keep_splits and os.path.exists(KFOLD_DIR):
        logger.info("Cleaning up fold split directories...")
        shutil.rmtree(KFOLD_DIR)
        logger.info("Done.")

    print()
    print("=" * 65)
    print("  K-FOLD CROSS VALIDATION COMPLETE!")
    print("=" * 65)
    print()
    print("  Next steps:")
    print("    1. Test detection:  python -m src.detect --image <path>")
    print("    2. Webcam demo:    python -m src.detect --webcam")
    print("    3. Streamlit app:  streamlit run src/app.py")
    print()


if __name__ == "__main__":
    main()
