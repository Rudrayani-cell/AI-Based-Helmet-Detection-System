"""
Dataset Manager
===============
Download and organize helmet detection datasets for training.
Supports downloading from Roboflow (public datasets) or providing
instructions for manual Kaggle dataset setup.

Usage:
    # Download and set up the dataset
    python -m src.dataset_manager

    # Specify a custom output directory
    python -m src.dataset_manager --output data/
"""

import argparse
import os
import sys
import shutil
import zipfile
import urllib.request
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_DIR, DATA_YAML, PROJECT_ROOT
from src.utils import setup_logger


logger = setup_logger("dataset_manager")


def create_dataset_yaml(data_dir, yaml_path):
    """
    Create/update the YOLO dataset configuration YAML file.

    Args:
        data_dir (str): Root directory of the dataset.
        yaml_path (str): Path to save the YAML config file.
    """
    config = {
        'path': os.path.abspath(data_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 2,
        'names': {0: 'Helmet', 1: 'No Helmet'}
    }

    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Dataset YAML config saved to: {yaml_path}")


def create_directory_structure(data_dir):
    """
    Create the required YOLO directory structure for the dataset.

    Args:
        data_dir (str): Root directory for the dataset.
    """
    directories = [
        os.path.join(data_dir, "images", "train"),
        os.path.join(data_dir, "images", "val"),
        os.path.join(data_dir, "images", "test"),
        os.path.join(data_dir, "labels", "train"),
        os.path.join(data_dir, "labels", "val"),
        os.path.join(data_dir, "labels", "test"),
    ]

    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")


def download_sample_dataset(data_dir):
    """
    Provide instructions for downloading a helmet detection dataset.
    Also creates the necessary folder structure.

    Args:
        data_dir (str): Root directory for the dataset.
    """
    print()
    print("=" * 60)
    print("  Dataset Setup Instructions")
    print("=" * 60)
    print()

    # Create directory structure
    create_directory_structure(data_dir)

    print("The directory structure has been created. You need to")
    print("download a helmet detection dataset and place the files")
    print("in the correct directories.")
    print()
    print("=" * 60)
    print("  OPTION 1: Roboflow (Recommended - Easy)")
    print("=" * 60)
    print()
    print("1. Go to: https://universe.roboflow.com/search?q=helmet+detection")
    print("2. Choose a dataset (e.g., 'Safety Helmet Detection')")
    print("3. Click 'Download Dataset'")
    print("4. Select format: 'YOLOv8'")
    print("5. Download and extract the ZIP file")
    print("6. Copy images to:")
    print(f"   - Train: {os.path.join(data_dir, 'images', 'train')}")
    print(f"   - Val:   {os.path.join(data_dir, 'images', 'val')}")
    print(f"   - Test:  {os.path.join(data_dir, 'images', 'test')}")
    print("7. Copy labels to:")
    print(f"   - Train: {os.path.join(data_dir, 'labels', 'train')}")
    print(f"   - Val:   {os.path.join(data_dir, 'labels', 'val')}")
    print(f"   - Test:  {os.path.join(data_dir, 'labels', 'test')}")
    print()
    print("=" * 60)
    print("  OPTION 2: Kaggle")
    print("=" * 60)
    print()
    print("1. Go to: https://www.kaggle.com/datasets")
    print("2. Search for 'Safety Helmet Detection YOLO'")
    print("3. Download the dataset")
    print("4. Place images and labels in the directories above")
    print()
    print("=" * 60)
    print("  OPTION 3: Auto-Download with Roboflow API")
    print("=" * 60)
    print()
    print("If you have a Roboflow API key, you can auto-download:")
    print()
    print("  pip install roboflow")
    print("  python -c \"")
    print("  from roboflow import Roboflow")
    print("  rf = Roboflow(api_key='YOUR_API_KEY')")
    print("  project = rf.workspace().project('safety-helmet-detection')")
    print("  dataset = project.version(1).download('yolov8')\"")
    print()

    # Update data.yaml with absolute paths
    create_dataset_yaml(data_dir, DATA_YAML)

    print("=" * 60)
    print(f"  Dataset YAML updated: {DATA_YAML}")
    print("=" * 60)
    print()
    print("After placing your dataset files, you can start training:")
    print("  python -m src.train")
    print()


def verify_dataset(data_dir):
    """
    Verify the dataset structure and report statistics.

    Args:
        data_dir (str): Root directory of the dataset.

    Returns:
        bool: True if the dataset structure is valid.
    """
    print()
    print("=" * 60)
    print("  Dataset Verification")
    print("=" * 60)
    print()

    splits = ['train', 'val', 'test']
    valid = True

    for split in splits:
        img_dir = os.path.join(data_dir, "images", split)
        lbl_dir = os.path.join(data_dir, "labels", split)

        if os.path.exists(img_dir):
            # Count images
            img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            images = [f for f in os.listdir(img_dir)
                      if os.path.splitext(f)[1].lower() in img_extensions]
            num_images = len(images)

            # Count labels
            labels = []
            if os.path.exists(lbl_dir):
                labels = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]

            num_labels = len(labels)

            status = "✓" if num_images > 0 else "✗"
            print(f"  {status} {split:>5}: {num_images:>5} images, {num_labels:>5} labels")

            if num_images == 0:
                valid = False
        else:
            print(f"  ✗ {split:>5}: Directory not found")
            valid = False

    print()
    if valid:
        print("  Dataset is ready for training!")
    else:
        print("  Dataset is incomplete. Please add images and labels.")
        print("  Run: python -m src.dataset_manager --setup")

    print()
    return valid


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Helmet Detection Dataset Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.dataset_manager --setup       # Create directories & show instructions
  python -m src.dataset_manager --verify      # Verify dataset integrity
  python -m src.dataset_manager --output data/ # Use custom data directory
        """
    )

    parser.add_argument(
        "--setup", action="store_true", default=True,
        help="Set up dataset directories and show download instructions"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify the dataset structure and count files"
    )
    parser.add_argument(
        "--output", type=str, default=DATA_DIR,
        help=f"Output directory for dataset (default: {DATA_DIR})"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.verify:
        verify_dataset(args.output)
    else:
        download_sample_dataset(args.output)


if __name__ == "__main__":
    main()
