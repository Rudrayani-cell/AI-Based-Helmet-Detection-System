"""
Dataset Download Script
=======================
Automated script to download the best motorcycle/riding helmet detection
dataset and set it up for training with YOLOv8.

Recommended datasets (motorcycle/riding helmet focused):
  1. "Smart Helmet Detection using YOLO V8" (Kaggle)
     - Motorcycle-specific with driver/passenger helmet status
     - Already in YOLO format
  2. "Bike Helmet Detection v2" (Roboflow)
     - 3,735 images, With Helmet / Without Helmet
     - YOLOv8 export ready

Usage:
    python download_dataset.py
    python download_dataset.py --method roboflow
    python download_dataset.py --method kaggle
    python download_dataset.py --verify
"""

import os
import sys
import shutil
import zipfile
import argparse
import yaml

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
DATA_YAML = os.path.join(CONFIG_DIR, "data.yaml")


def create_directory_structure():
    """Create the required YOLO directory structure."""
    dirs = [
        os.path.join(DATA_DIR, "images", "train"),
        os.path.join(DATA_DIR, "images", "val"),
        os.path.join(DATA_DIR, "images", "test"),
        os.path.join(DATA_DIR, "labels", "train"),
        os.path.join(DATA_DIR, "labels", "val"),
        os.path.join(DATA_DIR, "labels", "test"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("[INFO] Directory structure created.")


def update_data_yaml():
    """Update the data.yaml file with absolute paths."""
    config = {
        'path': os.path.abspath(DATA_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 2,
        'names': {0: 'Helmet', 1: 'No Helmet'}
    }
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(DATA_YAML, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"[INFO] Data YAML updated: {DATA_YAML}")


def download_roboflow_dataset():
    """
    Download the Bike Helmet Detection dataset from Roboflow.
    3,735 images | With Helmet / Without Helmet | YOLOv8 format.
    """
    print()
    print("=" * 65)
    print("  DOWNLOAD: Bike Helmet Detection v2 (Roboflow)")
    print("  3,735 images | YOLOv8 format | With/Without Helmet")
    print("=" * 65)
    print()

    try:
        from roboflow import Roboflow

        print("[INFO] Roboflow library found. Attempting API download...")
        print()
        print("To download via Roboflow API, you need an API key.")
        print("1. Go to: https://app.roboflow.com/settings/api")
        print("2. Copy your API key")
        api_key = input("3. Paste your Roboflow API key (or press Enter to skip): ").strip()

        if api_key:
            rf = Roboflow(api_key=api_key)
            project = rf.workspace("yolo-asijt").project("bike-helmet-detection-2vdpn")
            version = project.version(2)
            dataset = version.download("yolov8", location=DATA_DIR)
            print(f"[SUCCESS] Dataset downloaded to: {DATA_DIR}")
            _organize_roboflow_download(DATA_DIR)
            return True
        else:
            print("[INFO] No API key provided. Falling back to manual download...")
            return False

    except ImportError:
        print("[INFO] Roboflow library not installed.")
        print("[INFO] Installing roboflow...")
        os.system(f"{sys.executable} -m pip install roboflow --quiet")

        try:
            from roboflow import Roboflow
            print("[INFO] Roboflow installed successfully!")
            return download_roboflow_dataset()  # Retry
        except ImportError:
            print("[WARNING] Could not install roboflow. Falling back to manual method.")
            return False


def _organize_roboflow_download(data_dir):
    """
    Organize a Roboflow download into the standard directory structure.
    Roboflow downloads usually create train/valid/test folders with
    images and labels subfolders.
    """
    print("[INFO] Organizing dataset into standard structure...")

    # Map Roboflow folder names to our standard names
    mapping = {
        'valid': 'val',
        'validation': 'val',
    }

    for split_name in ['train', 'valid', 'validation', 'val', 'test']:
        src_dir = os.path.join(data_dir, split_name)
        if not os.path.exists(src_dir):
            continue

        # Determine target split name
        target_split = mapping.get(split_name, split_name)

        # Move images
        src_images = os.path.join(src_dir, "images")
        if os.path.exists(src_images):
            dst_images = os.path.join(data_dir, "images", target_split)
            os.makedirs(dst_images, exist_ok=True)
            for f in os.listdir(src_images):
                shutil.move(os.path.join(src_images, f), os.path.join(dst_images, f))

        # Move labels
        src_labels = os.path.join(src_dir, "labels")
        if os.path.exists(src_labels):
            dst_labels = os.path.join(data_dir, "labels", target_split)
            os.makedirs(dst_labels, exist_ok=True)
            for f in os.listdir(src_labels):
                shutil.move(os.path.join(src_labels, f), os.path.join(dst_labels, f))

        # Check if images/labels are directly in the split folder (flat structure)
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        for f in os.listdir(src_dir):
            fpath = os.path.join(src_dir, f)
            if os.path.isfile(fpath):
                ext = os.path.splitext(f)[1].lower()
                if ext in img_extensions:
                    dst = os.path.join(data_dir, "images", target_split, f)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.move(fpath, dst)
                elif ext == '.txt' and f != 'classes.txt':
                    dst = os.path.join(data_dir, "labels", target_split, f)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.move(fpath, dst)

    print("[INFO] Dataset organized successfully!")


def download_kaggle_dataset():
    """
    Download the Smart Helmet Detection dataset from Kaggle.
    Motorcycle-specific: Driver/Passenger with/without helmet.
    Requires kaggle CLI and API key.
    """
    print()
    print("=" * 65)
    print("  DOWNLOAD: Smart Helmet Detection (Kaggle)")
    print("  Motorcycle riders | YOLO format | driver/passenger")
    print("=" * 65)
    print()

    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "kaggle", "--quiet"],
            capture_output=True, text=True
        )

        # Check if kaggle is configured
        kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
        kaggle_json = os.path.join(kaggle_dir, "kaggle.json")

        if not os.path.exists(kaggle_json):
            print("[WARNING] Kaggle API key not found!")
            print()
            print("To set up Kaggle API:")
            print("1. Go to: https://www.kaggle.com/settings")
            print("2. Click 'Create New Token' under API section")
            print("3. Save kaggle.json to: " + kaggle_dir)
            print()
            return False

        # Download the dataset
        print("[INFO] Downloading Smart Helmet Detection dataset...")
        download_dir = os.path.join(DATA_DIR, "_kaggle_download")
        os.makedirs(download_dir, exist_ok=True)

        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d",
             "aneesarom/smart-helmet-detection-using-yolo-v8",
             "-p", download_dir, "--unzip"],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            print("[SUCCESS] Dataset downloaded!")
            _organize_kaggle_motorcycle_dataset(download_dir)
            return True
        else:
            print(f"[ERROR] Download failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"[ERROR] Kaggle download failed: {e}")
        return False


def _organize_kaggle_motorcycle_dataset(download_dir):
    """
    Organize the Kaggle Smart Helmet Detection dataset.
    This dataset is already in YOLO format with train/val/test splits.
    Classes are remapped to: 0=Helmet, 1=No Helmet.
    """
    print("[INFO] Organizing Kaggle motorcycle helmet dataset...")

    # The Kaggle dataset may have various structures.
    # Look for train/valid/test directories with images and labels.
    for root, dirs, files in os.walk(download_dir):
        for d in dirs:
            full_path = os.path.join(root, d)

            # Check for YOLO-structured directories
            if d in ['train', 'valid', 'validation', 'val', 'test']:
                target_split = 'val' if d in ['valid', 'validation'] else d

                # Look for images and labels subfolders
                img_src = os.path.join(full_path, "images")
                lbl_src = os.path.join(full_path, "labels")

                if os.path.exists(img_src):
                    dst = os.path.join(DATA_DIR, "images", target_split)
                    os.makedirs(dst, exist_ok=True)
                    for f in os.listdir(img_src):
                        src_file = os.path.join(img_src, f)
                        if os.path.isfile(src_file):
                            shutil.copy2(src_file, os.path.join(dst, f))
                    print(f"[INFO] Copied {target_split} images")

                if os.path.exists(lbl_src):
                    dst = os.path.join(DATA_DIR, "labels", target_split)
                    os.makedirs(dst, exist_ok=True)
                    for f in os.listdir(lbl_src):
                        src_file = os.path.join(lbl_src, f)
                        if os.path.isfile(src_file):
                            shutil.copy2(src_file, os.path.join(dst, f))
                    print(f"[INFO] Copied {target_split} labels")

    # Also check for a data.yaml in the downloaded data to understand class mapping
    for root, dirs, files in os.walk(download_dir):
        if 'data.yaml' in files:
            src_yaml = os.path.join(root, 'data.yaml')
            print(f"[INFO] Found dataset config: {src_yaml}")
            with open(src_yaml, 'r') as f:
                dataset_config = yaml.safe_load(f)
            if 'names' in dataset_config:
                print(f"[INFO] Original classes: {dataset_config['names']}")
            break

    # Clean up
    shutil.rmtree(download_dir, ignore_errors=True)
    print("[INFO] Dataset organization complete!")


def manual_download_instructions():
    """Show step-by-step manual download instructions for motorcycle helmet datasets."""
    print()
    print("=" * 65)
    print("  MANUAL DOWNLOAD INSTRUCTIONS")
    print("  (Motorcycle / Riding Helmet Detection)")
    print("=" * 65)
    print()
    print("  [*] OPTION A: Kaggle (RECOMMENDED - Best for Riding Helmets)")
    print("  " + "-" * 55)
    print("  'Smart Helmet Detection using YOLO V8'")
    print("  Motorcycle-specific: Driver/Passenger with/without helmet")
    print()
    print("  1. Go to:")
    print("     https://www.kaggle.com/datasets/aneesarom/smart-helmet-detection-using-yolo-v8")
    print()
    print("  2. Click 'Download' (requires free Kaggle account)")
    print("  3. Extract the ZIP file")
    print("  4. Copy the contents into the data/ folder:")
    print(f"     train/images/* -> {os.path.join(DATA_DIR, 'images', 'train')}")
    print(f"     train/labels/* -> {os.path.join(DATA_DIR, 'labels', 'train')}")
    print(f"     valid/images/* -> {os.path.join(DATA_DIR, 'images', 'val')}")
    print(f"     valid/labels/* -> {os.path.join(DATA_DIR, 'labels', 'val')}")
    print(f"     test/images/*  -> {os.path.join(DATA_DIR, 'images', 'test')}")
    print(f"     test/labels/*  -> {os.path.join(DATA_DIR, 'labels', 'test')}")
    print()
    print("  [*] OPTION B: Roboflow 'Bike Helmet Detection v2'")
    print("  " + "-" * 50)
    print("  3,735 images | With Helmet / Without Helmet")
    print()
    print("  1. Go to:")
    print("     https://universe.roboflow.com/yolo-asijt/bike-helmet-detection-2vdpn")
    print("  2. Click 'Dataset' -> 'Download Dataset'")
    print("  3. Select format: 'YOLOv8'")
    print("  4. Download and extract the ZIP")
    print("  5. Copy files as described in Option A")
    print()
    print("  [*] OPTION C: Roboflow 'With/Without Helmet'")
    print("  " + "-" * 44)
    print("  4,169 images | General helmet/no-helmet for riders")
    print()
    print("  1. Go to:")
    print("     https://universe.roboflow.com/helmet-xqbgb/with-and-without-helmet")
    print("  2. Download in YOLOv8 format")
    print("  3. Place files as described in Option A")
    print()


def verify_dataset():
    """Verify dataset is correctly set up and show stats."""
    print()
    print("=" * 65)
    print("  DATASET VERIFICATION")
    print("=" * 65)
    print()

    total_images = 0
    total_labels = 0
    all_good = True

    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(DATA_DIR, "images", split)
        lbl_dir = os.path.join(DATA_DIR, "labels", split)

        img_count = 0
        lbl_count = 0

        if os.path.exists(img_dir):
            img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            img_count = len([f for f in os.listdir(img_dir)
                           if os.path.splitext(f)[1].lower() in img_extensions])

        if os.path.exists(lbl_dir):
            lbl_count = len([f for f in os.listdir(lbl_dir) if f.endswith('.txt')])

        status = "[OK]" if img_count > 0 else "[--]"
        if img_count == 0:
            all_good = False

        print(f"  {status}  {split:>5}:  {img_count:>5} images  |  {lbl_count:>5} labels")
        total_images += img_count
        total_labels += lbl_count

    print(f"  {'-' * 45}")
    print(f"     TOTAL:  {total_images:>5} images  |  {total_labels:>5} labels")
    print()

    if all_good and total_images > 0:
        print("  [OK] Dataset is READY for training!")
        print()
        print("  Next step:")
        print("    python -m src.train --epochs 50")
    else:
        print("  [!!] Dataset is incomplete or missing.")
        print("     Please download a dataset using the instructions above.")
    print()

    return all_good


def main():
    parser = argparse.ArgumentParser(
        description="Download motorcycle/riding helmet detection dataset for training"
    )
    parser.add_argument(
        "--method", type=str, choices=["roboflow", "kaggle", "manual"],
        default="roboflow",
        help="Download method (default: roboflow)"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Only verify existing dataset"
    )
    args = parser.parse_args()

    print()
    print("=" * 65)
    print("  MOTORCYCLE HELMET DETECTION - DATASET DOWNLOAD TOOL")
    print("=" * 65)
    print()

    if args.verify:
        verify_dataset()
        return

    # Create directory structure
    create_directory_structure()

    # Try downloading
    success = False

    if args.method == "roboflow":
        success = download_roboflow_dataset()
    elif args.method == "kaggle":
        success = download_kaggle_dataset()

    if not success:
        manual_download_instructions()

    # Update data.yaml
    update_data_yaml()

    # Verify
    verify_dataset()


if __name__ == "__main__":
    main()
