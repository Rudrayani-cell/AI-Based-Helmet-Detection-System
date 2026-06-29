"""
Merge and organize both downloaded helmet datasets into the project data/ folder.
Combines:
  1. Roboflow "Bike Helmet Detection v2" (2 classes: With Helmet, Without Helmet)
  2. Kaggle "Smart Helmet Detection using YOLO V8" (7 classes, remapped to 2)

Output: Unified 2-class dataset (0=Helmet, 1=No Helmet)
"""

import os
import shutil
import glob

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Source dataset paths
ROBOFLOW_DIR = r"C:\Users\hp\Downloads\Bike Helmet Detection.v2-more-preprocessing-augmentation.yolov8"
KAGGLE_DIR = r"C:\Users\hp\Downloads\Smart Helmet Detection using YOLO V8\data"

# ---- Roboflow class mapping ----
# Original: 0=With Helmet, 1=Without Helmet
# Target:   0=Helmet,      1=No Helmet
ROBOFLOW_CLASS_MAP = {
    0: 0,   # With Helmet   -> Helmet
    1: 1,   # Without Helmet -> No Helmet
}

# ---- Kaggle class mapping ----
# Original: 0=driver_with_helmet, 1=bike, 2=driver,
#           3=passenger_with_helemt, 4=passenger,
#           5=driver_without_helmet, 6=passenger_without_helemt
# Target:   0=Helmet, 1=No Helmet
# We skip class 1 (bike) as it is not relevant to helmet detection.
KAGGLE_CLASS_MAP = {
    0: 0,     # driver_with_helmet       -> Helmet
    1: None,  # bike                     -> SKIP
    2: None,  # driver (ambiguous)       -> SKIP
    3: 0,     # passenger_with_helemt    -> Helmet
    4: None,  # passenger (ambiguous)    -> SKIP
    5: 1,     # driver_without_helmet    -> No Helmet
    6: 1,     # passenger_without_helemt -> No Helmet
}


def count_files(directory, extensions):
    """Count files with given extensions in a directory."""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory)
                if os.path.splitext(f)[1].lower() in extensions])


def copy_images(src_dir, dst_dir, prefix=""):
    """Copy all image files from src to dst, with optional prefix to avoid name collisions."""
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    os.makedirs(dst_dir, exist_ok=True)
    copied = 0
    for f in os.listdir(src_dir):
        if os.path.splitext(f)[1].lower() in img_exts:
            dst_name = f"{prefix}{f}" if prefix else f
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, dst_name))
            copied += 1
    return copied


def remap_and_copy_labels(src_dir, dst_dir, class_map, prefix=""):
    """
    Read YOLO label files, remap class IDs, and save to destination.
    Lines with class IDs mapped to None are skipped.
    """
    os.makedirs(dst_dir, exist_ok=True)
    copied = 0
    skipped_files = 0

    for f in os.listdir(src_dir):
        if not f.endswith('.txt'):
            continue

        src_path = os.path.join(src_dir, f)
        dst_name = f"{prefix}{f}" if prefix else f
        dst_path = os.path.join(dst_dir, dst_name)

        new_lines = []
        with open(src_path, 'r') as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                orig_class = int(parts[0])
                new_class = class_map.get(orig_class, None)
                if new_class is not None:
                    parts[0] = str(new_class)
                    new_lines.append(' '.join(parts))

        if new_lines:
            with open(dst_path, 'w') as fout:
                fout.write('\n'.join(new_lines) + '\n')
            copied += 1
        else:
            skipped_files += 1

    return copied, skipped_files


def main():
    print()
    print("=" * 65)
    print("  MERGING DATASETS INTO PROJECT")
    print("=" * 65)
    print()

    # Create directory structure
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(DATA_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, "labels", split), exist_ok=True)

    total_images = 0
    total_labels = 0

    # ================================================================
    # DATASET 1: Roboflow Bike Helmet Detection v2
    # ================================================================
    print("[1/2] Processing Roboflow 'Bike Helmet Detection v2'...")
    print("      Classes: With Helmet (0) -> Helmet, Without Helmet (1) -> No Helmet")
    print()

    roboflow_splits = {
        'train': 'train',
        'valid': 'val',
        'test': 'test'
    }

    for src_split, dst_split in roboflow_splits.items():
        src_img = os.path.join(ROBOFLOW_DIR, src_split, "images")
        src_lbl = os.path.join(ROBOFLOW_DIR, src_split, "labels")
        dst_img = os.path.join(DATA_DIR, "images", dst_split)
        dst_lbl = os.path.join(DATA_DIR, "labels", dst_split)

        if os.path.exists(src_img):
            n_img = copy_images(src_img, dst_img, prefix="rb_")
            total_images += n_img
            print(f"      {dst_split:>5} images: {n_img}")

        if os.path.exists(src_lbl):
            n_lbl, n_skip = remap_and_copy_labels(src_lbl, dst_lbl, ROBOFLOW_CLASS_MAP, prefix="rb_")
            total_labels += n_lbl
            print(f"      {dst_split:>5} labels: {n_lbl} (skipped {n_skip})")

    print()

    # ================================================================
    # DATASET 2: Kaggle Smart Helmet Detection
    # ================================================================
    print("[2/2] Processing Kaggle 'Smart Helmet Detection'...")
    print("      Remapping 7 classes -> 2 classes (Helmet / No Helmet)")
    print("      Skipping: bike, driver (ambiguous), passenger (ambiguous)")
    print()

    # Note: Kaggle dataset has "vaid" (typo) instead of "valid"
    kaggle_splits = {
        'train': 'train',
        'vaid': 'val',     # typo in original dataset
        'test': 'test'
    }

    for src_split, dst_split in kaggle_splits.items():
        src_img = os.path.join(KAGGLE_DIR, src_split, "images")
        src_lbl = os.path.join(KAGGLE_DIR, src_split, "labels")
        dst_img = os.path.join(DATA_DIR, "images", dst_split)
        dst_lbl = os.path.join(DATA_DIR, "labels", dst_split)

        if os.path.exists(src_img):
            n_img = copy_images(src_img, dst_img, prefix="kg_")
            total_images += n_img
            print(f"      {dst_split:>5} images: {n_img}")

        if os.path.exists(src_lbl):
            n_lbl, n_skip = remap_and_copy_labels(src_lbl, dst_lbl, KAGGLE_CLASS_MAP, prefix="kg_")
            total_labels += n_lbl
            print(f"      {dst_split:>5} labels: {n_lbl} (skipped {n_skip} empty)")

    print()
    print("=" * 65)
    print(f"  TOTAL: {total_images} images, {total_labels} labels")
    print("=" * 65)
    print()

    # ================================================================
    # Update data.yaml with absolute paths
    # ================================================================
    import yaml
    config = {
        'path': os.path.abspath(DATA_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 2,
        'names': {0: 'Helmet', 1: 'No Helmet'}
    }
    yaml_path = os.path.join(PROJECT_ROOT, "config", "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"[INFO] Updated {yaml_path}")

    # ================================================================
    # Final verification
    # ================================================================
    print()
    print("  VERIFICATION")
    print("  " + "-" * 50)
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(DATA_DIR, "images", split)
        lbl_dir = os.path.join(DATA_DIR, "labels", split)
        n_i = count_files(img_dir, {'.jpg', '.jpeg', '.png', '.bmp', '.webp'})
        n_l = count_files(lbl_dir, {'.txt'})
        print(f"  {split:>5}:  {n_i:>5} images  |  {n_l:>5} labels")

    print()
    print("  Dataset is READY! Start training with:")
    print("    python -m src.train --epochs 50")
    print()


if __name__ == "__main__":
    main()
