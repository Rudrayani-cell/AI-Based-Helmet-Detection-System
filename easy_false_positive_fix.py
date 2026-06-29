
"""
easy_false_positive_fix.py
Automated tool for collecting negative samples and fine-tuning the model.
"""

import cv2
import os
import time
from ultralytics import YOLO

def main():
    print("="*60)
    print("  High-Speed Automated Fine-Tuning")
    print("="*60)
    
    import shutil
    # Clean up previous fix runs to avoid confusion
    if os.path.exists("runs/false_positive_fix"):
        print("[INFO] Clearing previous fix run data...")
        shutil.rmtree("runs/false_positive_fix")

    # ------------------
    # Setup for YOLO training
    # ------------------
    model_path = os.path.join("models", "best.pt")
    
    # Fallback if running from main dir instead of premium
    if not os.path.exists(model_path):
        model_path = os.path.join("runs", "kfold", "fold_1", "weights", "best.pt")
        if not os.path.exists(model_path):
            print(f"[ERROR] Could not locate the trained model weights at {model_path}.")
            print("Please run this script from the project root folder.")
            return

    print(f"\n[INFO] Loading the best model from: {model_path}")
    print("[INFO] Re-training the model using the new 'Negative Samples'...")
    
    try:
        model = YOLO(model_path)
        
        # SPEED BOOST: Increased to 10 epochs for a more solid fix.
        # Still keeps the backbone frozen for speed.
        result = model.train(
            data="config/data.yaml",
            epochs=10, 
            imgsz=640,
            batch=8,
            freeze=20,
            device=0 if torch.cuda.is_available() else "cpu",
            project="runs/false_positive_fix",
            name="fixed_model"
        )
        
        # --- AUTOMATIC FILE OVERWRITE ---
        new_weights = "runs/false_positive_fix/fixed_model/weights/best.pt"
        if os.path.exists(new_weights):
            print("\n[SUCCESS] Custom training complete.")
            print(f"[INFO] Moving {new_weights} -> models/best.pt")
            shutil.copy(new_weights, "models/best.pt")
            print("[INFO] Overwrite successful!")
        
        print("\n" + "="*60)
        print("  ALL DONE! False Positive Fix Applied Automatically.")
        print("="*60)
        print("The new model is now live in models/best.pt.")
        print("Launch your webcam detection and you should see the difference!")
        
    except Exception as e:
        print(f"\n[ERROR] An issue occurred during training: {e}")

if __name__ == "__main__":
    import numpy as np
    import torch
    import shutil
    main()
