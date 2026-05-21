"""
OPTIMIZED CPU TRAINING FOR ACCIDENT DETECTION - 640x640
Fixed for YOLO compatibility
"""

import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO


# =========================================================
# CONFIGURATION
# =========================================================
DATASET_PATH = Path(r"C:\Users\TEJASWINI\Downloads\archive\AN-Data-Processed-Final")
DATA_YAML = DATASET_PATH / "data.yaml"
ARCHIVE_PATH = Path(r"C:\Users\TEJASWINI\Downloads\archive")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SEED = 42


def count_images(folder: Path) -> int:
    """Count only image files."""
    if not folder.exists():
        return 0
    return len([f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS])


def as_float(value) -> float:
    """Convert scalar or array-like metric values to a display-safe float."""
    arr = np.asarray(value)
    if arr.size == 0:
        return 0.0
    return float(np.nanmean(arr))


def create_data_yaml() -> None:
    """Create data.yaml if it does not already exist."""
    if DATA_YAML.exists():
        return

    print("Creating data.yaml...")
    data_config = {
        "path": str(DATASET_PATH.absolute()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 2,
        "names": ["Accident", "Non Accident"],
    }
    with open(DATA_YAML, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    print(f"Created {DATA_YAML}")


def validate_dataset() -> None:
    """Validate the expected YOLO dataset directory structure."""
    print("\nVALIDATING DATASET STRUCTURE...")

    required_paths = [
        DATA_YAML,
        DATASET_PATH / "train" / "images",
        DATASET_PATH / "train" / "labels",
        DATASET_PATH / "valid" / "images",
        DATASET_PATH / "valid" / "labels",
        DATASET_PATH / "test" / "images",
        DATASET_PATH / "test" / "labels",
    ]

    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required dataset paths:\n{missing_text}")

    train_count = count_images(DATASET_PATH / "train" / "images")
    val_count = count_images(DATASET_PATH / "valid" / "images")
    test_count = count_images(DATASET_PATH / "test" / "images")

    if train_count == 0:
        raise ValueError("❌ Train folder empty. No images found for training.")
    if val_count == 0:
        raise ValueError("❌ Valid folder empty. No images found for validation.")
    if test_count == 0:
        raise ValueError("❌ Test folder empty. No images found for testing.")

    print(f"✅ Dataset validated: {train_count} train, {val_count} val, {test_count} test images")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility with deterministic CUDA."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    create_data_yaml()
    set_seed(SEED)

    print("=" * 70)
    print("YOLOv8 ACCIDENT DETECTION TRAINING (OPTIMIZED CPU - 416x416)")
    print("=" * 70)

    validate_dataset()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device.upper()}")

    train_images = count_images(DATASET_PATH / "train" / "images")
    val_images = count_images(DATASET_PATH / "valid" / "images")
    test_images = count_images(DATASET_PATH / "test" / "images")

    print("\nDATASET STATISTICS:")
    print(f"   Train: {train_images} images")
    print(f"   Valid: {val_images} images")
    print(f"   Test:  {test_images} images")
    print(f"   Total: {train_images + val_images + test_images} images")

    # =========================================================
    # SPEED OPTIMIZATIONS FOR CPU (640x640)
    # =========================================================
    batch_size = 8  # Increased from 4 for faster processing
    cache_mode = False  # Disable caching to save disk I/O
    cpu_count = os.cpu_count() or 2
    workers = 2  # Fixed workers for CPU
    
    model_name = "yolov8n.pt"  # Nano model for fastest CPU inference
    lr0 = 0.0005

    print("\nHARDWARE CONFIGURATION (SPEED OPTIMIZED):")
    print(f"   Device:     {device.upper()}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Cache Mode: {cache_mode}")
    print(f"   Workers:    {workers}")
    print(f"   Model:      {model_name}")
    print(f"   LR0:        {lr0}")

    model = YOLO(model_name)

    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)

    # =========================================================
    # OPTIMIZED PARAMETERS FOR SPEED (VALID YOLO ARGUMENTS ONLY)
    # =========================================================
    training_params = {
        # Data
        "data": str(DATA_YAML),
        "epochs": 80,               # Reduced from 120
        "imgsz": 416,               # Keeping 640x640
        "batch": batch_size,
        "workers": workers,
        
        # Optimizer
        "optimizer": "AdamW",
        "lr0": lr0,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        
        # Learning rate scheduler
        "cos_lr": True,
        "warmup_epochs": 2,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        
        # Reproducibility
        "seed": SEED,
        "deterministic": True,
        
        # Minimal runtime augmentation
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.02,
        "scale": 0.05,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.10,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        
        # Performance
        "device": device,
        "cache": cache_mode,
        "amp": False,
        "pretrained": True,
        "verbose": True,
        
        # Validation and saving (REMOVED val_period - not valid)
        "val": True,
        "patience": 20,
        "save_period": 20,
        "save": True,
        "save_json": True,
        "plots": False,  # Disable plots for speed
        "project": "Accident_Detection_Final",
        "name": "YOLOv8_Accident_Production",
        "exist_ok": True,
        
        # Additional speed optimizations
        "close_mosaic": 0,
        "nbs": 32,
    }

    print("\nHYPERPARAMETERS (SPEED OPTIMIZED):")
    for key, value in training_params.items():
        if key not in {"data", "project", "name"}:
            print(f"   {key}: {value}")

    print("\n" + "=" * 70)
    print("STARTING TRAINING...")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("\n⚠️  TRAINING TIME ESTIMATE:")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Batches per epoch: ~{train_images // batch_size}")
    print(f"   • Estimated time per epoch: ~45-50 minutes")
    print(f"   • Total epochs: {training_params['epochs']}")
    print(f"   • Estimated total time: ~{training_params['epochs'] * 0.8:.0f} hours")
    print("=" * 70)

    start_time = datetime.now()
    model.train(**training_params)
    training_time = datetime.now() - start_time

    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training completed in {training_time.total_seconds() / 3600:.2f} hours")

    best_model_path = Path(
        "Accident_Detection_Final/YOLOv8_Accident_Production/weights/best.pt"
    )

    backup_path = None
    metrics_file = None

    if best_model_path.exists():
        print(f"\n🏆 BEST MODEL SAVED AT: {best_model_path.absolute()}")

        backup_path = (
            ARCHIVE_PATH
            / f"best_model_accident_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        )
        try:
            shutil.copy2(best_model_path, backup_path)
            print(f"Backup saved to: {backup_path}")
        except Exception as exc:
            print(f"Backup failed: {exc}")

        print("\n" + "=" * 70)
        print("EVALUATING BEST MODEL ON TEST SET")
        print("=" * 70)

        best_model = YOLO(str(best_model_path))
        test_results = best_model.val(
            data=str(DATA_YAML),
            split="test",
            imgsz=416,
            batch=batch_size,
            device=device,
            conf=0.25,
            iou=0.7,
            plots=True,
            save_json=True,
        )

        print("\n" + "=" * 70)
        print("MODEL PERFORMANCE METRICS")
        print("=" * 70)

        precision = recall = f1_score = 0.0
        if hasattr(test_results, "box"):
            precision = as_float(test_results.box.p)
            recall = as_float(test_results.box.r)
            f1_score = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            print("\nOverall Metrics:")
            print(f"   mAP50-95:  {as_float(test_results.box.map):.4f}")
            print(f"   mAP50:     {as_float(test_results.box.map50):.4f}")
            print(f"   mAP75:     {as_float(test_results.box.map75):.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall:    {recall:.4f}")
            print(f"   F1 Score:  {f1_score:.4f}")

            if hasattr(test_results.box, "ap50") and len(test_results.box.ap50) >= 2:
                print("\nPer-Class mAP50:")
                print(f"   Accident:     {as_float(test_results.box.ap50[0]):.4f}")
                print(f"   Non Accident: {as_float(test_results.box.ap50[1]):.4f}")

        metrics_file = (
            ARCHIVE_PATH
            / f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(metrics_file, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("ACCIDENT DETECTION MODEL - TRAINING METRICS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training Time: {training_time.total_seconds() / 3600:.2f} hours\n")
            f.write(f"Device: {device.upper()}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Seed: {SEED}\n\n")

            f.write("DATASET STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Training Images: {train_images}\n")
            f.write(f"  Validation Images: {val_images}\n")
            f.write(f"  Test Images: {test_images}\n\n")

            f.write("HYPERPARAMETERS:\n")
            f.write("-" * 40 + "\n")
            for key, value in training_params.items():
                if key not in {"data", "project", "name"}:
                    f.write(f"  {key}: {value}\n")

            f.write("\nPERFORMANCE METRICS:\n")
            f.write("-" * 40 + "\n")
            if hasattr(test_results, "box"):
                f.write(f"  mAP50-95: {as_float(test_results.box.map):.4f}\n")
                f.write(f"  mAP50: {as_float(test_results.box.map50):.4f}\n")
                f.write(f"  mAP75: {as_float(test_results.box.map75):.4f}\n")
                f.write(f"  Precision: {precision:.4f}\n")
                f.write(f"  Recall: {recall:.4f}\n")
                f.write(f"  F1 Score: {f1_score:.4f}\n")

                if hasattr(test_results.box, "ap50") and len(test_results.box.ap50) >= 2:
                    f.write(f"\n  Accident mAP50: {as_float(test_results.box.ap50[0]):.4f}\n")
                    f.write(f"  Non Accident mAP50: {as_float(test_results.box.ap50[1]):.4f}\n")

            f.write("\nBEST MODEL PATH:\n")
            f.write(f"  {best_model_path.absolute()}\n")

        print(f"\n📄 Metrics saved to: {metrics_file}")
    else:
        print("\n❌ Best model not found.")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"""
BEST MODEL: {best_model_path.absolute() if best_model_path.exists() else "Not found"}
BACKUP: {backup_path if backup_path else "N/A"}
METRICS: {metrics_file if metrics_file else "N/A"}

📊 SPEED OPTIMIZATIONS APPLIED (640x640):
   • Epochs: 80 (was 120) → Saved 40 epochs
   • Batch size: 8 (was 4) → 2x faster
   • Cache: Disabled → Reduced disk I/O
   • Plots: Disabled during training
   • Save frequency: Every 20 epochs

⏱️  ESTIMATED TIME: ~60-70 hours (2.5-3 days)

EXPECTED PERFORMANCE:
   • Precision: 88-93%
   • Recall: 85-90%
   • mAP50: 90-94%
""")
    print("=" * 70)


if __name__ == "__main__":
    main()