"""
COMPLETE DATASET BALANCING SCRIPT - FIXED VERSION
Properly copies both images AND labels
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from tqdm import tqdm
from collections import Counter

# =========================================================
# CONFIGURATION
# =========================================================
DATASET_ROOT = Path(r"C:\Users\TEJASWINI\Downloads\archive\AN-Data")
BALANCED_ROOT = Path(r"C:\Users\TEJASWINI\Downloads\archive\AN-Data-Balanced-Final-V2")

# Target counts
TARGET_ACCIDENT = 3000
TARGET_NO_ACCIDENT = 3000

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Set seed for reproducibility
random.seed(42)

print("="*70)
print("COMPLETE DATASET BALANCING FROM SCRATCH (FIXED)")
print("="*70)

# =========================================================
# STEP 1: ANALYZE DATASET AND COLLECT IMAGES BY CLASS
# =========================================================
print("\n📊 STEP 1: Analyzing dataset and collecting images by class")

def collect_images_with_labels(images_dir, labels_dir):
    """Collect images with their label information"""
    
    accident_images = []  # List of (image_path, label_path)
    no_accident_images = []  # List of (image_path, label_path)
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(images_dir.glob(ext))
    
    for img_path in tqdm(image_files, desc=f"Processing {images_dir.parent.name}"):
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        # If no label file, treat as no-accident
        if not label_path.exists():
            no_accident_images.append((img_path, None))
            continue
        
        # Read label file
        try:
            with open(label_path, 'r') as f:
                content = f.read().strip()
                
                # Empty label = no-accident
                if not content:
                    no_accident_images.append((img_path, label_path))
                    continue
                
                # Check each line for accident class (0)
                has_accident = False
                lines = content.split('\n')
                
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 1:
                            class_id = int(float(parts[0]))
                            if class_id == 0:  # Accident class
                                has_accident = True
                                break
                
                if has_accident:
                    accident_images.append((img_path, label_path))
                else:
                    no_accident_images.append((img_path, label_path))
                    
        except Exception as e:
            print(f"Error reading {label_path}: {e}")
            no_accident_images.append((img_path, None))
    
    return accident_images, no_accident_images

# Collect from each split
all_accident = []
all_no_accident = []

for split in ['train', 'valid', 'test']:
    images_dir = DATASET_ROOT / split / 'images'
    labels_dir = DATASET_ROOT / split / 'labels'
    
    if not images_dir.exists():
        print(f"⚠️ {split} images directory not found")
        continue
    
    print(f"\n📁 Analyzing {split.upper()} set...")
    accident, no_accident = collect_images_with_labels(images_dir, labels_dir)
    
    all_accident.extend(accident)
    all_no_accident.extend(no_accident)
    
    print(f"   Accident: {len(accident)}")
    print(f"   No-Accident: {len(no_accident)}")

print("\n" + "="*70)
print("📊 ORIGINAL DATASET STATISTICS")
print("="*70)
print(f"Total Accident images: {len(all_accident)}")
print(f"Total No-Accident images: {len(all_no_accident)}")
print(f"Total images: {len(all_accident) + len(all_no_accident)}")

# =========================================================
# STEP 2: SELECT BALANCED SUBSET
# =========================================================
print("\n" + "="*70)
print("🎯 STEP 2: Creating balanced subset")
print("="*70)

# Check if we have enough images
available_accident = len(all_accident)
available_no_accident = len(all_no_accident)

if available_accident < TARGET_ACCIDENT:
    print(f"⚠️ Only {available_accident} accident images available (need {TARGET_ACCIDENT})")
    TARGET_ACCIDENT_ACTUAL = available_accident
else:
    TARGET_ACCIDENT_ACTUAL = TARGET_ACCIDENT

if available_no_accident < TARGET_NO_ACCIDENT:
    print(f"⚠️ Only {available_no_accident} no-accident images available (need {TARGET_NO_ACCIDENT})")
    TARGET_NO_ACCIDENT_ACTUAL = available_no_accident
else:
    TARGET_NO_ACCIDENT_ACTUAL = TARGET_NO_ACCIDENT

# Select random samples
selected_accident = random.sample(all_accident, TARGET_ACCIDENT_ACTUAL)
selected_no_accident = random.sample(all_no_accident, TARGET_NO_ACCIDENT_ACTUAL)

print(f"\n✅ Selected:")
print(f"   Accident: {len(selected_accident)} images")
print(f"   No-Accident: {len(selected_no_accident)} images")
print(f"   Total: {len(selected_accident) + len(selected_no_accident)} images")

# =========================================================
# STEP 3: SPLIT INTO TRAIN/VAL/TEST
# =========================================================
print("\n" + "="*70)
print("📁 STEP 3: Splitting into Train/Val/Test")
print("="*70)

def split_images_with_labels(image_list):
    """Split images (with labels) into train/val/test"""
    random.shuffle(image_list)
    
    train_end = int(len(image_list) * TRAIN_RATIO)
    val_end = train_end + int(len(image_list) * VAL_RATIO)
    
    train = image_list[:train_end]
    val = image_list[train_end:val_end]
    test = image_list[val_end:]
    
    return train, val, test

# Split accident images
train_acc, val_acc, test_acc = split_images_with_labels(selected_accident)

# Split no-accident images
train_no, val_no, test_no = split_images_with_labels(selected_no_accident)

# Combine
train_images = train_acc + train_no
val_images = val_acc + val_no
test_images = test_acc + test_no

print(f"\n📊 Split Statistics:")
print(f"   TRAIN: {len(train_images)} images ({len(train_acc)} accident + {len(train_no)} no-accident)")
print(f"   VALID: {len(val_images)} images ({len(val_acc)} accident + {len(val_no)} no-accident)")
print(f"   TEST: {len(test_images)} images ({len(test_acc)} accident + {len(test_no)} no-accident)")

# =========================================================
# STEP 4: CREATE BALANCED DATASET DIRECTORY
# =========================================================
print("\n" + "="*70)
print("📂 STEP 4: Creating balanced dataset")
print("="*70)

# Delete existing directory if exists
if BALANCED_ROOT.exists():
    shutil.rmtree(BALANCED_ROOT)

# Create directory structure
for split in ['train', 'valid', 'test']:
    (BALANCED_ROOT / split / 'images').mkdir(parents=True, exist_ok=True)
    (BALANCED_ROOT / split / 'labels').mkdir(parents=True, exist_ok=True)

def copy_files_with_labels(image_list, dest_root, split_name):
    """Copy images and their corresponding labels"""
    
    copied = 0
    no_labels = 0
    
    for img_path, label_path in tqdm(image_list, desc=f"Copying {split_name}"):
        # Copy image
        dest_img_path = dest_root / split_name / 'images' / img_path.name
        shutil.copy2(img_path, dest_img_path)
        
        # Copy or create label
        dest_label_path = dest_root / split_name / 'labels' / f"{img_path.stem}.txt"
        
        if label_path and label_path.exists():
            shutil.copy2(label_path, dest_label_path)
            copied += 1
        else:
            # Create empty label for no-accident
            dest_label_path.touch()
            no_labels += 1
    
    return copied, no_labels

# Copy files
train_copied, train_no_labels = copy_files_with_labels(train_images, BALANCED_ROOT, 'train')
val_copied, val_no_labels = copy_files_with_labels(val_images, BALANCED_ROOT, 'valid')
test_copied, test_no_labels = copy_files_with_labels(test_images, BALANCED_ROOT, 'test')

print(f"\n✅ Dataset created at: {BALANCED_ROOT}")
print(f"   Train: {train_copied} with labels, {train_no_labels} no-accident")
print(f"   Valid: {val_copied} with labels, {val_no_labels} no-accident")
print(f"   Test: {test_copied} with labels, {test_no_labels} no-accident")

# =========================================================
# STEP 5: FIX NEGATIVE BOUNDING BOXES
# =========================================================
print("\n" + "="*70)
print("🔧 STEP 5: Fixing negative bounding boxes")
print("="*70)

def fix_negative_bboxes(label_path):
    """Fix negative width/height in a single label file"""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return False
    
    with open(label_path, 'r') as f:
        lines = f.read().strip().split('\n')
    
    modified = False
    new_lines = []
    
    for line in lines:
        if not line.strip():
            continue
        
        parts = line.split()
        if len(parts) == 5:
            class_id = parts[0]
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Fix negative values
            if width < 0:
                width = abs(width)
                modified = True
            if height < 0:
                height = abs(height)
                modified = True
            
            # Ensure valid range
            width = max(0.001, min(1.0, width))
            height = max(0.001, min(1.0, height))
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            
            new_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    if modified and new_lines:
        with open(label_path, 'w') as f:
            f.write('\n'.join(new_lines))
        return True
    
    return False

# Fix each split
for split in ['train', 'valid', 'test']:
    labels_dir = BALANCED_ROOT / split / 'labels'
    if labels_dir.exists():
        fixed_count = 0
        for label_file in labels_dir.glob("*.txt"):
            if fix_negative_bboxes(label_file):
                fixed_count += 1
        print(f"   {split}: Fixed {fixed_count} files")

# =========================================================
# STEP 6: CREATE data.yaml
# =========================================================
print("\n" + "="*70)
print("📝 STEP 6: Creating data.yaml")
print("="*70)

data_yaml = {
    'path': str(BALANCED_ROOT.absolute()),
    'train': 'train/images',
    'val': 'valid/images',
    'test': 'test/images',
    'nc': 2,
    'names': ['Accident', 'Non Accident']
}

with open(BALANCED_ROOT / 'data.yaml', 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print("data.yaml created:")
print(f"   path: {data_yaml['path']}")
print(f"   nc: {data_yaml['nc']}")
print(f"   names: {data_yaml['names']}")

# =========================================================
# STEP 7: VERIFY FINAL DATASET
# =========================================================
print("\n" + "="*70)
print("✅ STEP 7: Verifying final dataset")
print("="*70)

def verify_dataset(balanced_root):
    """Verify the balanced dataset"""
    
    stats = {}
    total_accident = 0
    total_no_accident = 0
    
    for split in ['train', 'valid', 'test']:
        images_dir = balanced_root / split / 'images'
        labels_dir = balanced_root / split / 'labels'
        
        if not images_dir.exists():
            continue
        
        image_count = len(list(images_dir.glob("*")))
        
        # Count accident vs no-accident
        accident_count = 0
        no_accident_count = 0
        
        for label_file in labels_dir.glob("*.txt"):
            if label_file.stat().st_size == 0:
                no_accident_count += 1
            else:
                with open(label_file, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        no_accident_count += 1
                    else:
                        has_accident = False
                        for line in content.split('\n'):
                            if line.strip():
                                parts = line.split()
                                if len(parts) >= 1:
                                    try:
                                        class_id = int(float(parts[0]))
                                        if class_id == 0:
                                            has_accident = True
                                            break
                                    except:
                                        pass
                        
                        if has_accident:
                            accident_count += 1
                        else:
                            no_accident_count += 1
        
        stats[split] = {
            'total': image_count,
            'accident': accident_count,
            'no_accident': no_accident_count
        }
        
        total_accident += accident_count
        total_no_accident += no_accident_count
    
    return stats, total_accident, total_no_accident

final_stats, final_acc, final_no_acc = verify_dataset(BALANCED_ROOT)

print("\n📊 FINAL BALANCED DATASET STATISTICS:")
print("="*70)

for split, data in final_stats.items():
    accident_pct = (data['accident'] / data['total'] * 100) if data['total'] > 0 else 0
    print(f"\n📁 {split.upper()}:")
    print(f"   Total: {data['total']} images")
    print(f"   Accident: {data['accident']} ({accident_pct:.1f}%)")
    print(f"   No-Accident: {data['no_accident']} ({100-accident_pct:.1f}%)")

print("\n" + "="*70)
print("🎯 FINAL TOTALS:")
print("="*70)
print(f"   Total Accident: {final_acc}")
print(f"   Total No-Accident: {final_no_acc}")
print(f"   Grand Total: {final_acc + final_no_acc}")

if final_acc > 0:
    balance_ratio = final_no_acc / final_acc
    print(f"   Balance Ratio: 1:{balance_ratio:.2f}")
    
    if abs(balance_ratio - 1.0) < 0.05:
        print("\n✅ PERFECT BALANCE! Accident ≈ No-Accident")
    else:
        print(f"\n⚠️ Not perfectly balanced (difference: {abs(final_acc - final_no_acc)} images)")
else:
    print("\n❌ ERROR: No accident images found in balanced dataset!")

# =========================================================
# FINAL SUMMARY
# =========================================================
print("\n" + "="*70)
print("🎉 DATASET CREATION COMPLETE!")
print("="*70)
print(f"""
📁 Dataset Location: {BALANCED_ROOT}
📄 Config File: {BALANCED_ROOT / 'data.yaml'}

📊 Dataset Statistics:
   • Total Images: {final_acc + final_no_acc}
   • Accident Images: {final_acc}
   • No-Accident Images: {final_no_acc}

📈 Split Distribution:
   • Train: {final_stats['train']['total']} images ({final_stats['train']['accident']} accident + {final_stats['train']['no_accident']} no-accident)
   • Valid: {final_stats['valid']['total']} images ({final_stats['valid']['accident']} accident + {final_stats['valid']['no_accident']} no-accident)
   • Test: {final_stats['test']['total']} images ({final_stats['test']['accident']} accident + {final_stats['test']['no_accident']} no-accident)

✅ Dataset is READY for YOLOv8 training!
""")

# Save info file
info_path = BALANCED_ROOT / "dataset_info.txt"
with open(info_path, 'w') as f:
    f.write("DATASET INFORMATION\n")
    f.write("="*50 + "\n\n")
    f.write(f"Total Images: {final_acc + final_no_acc}\n")
    f.write(f"Accident Images: {final_acc}\n")
    f.write(f"No-Accident Images: {final_no_acc}\n\n")
    f.write("Split Distribution:\n")
    for split, data in final_stats.items():
        f.write(f"  {split.upper()}: {data['total']} images ({data['accident']} accident, {data['no_accident']} no-accident)\n")

print(f"📄 Info saved to: {info_path}")