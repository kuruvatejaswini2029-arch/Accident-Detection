"""
ACCIDENT DETECTION ON SINGLE IMAGE
Run inference on any image file
"""

from ultralytics import YOLO
import cv2
import torch
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURATION
# =========================================================
MODEL_PATH = Path(r"C:\Users\TEJASWINI\runs\detect\Accident_Detection_Final\YOLOv8_Accident_Production\weights\best.pt")
IMAGE_PATH = input("Enter image path: ").strip().strip('"')
OUTPUT_DIR = Path(r"C:\Users\TEJASWINI\Downloads\accident_detection_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Detection parameters
CONF_THRESHOLD = 0.25  # Confidence threshold (lower = more detections)

print("="*60)
print("ACCIDENT DETECTION ON IMAGE")
print("="*60)

# =========================================================
# CHECK IF IMAGE EXISTS
# =========================================================
image_path = Path(IMAGE_PATH)
if not image_path.exists():
    print(f"ERROR: Image not found at {IMAGE_PATH}")
    print("Please provide a valid image path")
    exit(1)

# =========================================================
# LOAD MODEL
# =========================================================
print(f"\nLoading model from: {MODEL_PATH}")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(str(MODEL_PATH))
model.to(device)
print(f"Model loaded on: {device.upper()}")

# =========================================================
# RUN DETECTION
# =========================================================
print(f"\nProcessing image: {image_path.name}")

# Run inference
results = model(
    str(image_path),
    conf=CONF_THRESHOLD,
    device=device,
    verbose=True
)

# Get results
result = results[0]

# =========================================================
# ANALYZE RESULTS
# =========================================================
print("\n" + "="*60)
print("DETECTION RESULTS")
print("="*60)

# Count detections by class
accident_count = 0
non_accident_count = 0

if result.boxes is not None:
    for box in result.boxes:
        class_id = int(box.cls)
        confidence = float(box.conf)
        
        if class_id == 0:
            accident_count += 1
            print(f"\n[ACCIDENT DETECTED]")
            print(f"   Confidence: {confidence:.2%}")
            print(f"   Bounding Box: {box.xyxy[0].tolist()}")
        else:
            non_accident_count += 1

if accident_count == 0:
    print("\n[NO ACCIDENT] No accident detected in this image")

print(f"\nSummary:")
print(f"   Accidents detected: {accident_count}")
print(f"   Other objects detected: {non_accident_count}")

# =========================================================
# SAVE RESULT IMAGE
# =========================================================
# Get annotated image
annotated_image = result.plot()

# Save with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = OUTPUT_DIR / f"detection_{image_path.stem}_{timestamp}{image_path.suffix}"
cv2.imwrite(str(output_path), annotated_image)
print(f"\nAnnotated image saved to: {output_path}")

# =========================================================
# SAVE DETAILED REPORT
# =========================================================
report_path = OUTPUT_DIR / f"report_{image_path.stem}_{timestamp}.txt"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("ACCIDENT DETECTION REPORT\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"Image: {image_path.name}\n")
    f.write(f"Detection Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: YOLOv8 Accident Detection\n")
    f.write(f"Confidence Threshold: {CONF_THRESHOLD}\n\n")
    
    f.write("Detection Results:\n")
    f.write("-"*40 + "\n")
    
    if result.boxes is not None:
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls)
            confidence = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            class_name = "ACCIDENT" if class_id == 0 else "Non Accident"
            f.write(f"\nDetection {i+1}:\n")
            f.write(f"   Class: {class_name}\n")
            f.write(f"   Confidence: {confidence:.2%}\n")
            f.write(f"   Location: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]\n")
    else:
        f.write("\nNo detections found\n")
    
    f.write(f"\nSummary:\n")
    f.write(f"   Accidents Detected: {accident_count}\n")
    f.write(f"   Total Detections: {len(result.boxes) if result.boxes else 0}\n")

print(f"Report saved to: {report_path}")

# =========================================================
# DISPLAY RESULT
# =========================================================
print("\n" + "="*60)
print("DISPLAYING RESULT")
print("="*60)

# Convert BGR to RGB for matplotlib display
image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Original image
original = cv2.imread(str(image_path))
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
ax1.imshow(original_rgb)
ax1.set_title(f'Original Image\n{image_path.name}', fontsize=12)
ax1.axis('off')

# Annotated image
ax2.imshow(image_rgb)
ax2.set_title(f'Accident Detection Result\n{"ACCIDENT DETECTED!" if accident_count > 0 else "No Accident Detected"}', fontsize=12)
ax2.axis('off')

# Add confidence text if accident detected
if accident_count > 0 and result.boxes is not None:
    for box in result.boxes:
        if int(box.cls) == 0:
            conf = float(box.conf)
            ax2.text(10, 30, f'Accident: {conf:.1%}', 
                    fontsize=14, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            break

plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"visualization_{image_path.stem}_{timestamp}.png", dpi=150, bbox_inches='tight')
plt.show()

# =========================================================
# FINAL SUMMARY
# =========================================================
print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print(f"""
OUTPUT FILES:
   - Annotated Image: {output_path}
   - Detection Report: {report_path}
   - Visualization: {OUTPUT_DIR / f'visualization_{image_path.stem}_{timestamp}.png'}

RESULT: {'ACCIDENT DETECTED!' if accident_count > 0 else 'No accident detected'}

To view the annotated image, open:
   {output_path}
""")
print("="*60)

# =========================================================
# OPTIONAL: OPEN THE ANNOTATED IMAGE
# =========================================================
print("\nDo you want to open the annotated image? (y/n)")
choice = input().strip().lower()
if choice == 'y':
    import subprocess
    subprocess.run(['start', str(output_path)], shell=True)