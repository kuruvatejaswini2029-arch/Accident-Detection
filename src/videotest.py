"""
ACCIDENT DETECTION ON VIDEO WITH SMOOTHING (OPTIMIZED FOR CPU)
Includes temporal smoothing to reduce flickering and false positives
"""

from ultralytics import YOLO
import cv2
import torch
from pathlib import Path
from datetime import datetime
import numpy as np
from collections import deque
import gc

# =========================================================
# CONFIGURATION
# =========================================================
MODEL_PATH = Path(r"C:\Users\TEJASWINI\runs\detect\Accident_Detection_Final\YOLOv8_Accident_Production\weights\best.pt")
VIDEO_PATH = r"C:\Users\TEJASWINI\Downloads\Video Project 1.mp4"
OUTPUT_DIR = Path(r"C:\Users\TEJASWINI\Downloads\accident_detection_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Detection parameters (OPTIMIZED)
CONF_THRESHOLD = 0.35  # Higher threshold for cleaner detections
IOU_THRESHOLD = 0.45

# =========================================================
# SMOOTHING PARAMETERS (STRONGER FOR STABILITY)
# =========================================================
SMOOTHING_WINDOW = 8      # Increased from 5
ACCIDENT_THRESHOLD = 4     # Increased from 3
CONFIDENCE_SMOOTHING = 0.8 # Increased from 0.7
MIN_ACCIDENT_DURATION = 12 # Increased from 10

print("="*60)
print("ACCIDENT DETECTION ON VIDEO (OPTIMIZED CPU + SMOOTHING)")
print("="*60)

# =========================================================
# SAFETY CHECKS
# =========================================================
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not Path(VIDEO_PATH).exists():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

# =========================================================
# LOAD MODEL
# =========================================================
print(f"\nLoading model from: {MODEL_PATH}")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(str(MODEL_PATH))
model.to(device)
print(f"Model loaded on: {device.upper()}")

# =========================================================
# SMOOTHING CLASS
# =========================================================
class TemporalSmoother:
    """Temporal smoothing for accident detection"""
    
    def __init__(self, window_size=8, confidence_threshold=0.35):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.detection_buffer = deque(maxlen=window_size)
        self.accident_state = False
        self.accident_counter = 0
        self.smoothed_confidence = 0.0
        
    def update(self, current_detections):
        """Update smoother with current frame detections"""
        
        # Check if accident detected in current frame
        current_accident = False
        current_confidence = 0.0
        
        for det in current_detections:
            if det['class_id'] == 0:  # Accident class
                current_accident = True
                current_confidence = max(current_confidence, det['confidence'])
        
        # Add to buffer
        self.detection_buffer.append({
            'accident': current_accident,
            'confidence': current_confidence
        })
        
        # Calculate smoothed detection
        accident_count = sum(1 for d in self.detection_buffer if d['accident'])
        avg_confidence = np.mean([d['confidence'] for d in self.detection_buffer if d['accident']]) if accident_count > 0 else 0
        
        # Apply confidence smoothing
        self.smoothed_confidence = (self.smoothed_confidence * CONFIDENCE_SMOOTHING + 
                                   avg_confidence * (1 - CONFIDENCE_SMOOTHING))
        
        # Determine if accident is confirmed
        if accident_count >= ACCIDENT_THRESHOLD:
            self.accident_counter += 1
            if self.accident_counter >= MIN_ACCIDENT_DURATION:
                self.accident_state = True
        else:
            self.accident_counter = max(0, self.accident_counter - 2)
            if self.accident_counter == 0:
                self.accident_state = False
        
        return {
            'accident_detected': self.accident_state,
            'raw_detections': accident_count,
            'smoothed_confidence': self.smoothed_confidence,
            'detection_count': accident_count
        }
    
    def reset(self):
        """Reset smoother state"""
        self.detection_buffer.clear()
        self.accident_state = False
        self.accident_counter = 0
        self.smoothed_confidence = 0.0

# =========================================================
# RUN INFERENCE AND SAVE OUTPUT VIDEO
# =========================================================
print(f"\nProcessing video: {VIDEO_PATH}")

# Output video paths
output_video_path = OUTPUT_DIR / f"accident_detection_smoothed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
output_raw_path = OUTPUT_DIR / f"accident_detection_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

# Process video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

# Safety check for FPS
if fps <= 0 or fps is None:
    fps = 30
    print(f"   Warning: FPS not detected, using default: {fps}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Video writers (using better codec)
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Better compression
out_smoothed = cv2.VideoWriter(str(output_video_path), fourcc, int(fps), (width, height))
out_raw = cv2.VideoWriter(str(output_raw_path), fourcc, int(fps), (width, height))

print(f"\nVideo Info:")
print(f"   Frames: {total_frames}")
print(f"   FPS: {fps}")
print(f"   Resolution: {width}x{height}")
print(f"   Smoothing Window: {SMOOTHING_WINDOW} frames")
print(f"   Accident Threshold: {ACCIDENT_THRESHOLD} detections")
print(f"   Confidence Threshold: {CONF_THRESHOLD}")
print(f"\nProcessing frames...")

# Initialize smoother
smoother = TemporalSmoother(window_size=SMOOTHING_WINDOW)

frame_count = 0
accident_frames = []
accident_detections = []
raw_accident_frames = []
timeline_events = []

# Run inference with streaming (OPTIMIZED FOR CPU)
results = model(
    VIDEO_PATH,
    stream=True,
    conf=CONF_THRESHOLD,
    iou=IOU_THRESHOLD,
    imgsz=640,
    half=False,
    device=device,
    verbose=False
)

for result in results:
    frame_count += 1
    
    # Plot once (performance optimization)
    annotated = result.plot()
    
    # Extract raw detections
    current_detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            bbox = box.xyxy[0].tolist()
            current_detections.append({
                'class_id': class_id,
                'confidence': confidence,
                'bbox': bbox
            })
    
    # Apply temporal smoothing
    smoothed_result = smoother.update(current_detections)
    
    # Create frames (copy once to avoid double plotting)
    raw_frame = annotated.copy()
    smoothed_frame = annotated.copy()
    
    # Add smoothing info overlay
    info_text = [
        f"Raw Detections: {smoothed_result['detection_count']}",
        f"Accident Confirmed: {'YES' if smoothed_result['accident_detected'] else 'NO'}",
        f"Confidence: {smoothed_result['smoothed_confidence']:.2f}"
    ]
    
    for i, text in enumerate(info_text):
        cv2.putText(smoothed_frame, text, (10, 30 + i*25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Add status indicator
    if smoothed_result['accident_detected']:
        # Red border for accident
        cv2.rectangle(smoothed_frame, (0, 0), (width-1, height-1), (0, 0, 255), 5)
        cv2.putText(smoothed_frame, "ACCIDENT DETECTED!", (width//2-150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        accident_frames.append(frame_count)
        accident_detections.append({
            'frame': frame_count,
            'smoothed_confidence': smoothed_result['smoothed_confidence'],
            'raw_confidence': smoothed_result['detection_count']
        })
        
        # Record timeline event
        if len(timeline_events) == 0 or timeline_events[-1]['end'] < frame_count - 10:
            timeline_events.append({
                'start': frame_count,
                'end': frame_count,
                'confidence': smoothed_result['smoothed_confidence']
            })
        else:
            timeline_events[-1]['end'] = frame_count
            timeline_events[-1]['confidence'] = max(timeline_events[-1]['confidence'], 
                                                    smoothed_result['smoothed_confidence'])
    
    # Track raw detections
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            if int(box.cls) == 0:
                raw_accident_frames.append(frame_count)
                break
    
    # Write frames
    out_smoothed.write(smoothed_frame)
    out_raw.write(raw_frame)
    
    # Progress update
    if frame_count % 100 == 0 or frame_count == total_frames:
        progress = (frame_count / total_frames) * 100
        status = "ACCIDENT" if smoothed_result['accident_detected'] else "SAFE"
        print(f"   Progress: {progress:.1f}% ({frame_count}/{total_frames}) - Status: {status}")

cap.release()
out_smoothed.release()
out_raw.release()

print(f"\nOutput videos saved to:")
print(f"   Smoothed: {output_video_path}")
print(f"   Raw: {output_raw_path}")

# =========================================================
# DETAILED ANALYSIS WITH SMOOTHING
# =========================================================
print("\n" + "="*60)
print("DETAILED ACCIDENT ANALYSIS (WITH SMOOTHING)")
print("="*60)

if accident_frames:
    print(f"\n[ACCIDENT CONFIRMED] in {len(accident_frames)} frames!")
    print(f"Raw accident frames: {len(raw_accident_frames)}")
    print(f"False positives filtered: {len(raw_accident_frames) - len(accident_frames)}")
    
    # Extract key accident frames
    cap = cv2.VideoCapture(VIDEO_PATH)
    frames_dir = OUTPUT_DIR / "accident_frames_smoothed"
    frames_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving key accident frames to: {frames_dir}")
    
    for idx, event in enumerate(timeline_events):
        start = event['start']
        end = event['end']
        mid_frame = (start + end) // 2
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        if ret:
            frame_path = frames_dir / f"accident_event_{idx+1}_frame_{mid_frame}.jpg"
            cv2.imwrite(str(frame_path), frame)
            print(f"   Saved: {frame_path.name}")
    
    cap.release()
    
    # Find highest confidence detection
    if accident_detections:
        best_detection = max(accident_detections, key=lambda x: x['smoothed_confidence'])
        print(f"\nHighest confidence detection:")
        print(f"   Frame: {best_detection['frame']}")
        print(f"   Confidence: {best_detection['smoothed_confidence']:.3f}")
        
else:
    print(f"\n[NO ACCIDENT] No accidents confirmed in this video")
    print(f"Raw detections found: {len(raw_accident_frames)} (filtered out)")

# =========================================================
# GENERATE ACCIDENT TIMELINE
# =========================================================
print("\n" + "="*60)
print("ACCIDENT TIMELINE (SMOOTHED)")
print("="*60)

if timeline_events:
    print(f"\nAccident events detected:")
    for idx, event in enumerate(timeline_events):
        start_time = event['start'] / fps
        end_time = event['end'] / fps
        duration = (event['end'] - event['start']) / fps
        
        start_min = int(start_time // 60)
        start_sec = int(start_time % 60)
        end_min = int(end_time // 60)
        end_sec = int(end_time % 60)
        
        print(f"   Event {idx+1}: {start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d} (duration: {duration:.1f}s)")
    
    # Generate timeline text file
    timeline_path = OUTPUT_DIR / "accident_timeline_smoothed.txt"
    with open(timeline_path, 'w', encoding='utf-8') as f:
        f.write("ACCIDENT DETECTION TIMELINE (WITH SMOOTHING)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Video: {Path(VIDEO_PATH).name}\n")
        f.write(f"Total Frames: {total_frames}\n")
        f.write(f"Raw Detections: {len(raw_accident_frames)}\n")
        f.write(f"Confirmed Detections: {len(accident_frames)}\n")
        f.write(f"False Positives Filtered: {len(raw_accident_frames) - len(accident_frames)}\n\n")
        f.write("Accident Events:\n")
        
        for idx, event in enumerate(timeline_events):
            start_time = event['start'] / fps
            end_time = event['end'] / fps
            start_min = int(start_time // 60)
            start_sec = int(start_time % 60)
            end_min = int(end_time // 60)
            end_sec = int(end_time % 60)
            duration = (event['end'] - event['start']) / fps
            f.write(f"  Event {idx+1}: {start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d} (duration: {duration:.1f}s)\n")
    
    print(f"\nTimeline saved to: {timeline_path}")

# =========================================================
# CREATE SUMMARY REPORT
# =========================================================
print("\n" + "="*60)
print("SUMMARY REPORT")
print("="*60)

report_path = OUTPUT_DIR / "detection_report_smoothed.txt"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("ACCIDENT DETECTION REPORT (WITH TEMPORAL SMOOTHING)\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"Video File: {Path(VIDEO_PATH).name}\n")
    f.write(f"Detection Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: YOLOv8 Accident Detection\n\n")
    
    f.write("Smoothing Parameters:\n")
    f.write("-"*40 + "\n")
    f.write(f"  Window Size: {SMOOTHING_WINDOW} frames\n")
    f.write(f"  Threshold: {ACCIDENT_THRESHOLD} detections\n")
    f.write(f"  Confidence Smoothing: {CONFIDENCE_SMOOTHING}\n")
    f.write(f"  Min Duration: {MIN_ACCIDENT_DURATION} frames\n\n")
    
    f.write("Detection Results:\n")
    f.write("-"*40 + "\n")
    if timeline_events:
        f.write(f"RESULT: ACCIDENT CONFIRMED!\n")
        f.write(f"   Raw detections: {len(raw_accident_frames)}\n")
        f.write(f"   Confirmed detections: {len(accident_frames)}\n")
        f.write(f"   False positives filtered: {len(raw_accident_frames) - len(accident_frames)}\n")
        f.write(f"   Unique accident events: {len(timeline_events)}\n\n")
        
        f.write("Accident Events:\n")
        for idx, event in enumerate(timeline_events):
            start_time = event['start'] / fps
            end_time = event['end'] / fps
            start_min = int(start_time // 60)
            start_sec = int(start_time % 60)
            end_min = int(end_time // 60)
            end_sec = int(end_time % 60)
            duration = (event['end'] - event['start']) / fps
            f.write(f"   Event {idx+1}: {start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d} (duration: {duration:.1f}s)\n")
    else:
        f.write(f"RESULT: No accident confirmed\n")
        if raw_accident_frames:
            f.write(f"   Raw detections: {len(raw_accident_frames)} (filtered out by smoothing)\n")
    
    f.write("\nOutput Files:\n")
    f.write("-"*40 + "\n")
    f.write(f"  Smoothed Video: {output_video_path}\n")
    f.write(f"  Raw Video: {output_raw_path}\n")
    f.write(f"  Accident Frames: {frames_dir if timeline_events else 'None'}\n")
    f.write(f"  Timeline: {timeline_path if timeline_events else 'None'}\n")

print(f"\nReport saved to: {report_path}")

# =========================================================
# FINAL SUMMARY
# =========================================================
print("\n" + "="*60)
print("VIDEO ANALYSIS COMPLETE!")
print("="*60)
print(f"""
OUTPUT FILES:
   - Smoothed Video (Recommended): {output_video_path}
   - Raw Video (No smoothing): {output_raw_path}
   - Detection Report: {report_path}
   - Accident Frames: {frames_dir if timeline_events else 'No accidents detected'}
   - Timeline: {timeline_path if timeline_events else 'No accidents detected'}

{'RESULT: ACCIDENT DETECTED AND CONFIRMED!' if timeline_events else 'RESULT: No accident confirmed after smoothing'}

Detection Statistics:
   - Total Frames: {total_frames}
   - Raw Detections: {len(raw_accident_frames)}
   - Confirmed Detections: {len(accident_frames)}
   - False Positives Filtered: {len(raw_accident_frames) - len(accident_frames)}
   - Unique Accident Events: {len(timeline_events)}

Smoothing Benefits:
   - Reduced flickering
   - Filtered false positives
   - More stable detection
""")

print("="*60)

# =========================================================
# CLEANUP
# =========================================================
del model
gc.collect()

# =========================================================
# OPTIONAL: PLAY THE OUTPUT VIDEO
# =========================================================
print("\nDo you want to play the SMOOTHED annotated video? (y/n)")
choice = input().strip().lower()
if choice == 'y':
    import subprocess
    subprocess.run(['start', str(output_video_path)], shell=True)