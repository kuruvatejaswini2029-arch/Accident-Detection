"""
Accident Detection Module
Contains core detection logic with temporal smoothing
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from collections import deque
import time
import tempfile
from ultralytics import YOLO

class TemporalSmoother:
    """Temporal smoothing for accident detection to reduce false positives"""
    
    def __init__(self, window_size=8, threshold=4, confidence_smoothing=0.8, min_duration=12):
        self.window_size = window_size
        self.threshold = threshold
        self.confidence_smoothing = confidence_smoothing
        self.min_duration = min_duration
        self.detection_buffer = deque(maxlen=window_size)
        self.accident_state = False
        self.accident_counter = 0
        self.smoothed_confidence = 0.0
        
    def update(self, current_detections):
        """Update smoother with current frame detections"""
        current_accident = False
        current_confidence = 0.0
        
        for det in current_detections:
            if det['class_id'] == 0:  # Accident class
                current_accident = True
                current_confidence = max(current_confidence, det['confidence'])
        
        self.detection_buffer.append({
            'accident': current_accident,
            'confidence': current_confidence
        })
        
        accident_count = sum(1 for d in self.detection_buffer if d['accident'])
        avg_confidence = np.mean([d['confidence'] for d in self.detection_buffer if d['accident']]) if accident_count > 0 else 0
        
        self.smoothed_confidence = (self.smoothed_confidence * self.confidence_smoothing + 
                                   avg_confidence * (1 - self.confidence_smoothing))
        
        if accident_count >= self.threshold:
            self.accident_counter += 1
            if self.accident_counter >= self.min_duration:
                self.accident_state = True
        else:
            self.accident_counter = max(0, self.accident_counter - 2)
            if self.accident_counter == 0:
                self.accident_state = False
        
        return {
            'accident_detected': self.accident_state,
            'detection_count': accident_count,
            'smoothed_confidence': self.smoothed_confidence
        }
    
    def reset(self):
        """Reset smoother state"""
        self.detection_buffer.clear()
        self.accident_state = False
        self.accident_counter = 0
        self.smoothed_confidence = 0.0


class AccidentDetector:
    """Main accident detector class"""
    
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(model_path))
        self.model.to(self.device)
        self.confidence = 0.35
        self.enable_smoothing = True
        self.smoother = TemporalSmoother()
        
    def set_confidence(self, confidence):
        """Set confidence threshold"""
        self.confidence = confidence
        
    def set_smoothing(self, enable, window_size=8):
        """Enable/disable temporal smoothing"""
        self.enable_smoothing = enable
        if enable:
            self.smoother = TemporalSmoother(window_size=window_size)
        else:
            self.smoother = None
    
    def detect_image(self, image):
        """Detect accidents in a single image"""
        start_time = time.time()
        
        # Run inference
        results = self.model(image, conf=self.confidence, device=self.device, verbose=False)
        result = results[0]
        
        # Check for accident
        accident_detected = False
        max_confidence = 0.0
        
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                if class_id == 0:  # Accident class
                    accident_detected = True
                    max_confidence = max(max_confidence, confidence)
        
        # Get annotated image
        annotated = result.plot()
        
        processing_time = time.time() - start_time
        
        return {
            'accident_detected': accident_detected,
            'confidence': max_confidence,
            'annotated': annotated,
            'processing_time': processing_time,
            'result': result
        }
    
    def detect_frame(self, frame):
        """Detect accidents in a single frame (for video)"""
        results = self.model(frame, conf=self.confidence, device=self.device, verbose=False)
        result = results[0]
        
        # Extract detections
        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                detections.append({
                    'class_id': int(box.cls),
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist()
                })
        
        # Apply smoothing if enabled
        if self.enable_smoothing and self.smoother:
            smoothed = self.smoother.update(detections)
            accident_detected = smoothed['accident_detected']
            confidence = smoothed['smoothed_confidence']
        else:
            accident_detected = any(d['class_id'] == 0 for d in detections)
            confidence = max([d['confidence'] for d in detections if d['class_id'] == 0], default=0.0)
        
        # Get annotated frame
        annotated = result.plot()
        
        return {
            'accident_detected': accident_detected,
            'confidence': confidence,
            'annotated': annotated,
            'detections': detections
        }


def process_video_stream(video_path, detector, progress_callback=None, status_callback=None):
    """Process video stream with progress tracking"""
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Reset smoother
    if detector.smoother:
        detector.smoother.reset()
    
    # Output video
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))
    
    frame_count = 0
    accident_frames = []
    timeline_events = []
    last_event_end = 0
    
    start_time = time.time()
    
    # Process frames
    results = detector.model(
        video_path,
        stream=True,
        conf=detector.confidence,
        device=detector.device,
        verbose=False
    )
    
    for result in results:
        frame_count += 1
        
        # Update progress
        if progress_callback:
            progress_callback(int(frame_count / total_frames * 100))
        if status_callback:
            status_callback(f"Processing frame {frame_count}/{total_frames}")
        
        # Process frame
        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                detections.append({
                    'class_id': int(box.cls),
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist()
                })
        
        # Apply smoothing
        if detector.enable_smoothing and detector.smoother:
            smoothed = detector.smoother.update(detections)
            accident_detected = smoothed['accident_detected']
            current_confidence = smoothed['smoothed_confidence']
        else:
            accident_detected = any(d['class_id'] == 0 for d in detections)
            current_confidence = max([d['confidence'] for d in detections if d['class_id'] == 0], default=0.0)
        
        if accident_detected:
            accident_frames.append(frame_count)
            
            # Record timeline event
            if frame_count > last_event_end + 10:
                timeline_events.append({
                    'id': len(timeline_events) + 1,
                    'start_frame': frame_count,
                    'end_frame': frame_count,
                    'start_time': frame_count / fps,
                    'end_time': frame_count / fps,
                    'confidence': current_confidence
                })
                last_event_end = frame_count
            else:
                timeline_events[-1]['end_frame'] = frame_count
                timeline_events[-1]['end_time'] = frame_count / fps
                timeline_events[-1]['confidence'] = max(timeline_events[-1]['confidence'], current_confidence)
        
        # Annotate frame
        annotated = result.plot()
        
        # Add status text
        if accident_detected:
            cv2.rectangle(annotated, (0, 0), (width-1, height-1), (0, 0, 255), 3)
            cv2.putText(annotated, "ACCIDENT DETECTED!", (width//2-150, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Write frame
        out.write(annotated)
    
    cap.release()
    out.release()
    
    processing_time = time.time() - start_time
    
    # Calculate event durations
    for event in timeline_events:
        event['duration'] = event['end_time'] - event['start_time']
    
    return {
        'total_frames': total_frames,
        'accident_frames': len(accident_frames),
        'accident_events': len(timeline_events),
        'timeline_events': timeline_events,
        'output_video': temp_output.name,
        'processing_time': processing_time
    }
