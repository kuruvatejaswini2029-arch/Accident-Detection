"""
Accident Detection Web Application
Streamlit-based UI for accident detection
"""

import streamlit as st
import sys
import os

# Fix for headless OpenCV
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

# Import OpenCV with fallback
try:
    import cv2
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2

import torch
import tempfile
from pathlib import Path
import time
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import gc

# Import detection functions
from detection import AccidentDetector, process_video_stream

# Page configuration
st.set_page_config(
    page_title="Accident Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #ff4444;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .accident-alert {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        50% { opacity: 0.5; }
    }
    .stButton > button {
        background-color: #ff4444;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("⚙️ Configuration")

# Model loading
st.sidebar.subheader("Model Settings")

# Check if model exists in different locations
model_paths = [
    "models/best.pt",
    "best.pt",
    "../models/best.pt",
    "../best.pt",
    "Accident-Detection/models/best.pt"
]

model_path = None
for path in model_paths:
    if os.path.exists(path):
        model_path = path
        break

if model_path is None:
    st.sidebar.warning("⚠️ Model not found. Please ensure best.pt is in the models/ folder")
    model_path = st.sidebar.text_input(
        "Model Path",
        value="models/best.pt",
        help="Path to the trained YOLOv8 model"
    )

if st.sidebar.button("Load Model", type="primary"):
    with st.spinner("Loading model..."):
        try:
            st.session_state.detector = AccidentDetector(model_path)
            st.sidebar.success("✅ Model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)[:100]}")

# Detection settings
st.sidebar.subheader("Detection Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    step=0.05,
    help="Higher values reduce false positives"
)

# Smoothing settings
st.sidebar.subheader("Smoothing Settings")
enable_smoothing = st.sidebar.checkbox("Enable Temporal Smoothing", value=True)
smoothing_window = st.sidebar.slider(
    "Smoothing Window",
    min_value=1,
    max_value=15,
    value=8,
    step=1,
    disabled=not enable_smoothing,
    help="Number of frames to consider for smoothing"
)

# =========================================================
# MAIN CONTENT
# =========================================================
st.markdown('<div class="main-header">🚗 Accident Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time accident detection using YOLOv8 with Temporal Smoothing</div>', unsafe_allow_html=True)

# Check if model is loaded
if st.session_state.detector is None:
    st.info("ℹ️ Please load the model from the sidebar to start detection")
    st.stop()

# Update detector settings
st.session_state.detector.set_confidence(confidence_threshold)
st.session_state.detector.set_smoothing(enable_smoothing, smoothing_window)

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(["📷 Image Detection", "🎥 Video Detection", "📊 Analytics", "ℹ️ About"])

# =========================================================
# TAB 1: IMAGE DETECTION
# =========================================================
with tab1:
    st.header("📷 Single Image Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image to detect accidents"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 Detect Accident", type="primary"):
                with st.spinner("Detecting..."):
                    # Convert PIL to numpy array
                    image_np = np.array(image)
                    
                    # Convert RGB to BGR for OpenCV
                    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    
                    # Run detection
                    result = st.session_state.detector.detect_image(image_np)
                    
                    # Display result
                    with col2:
                        # Convert BGR back to RGB for display
                        annotated_rgb = cv2.cvtColor(result['annotated'], cv2.COLOR_BGR2RGB)
                        st.image(annotated_rgb, caption="Detection Result", use_container_width=True)
                        
                        if result['accident_detected']:
                            st.markdown(
                                f'<div class="accident-alert">🚨 ACCIDENT DETECTED! (Confidence: {result["confidence"]:.2%})</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.success("✅ No accident detected")
                        
                        # Display detection details
                        st.write("### Detection Details")
                        st.write(f"**Accident Detected:** {'Yes' if result['accident_detected'] else 'No'}")
                        st.write(f"**Confidence:** {result['confidence']:.2%}")
                        st.write(f"**Processing Time:** {result['processing_time']:.3f} seconds")
                        
                        # Add to history
                        st.session_state.detection_history.append({
                            'timestamp': datetime.now(),
                            'type': 'image',
                            'accident': result['accident_detected'],
                            'confidence': result['confidence']
                        })

# =========================================================
# TAB 2: VIDEO DETECTION
# =========================================================
with tab2:
    st.header("🎥 Video Detection")
    
    video_file = st.file_uploader(
        "Upload a video",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video for accident detection"
    )
    
    if video_file is not None:
        # Save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        video_path = tfile.name
        
        st.video(video_path)
        
        col1, col2 = st.columns(2)
        
        with col1:
            process_video = st.button("🎬 Process Video", type="primary")
            
        with col2:
            save_results = st.checkbox("Save processed video", value=True)
        
        if process_video:
            with st.spinner("Processing video... This may take a few minutes"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process video
                result = process_video_stream(
                    video_path,
                    st.session_state.detector,
                    progress_callback=lambda p: progress_bar.progress(p),
                    status_callback=lambda s: status_text.text(s)
                )
                
                progress_bar.progress(100)
                status_text.text("Processing complete!")
                
                # Display results
                if result['accident_frames'] > 0:
                    st.success(f"✅ ACCIDENT DETECTED! Found {result['accident_frames']} frames with accidents")
                else:
                    st.info(f"ℹ️ No accidents detected in this video")
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Frames", result['total_frames'])
                with col2:
                    st.metric("Accident Frames", result['accident_frames'])
                with col3:
                    st.metric("Accident Events", result['accident_events'])
                with col4:
                    st.metric("Processing Time", f"{result['processing_time']:.1f}s")
                
                # Display timeline
                if result['timeline_events']:
                    st.subheader("📅 Accident Timeline")
                    
                    timeline_data = []
                    for event in result['timeline_events']:
                        timeline_data.append({
                            'Event': event['id'],
                            'Start (sec)': f"{event['start_time']:.1f}",
                            'End (sec)': f"{event['end_time']:.1f}",
                            'Duration (sec)': f"{event['duration']:.1f}",
                            'Confidence': f"{event['confidence']:.2%}"
                        })
                    
                    st.dataframe(pd.DataFrame(timeline_data), use_container_width=True)
                    
                    # Add to history
                    st.session_state.detection_history.append({
                        'timestamp': datetime.now(),
                        'type': 'video',
                        'accident': True,
                        'accident_frames': result['accident_frames'],
                        'duration': result['processing_time']
                    })
                
                # Download processed video
                if save_results and result.get('output_video') and os.path.exists(result['output_video']):
                    with open(result['output_video'], 'rb') as f:
                        st.download_button(
                            label="📥 Download Processed Video",
                            data=f,
                            file_name=f"processed_{Path(video_file.name).stem}.mp4",
                            mime="video/mp4"
                        )
                
                # Cleanup temp file
                try:
                    os.unlink(video_path)
                except:
                    pass

# =========================================================
# TAB 3: ANALYTICS
# =========================================================
with tab3:
    st.header("📊 Detection Analytics")
    
    if st.session_state.detection_history:
        # Convert history to DataFrame
        df = pd.DataFrame(st.session_state.detection_history)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Detections", len(df))
        with col2:
            accident_count = df[df['accident'] == True].shape[0] if 'accident' in df else 0
            st.metric("Accidents Detected", accident_count)
        with col3:
            if 'confidence' in df:
                valid_conf = df[df['confidence'].notna() & (df['confidence'] > 0)]
                avg_conf = valid_conf['confidence'].mean() if len(valid_conf) > 0 else 0
                st.metric("Avg Confidence", f"{avg_conf:.2%}")
        
        # Detection timeline
        if 'timestamp' in df and 'confidence' in df:
            st.subheader("Detection Timeline")
            valid_df = df[df['confidence'].notna() & (df['confidence'] > 0)]
            if len(valid_df) > 0:
                fig = px.line(valid_df, x='timestamp', y='confidence', 
                              title="Detection Confidence Over Time")
                st.plotly_chart(fig, use_container_width=True)
        
        # Detection type distribution
        if 'type' in df:
            st.subheader("Detection Type Distribution")
            type_counts = df['type'].value_counts()
            fig = px.pie(values=type_counts.values, names=type_counts.index, 
                         title="Image vs Video Detections")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No detection history yet. Run some detections to see analytics!")

# =========================================================
# TAB 4: ABOUT
# =========================================================
with tab4:
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ### 🚗 Accident Detection System
    
    This system uses a custom-trained YOLOv8 model to detect road accidents in images and videos with **temporal smoothing** to reduce false positives.
    
    #### Key Features:
    - **Real-time accident detection** with adjustable confidence threshold
    - **Temporal smoothing** to reduce flickering and false positives
    - **Support for images and videos**
    - **Detailed analytics and timeline generation**
    - **Exportable results**
    
    #### Model Performance:
    | Metric | Value |
    |--------|-------|
    | Precision | 85.9% |
    | Recall | 72.1% |
    | mAP50 | 78.3% |
    | Inference Speed | 33.9ms/frame |
    
    #### How to Use:
    1. Load the trained model from the sidebar
    2. Upload an image or video
    3. Adjust detection settings as needed
    4. Click detect and view results
    
    #### Technologies Used:
    - **YOLOv8** - State-of-the-art object detection
    - **Streamlit** - Interactive web framework
    - **OpenCV** - Image and video processing
    - **PyTorch** - Deep learning framework
    - **Plotly** - Interactive visualizations
    
    ---
    **Created for Accident Detection Research | IEEE Paper Ready**
    """)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Accident Detection System | Powered by YOLOv8 | Temporal Smoothing Enabled</div>",
    unsafe_allow_html=True
)

# Cleanup on exit
def cleanup():
    if st.session_state.detector:
        del st.session_state.detector
    gc.collect()
