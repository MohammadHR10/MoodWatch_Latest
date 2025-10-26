"""
Simple Video Interface - Basic OpenCV functionality with MediaPipe integration
Provides basic facial analysis as fallback when MediaPipe interface fails
"""

# DEPRECATED: Streamlit video interfaces have been removed. Use Flask + OpenFace.
raise RuntimeError(
    "video_interface_clean.py is deprecated and no longer used. Run flask_app.py and OpenFace backend."
)

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

def render_simple_video_interface():
    """Simple video interface with basic OpenCV"""
    
    st.subheader("🎥 Basic Video Analysis")
    st.markdown("Facial detection using OpenCV - fallback mode")
    st.info("💡 For advanced Action Units analysis, use the MediaPipe interface")
    
    # Recording interface
    col1, col2 = st.columns(2)
    with col1:
        duration = st.slider("Duration (seconds)", 3, 15, 6)
        show_preview = st.checkbox("📹 Show live preview", value=True)
    
    with col2:
        width = st.selectbox("Width", [640, 800, 1280], index=0)
        height = st.selectbox("Height", [480, 600, 720], index=0)
    
    if st.button("🎥 Start Recording", type="primary"):
        video_path = record_simple_video(duration, width, height, show_preview)
        
        if video_path:
            st.success("✅ Video recorded successfully!")
            
            # Offer download
            with open(video_path, 'rb') as f:
                st.download_button(
                    "📁 Download Video",
                    f.read(),
                    file_name=f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                    mime="video/mp4"
                )
            
            # Analyze video
            st.markdown("---")
            st.info("🔍 Running basic facial analysis...")
            analyze_simple_video(video_path)

def record_simple_video(duration: float, width: int, height: int, show_preview: bool) -> Optional[Path]:
    """Record video with basic OpenCV"""
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot access camera")
            return None
        
        # Set properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(tempfile.gettempdir()) / f"basic_video_{timestamp}.mp4"
        
        # Video writer
        fps = 20
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            st.error("❌ Failed to initialize video writer")
            cap.release()
            return None
        
        frames_to_record = int(fps * duration)
        
        # Setup UI elements
        if show_preview:
            preview_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Countdown
        if show_preview:
            status_text.info("🔴 Get ready! Starting in 3 seconds...")
            for i in range(3, 0, -1):
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    preview_placeholder.image(frame_rgb, caption=f"Starting in {i}...", width="stretch")
                time.sleep(1)
        
        status_text.success("🎬 Recording NOW!")
        
        # Record frames
        for i in range(frames_to_record):
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            
            # Update preview
            if show_preview and i % 5 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Add recording indicator
                cv2.circle(frame_rgb, (30, 30), 10, (255, 0, 0), -1)
                
                current_time = (i + 1) / fps
                remaining = duration - current_time
                
                preview_placeholder.image(
                    frame_rgb,
                    caption=f"🔴 Recording: {current_time:.1f}s / {duration:.1f}s (Remaining: {remaining:.1f}s)",
                    width="stretch"
                )
            
            progress_bar.progress((i + 1) / frames_to_record)
        
        # Cleanup
        cap.release()
        out.release()
        
        if show_preview:
            preview_placeholder.empty()
        progress_bar.empty()
        status_text.success("✅ Recording completed!")
        
        return output_path if output_path.exists() else None
        
    except Exception as e:
        st.error(f"❌ Recording failed: {str(e)}")
        return None

def analyze_simple_video(video_path: Path):
    """Basic video analysis with OpenCV"""
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            st.error("Cannot open video file")
            return
        
        # Face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        frame_count = 0
        faces_detected = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        progress_bar = st.progress(0, text="Analyzing video...")
        
        face_data = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                faces_detected += 1
                face_data.append({
                    'frame': frame_count,
                    'faces_count': len(faces),
                    'timestamp': frame_count / 20.0
                })
            
            frame_count += 1
            if frame_count % 10 == 0:
                progress_bar.progress(frame_count / total_frames)
        
        cap.release()
        progress_bar.empty()
        
        # Display results
        st.subheader("📊 Basic Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Frames", frame_count)
        with col2:
            st.metric("Frames with Faces", faces_detected)
        with col3:
            detection_rate = faces_detected / frame_count if frame_count > 0 else 0
            st.metric("Detection Rate", f"{detection_rate:.1%}")
        
        if face_data:
            # Create timeline
            df = pd.DataFrame(face_data)
            
            import altair as alt
            
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('timestamp:Q', title='Time (seconds)'),
                y=alt.Y('faces_count:Q', title='Number of Faces'),
                tooltip=['timestamp', 'faces_count']
            ).properties(height=200, title='Face Detection Timeline')
            
            st.altair_chart(chart, use_container_width=True)
        
        # Simple mood estimation
        if detection_rate > 0.7:
            st.success("🎭 **Estimated Mood**: Engaged (High face detection rate)")
        elif detection_rate > 0.4:
            st.info("🎭 **Estimated Mood**: Neutral (Moderate detection)")
        else:
            st.warning("🎭 **Estimated Mood**: Distracted (Low detection)")
            
        # Show upgrade suggestion
        st.info("💡 **Upgrade to MediaPipe** for advanced features: Action Units, emotion classification, and 468 facial landmarks!")
        
    except Exception as e:
        st.error(f"Analysis failed: {e}")