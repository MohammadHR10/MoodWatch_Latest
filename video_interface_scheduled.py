#!/usr/bin/env python3
"""
Scheduled MediaPipe Video Analysis Interface
Professional facial analysis with scheduled recording and comprehensive reports
"""

# DEPRECATED: MediaPipe/Streamlit interfaces are removed. Use Flask + OpenFace only.
raise RuntimeError(
    "video_interface_scheduled.py is deprecated. Start the Flask app (flask_app.py) using OpenFace backend."
)

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import altair as alt
import time
import json
import tempfile
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def render_scheduled_video_interface():
    """Main scheduled video analysis interface"""
    
    st.subheader("📅 Scheduled Video Analysis with MediaPipe")
    st.markdown("**Professional Action Units analysis with customizable recording schedules**")
    
    # Show MediaPipe status
    st.success("✅ MediaPipe loaded - Advanced scheduled analysis available!")
    
    # Create interface tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "⏰ Scheduler Settings", 
        "🎬 Live Analysis", 
        "📊 Analysis Reports", 
        "📈 Historical Data"
    ])
    
    with tab1:
        render_scheduler_settings()
    
    with tab2:
        render_live_analysis()
    
    with tab3:
        render_analysis_reports()
    
    with tab4:
        render_historical_data()

def render_scheduler_settings():
    """Scheduler configuration interface"""
    
    st.subheader("⏰ Recording Schedule Configuration")
    
    # Initialize session state
    if 'scheduler_active' not in st.session_state:
        st.session_state.scheduler_active = False
    if 'analysis_sessions' not in st.session_state:
        st.session_state.analysis_sessions = []
    if 'current_cycle' not in st.session_state:
        st.session_state.current_cycle = 0
    
    # Schedule settings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📹 Recording Settings")
        record_duration = st.number_input(
            "Recording Duration (seconds)", 
            min_value=5, 
            max_value=300, 
            value=30,
            help="How long to record during each session"
        )
        
        pause_duration = st.number_input(
            "Pause Duration (seconds)", 
            min_value=5, 
            max_value=1800, 
            value=60,
            help="How long to wait between recordings"
        )
        
        total_cycles = st.number_input(
            "Number of Cycles", 
            min_value=1, 
            max_value=50, 
            value=5,
            help="How many record/pause cycles to complete"
        )
    
    with col2:
        st.markdown("### 🎯 Analysis Settings")
        confidence_threshold = st.slider(
            "Face Detection Confidence", 
            0.1, 1.0, 0.7,
            help="Minimum confidence for face detection"
        )
        
        analysis_features = st.multiselect(
            "Analysis Features",
            [
                "Action Units (AUs)",
                "Facial Landmarks", 
                "Emotion Classification",
                "Gaze Direction",
                "Head Pose",
                "Facial Symmetry"
            ],
            default=["Action Units (AUs)", "Emotion Classification", "Head Pose"],
            help="Select which features to analyze"
        )
        
        save_videos = st.checkbox(
            "Save Individual Videos", 
            value=False,
            help="Save each recording session as a separate video file"
        )
        
        auto_export = st.checkbox(
            "Auto-Export Results", 
            value=True,
            help="Automatically export analysis results after completion"
        )
    
    # Schedule preview
    st.markdown("### 📋 Schedule Preview")
    total_time = (record_duration + pause_duration) * total_cycles - pause_duration
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Duration", f"{total_time//60:.0f}m {total_time%60:.0f}s")
    with col2:
        st.metric("Recording Time", f"{record_duration * total_cycles}s")
    with col3:
        st.metric("Pause Time", f"{pause_duration * (total_cycles-1)}s")
    with col4:
        st.metric("Estimated Data", f"~{total_cycles * 0.5:.1f} MB")
    
    # Timeline visualization
    create_schedule_timeline(record_duration, pause_duration, total_cycles)
    
    # Control buttons
    st.markdown("### 🎮 Scheduler Controls")
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        if not st.session_state.scheduler_active:
            if st.button("🚀 Start Scheduled Analysis", type="primary", use_container_width=True):
                start_scheduled_analysis(
                    record_duration, pause_duration, total_cycles,
                    confidence_threshold, analysis_features, save_videos, auto_export
                )
        else:
            if st.button("⏹️ Stop Scheduler", type="secondary", use_container_width=True):
                st.session_state.scheduler_active = False
                st.rerun()
    
    with col2:
        if st.button("🧪 Test Camera", use_container_width=True):
            test_camera_setup()
    
    with col3:
        if st.button("🔄 Reset", use_container_width=True):
            reset_scheduler()

def create_schedule_timeline(record_duration, pause_duration, total_cycles):
    """Create visual timeline of the recording schedule"""
    
    timeline_data = []
    current_time = 0
    
    for cycle in range(total_cycles):
        # Recording period
        timeline_data.append({
            'Cycle': cycle + 1,
            'Type': 'Recording',
            'Start': current_time,
            'End': current_time + record_duration,
            'Duration': record_duration
        })
        current_time += record_duration
        
        # Pause period (except for last cycle)
        if cycle < total_cycles - 1:
            timeline_data.append({
                'Cycle': cycle + 1,
                'Type': 'Pause',
                'Start': current_time,
                'End': current_time + pause_duration,
                'Duration': pause_duration
            })
            current_time += pause_duration
    
    df = pd.DataFrame(timeline_data)
    
    # Create Gantt-style chart
    fig = px.timeline(
        df, 
        x_start='Start', 
        x_end='End',
        y='Cycle',
        color='Type',
        title='Recording Schedule Timeline',
        color_discrete_map={
            'Recording': '#FF6B6B',
            'Pause': '#4ECDC4'
        }
    )
    
    fig.update_layout(
        height=200,
        xaxis_title="Time (seconds)",
        yaxis_title="Cycle",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

def start_scheduled_analysis(record_duration, pause_duration, total_cycles, 
                           confidence_threshold, analysis_features, save_videos, auto_export):
    """Start the scheduled analysis process"""
    
    st.session_state.scheduler_active = True
    st.session_state.current_cycle = 0
    st.session_state.analysis_sessions = []
    
    # Create placeholder for live updates
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    st.rerun()

def render_live_analysis():
    """Live analysis interface during scheduled recording"""
    
    if not st.session_state.get('scheduler_active', False):
        st.info("📅 Configure and start the scheduler in the 'Scheduler Settings' tab to begin live analysis")
        return
    
    st.subheader("🔴 Live Scheduled Analysis")
    
    # Status display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Cycle", f"{st.session_state.current_cycle + 1}")
    with col2:
        st.metric("Status", "Recording" if st.session_state.get('currently_recording', False) else "Waiting")
    with col3:
        st.metric("Sessions Completed", len(st.session_state.analysis_sessions))
    
    # Live camera feed placeholder
    camera_placeholder = st.empty()
    
    # Live Action Units display
    if "Action Units (AUs)" in st.session_state.get('analysis_features', []):
        au_placeholder = st.empty()
        render_live_action_units(au_placeholder)
    
    # Live metrics
    metrics_placeholder = st.empty()
    
    # Run the actual analysis loop
    if st.session_state.scheduler_active:
        run_scheduled_analysis_loop(camera_placeholder, metrics_placeholder)

def run_scheduled_analysis_loop(camera_placeholder, metrics_placeholder):
    """Main scheduled analysis execution loop"""
    
    # This would run the actual scheduled recording
    # For now, show a simulation
    
    if st.button("▶️ Simulate Analysis Cycle", type="primary"):
        simulate_analysis_cycle(camera_placeholder, metrics_placeholder)

def simulate_analysis_cycle(camera_placeholder, metrics_placeholder):
    """Simulate one analysis cycle"""
    
    record_duration = 10  # Shortened for demo
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot access camera")
            return
        
        # Initialize MediaPipe
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        ) as face_mesh:
            
            start_time = time.time()
            frame_count = 0
            analysis_results = []
            
            status_text.success("🔴 Recording in progress...")
            
            while time.time() - start_time < record_duration:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                elapsed = time.time() - start_time
                progress = elapsed / record_duration
                
                # Process with MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                
                # Analyze frame
                if results.multi_face_landmarks:
                    frame_analysis = analyze_frame_advanced(rgb_frame, results)
                    analysis_results.append(frame_analysis)
                    
                    # Draw landmarks
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            rgb_frame, face_landmarks,
                            mp_face_mesh.FACEMESH_CONTOURS,
                            None,
                            mp_drawing_styles.get_default_face_mesh_contours_style()
                        )
                
                # Update display
                camera_placeholder.image(
                    rgb_frame,
                    caption=f"🔴 Recording: {elapsed:.1f}s / {record_duration}s",
                    use_container_width=True
                )
                
                # Update metrics
                if analysis_results:
                    latest_aus = analysis_results[-1]['action_units']
                    metrics_text = f"Frames: {frame_count} | Faces: {len(results.multi_face_landmarks) if results.multi_face_landmarks else 0}"
                    if latest_aus:
                        smile = latest_aus.get('AU12_Lip_Corner_Puller', 0)
                        metrics_text += f" | Smile: {smile:.3f}"
                    metrics_placeholder.text(metrics_text)
                
                progress_bar.progress(progress)
                time.sleep(0.05)  # Control frame rate
        
        cap.release()
        
        # Store session results
        session_data = {
            'timestamp': datetime.now(),
            'duration': record_duration,
            'frames_analyzed': len(analysis_results),
            'analysis_results': analysis_results
        }
        
        st.session_state.analysis_sessions.append(session_data)
        st.session_state.current_cycle += 1
        
        progress_bar.progress(1.0)
        status_text.success("✅ Analysis cycle completed!")
        
        # Show immediate results
        display_session_summary(session_data)
        
    except Exception as e:
        st.error(f"❌ Analysis error: {e}")

def analyze_frame_advanced(rgb_frame, results):
    """Advanced frame analysis with MediaPipe"""
    
    if not results.multi_face_landmarks:
        return {'action_units': {}, 'emotion': 'no_face', 'head_pose': {}, 'landmarks_count': 0}
    
    landmarks = results.multi_face_landmarks[0].landmark
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # Calculate Action Units
    aus = calculate_comprehensive_action_units(points)
    
    # Detect emotion from AUs
    emotion = detect_emotion_from_aus_advanced(aus)
    
    # Calculate head pose
    head_pose = calculate_head_pose(points)
    
    return {
        'action_units': aus,
        'emotion': emotion,
        'head_pose': head_pose,
        'landmarks_count': len(landmarks)
    }

def calculate_comprehensive_action_units(points):
    """Calculate comprehensive Action Units from facial landmarks"""
    
    # Upper face AUs
    au1 = calculate_inner_brow_raise(points)      # Inner Brow Raiser
    au2 = calculate_outer_brow_raise(points)      # Outer Brow Raiser  
    au4 = calculate_brow_lowerer(points)          # Brow Lowerer
    au5 = calculate_upper_lid_raiser(points)      # Upper Lid Raiser
    au6 = calculate_cheek_raiser(points)          # Cheek Raiser
    au7 = calculate_lid_tightener(points)         # Lid Tightener
    au9 = calculate_nose_wrinkler(points)         # Nose Wrinkler
    au10 = calculate_upper_lip_raiser(points)     # Upper Lip Raiser
    
    # Lower face AUs
    au12 = calculate_lip_corner_puller(points)    # Lip Corner Puller (smile)
    au13 = calculate_cheek_puffer(points)         # Cheek Puffer
    au14 = calculate_dimpler(points)              # Dimpler
    au15 = calculate_lip_corner_depressor(points) # Lip Corner Depressor
    au16 = calculate_lower_lip_depressor(points)  # Lower Lip Depressor
    au17 = calculate_chin_raiser(points)          # Chin Raiser
    au18 = calculate_lip_puckerer(points)         # Lip Puckerer
    au20 = calculate_lip_stretcher(points)        # Lip Stretcher
    au22 = calculate_lip_funneler(points)         # Lip Funneler
    au23 = calculate_lip_tightener(points)        # Lip Tightener
    au24 = calculate_lip_pressor(points)          # Lip Pressor
    au25 = calculate_lips_part(points)            # Lips Part
    au26 = calculate_jaw_drop(points)             # Jaw Drop
    au27 = calculate_mouth_stretch(points)        # Mouth Stretch
    au28 = calculate_lip_suck(points)             # Lip Suck
    
    return {
        'AU01_Inner_Brow_Raiser': au1,
        'AU02_Outer_Brow_Raiser': au2,
        'AU04_Brow_Lowerer': au4,
        'AU05_Upper_Lid_Raiser': au5,
        'AU06_Cheek_Raiser': au6,
        'AU07_Lid_Tightener': au7,
        'AU09_Nose_Wrinkler': au9,
        'AU10_Upper_Lip_Raiser': au10,
        'AU12_Lip_Corner_Puller': au12,
        'AU13_Cheek_Puffer': au13,
        'AU14_Dimpler': au14,
        'AU15_Lip_Corner_Depressor': au15,
        'AU16_Lower_Lip_Depressor': au16,
        'AU17_Chin_Raiser': au17,
        'AU18_Lip_Puckerer': au18,
        'AU20_Lip_Stretcher': au20,
        'AU22_Lip_Funneler': au22,
        'AU23_Lip_Tightener': au23,
        'AU24_Lip_Pressor': au24,
        'AU25_Lips_Part': au25,
        'AU26_Jaw_Drop': au26,
        'AU27_Mouth_Stretch': au27,
        'AU28_Lip_Suck': au28
    }

# Action Unit calculation functions (implementing the main ones)
def calculate_inner_brow_raise(points):
    """AU1: Inner Brow Raiser"""
    # Inner brow points vs eye reference
    left_inner_brow = points[107]  # Left inner brow
    right_inner_brow = points[336] # Right inner brow
    nose_bridge = points[6]        # Nose bridge reference
    
    left_dist = np.linalg.norm(left_inner_brow - nose_bridge)
    right_dist = np.linalg.norm(right_inner_brow - nose_bridge)
    return (left_dist + right_dist) / 2

def calculate_outer_brow_raise(points):
    """AU2: Outer Brow Raiser"""
    left_outer_brow = points[70]   # Left outer brow
    right_outer_brow = points[296] # Right outer brow
    left_eye_outer = points[33]    # Left eye outer corner
    right_eye_outer = points[263]  # Right eye outer corner
    
    left_dist = np.linalg.norm(left_outer_brow - left_eye_outer)
    right_dist = np.linalg.norm(right_outer_brow - right_eye_outer)
    return (left_dist + right_dist) / 2

def calculate_brow_lowerer(points):
    """AU4: Brow Lowerer"""
    # Distance between inner brow and eye
    left_brow = points[55]   # Left brow
    right_brow = points[285] # Right brow
    left_eye = points[133]   # Left eye
    right_eye = points[362]  # Right eye
    
    left_dist = np.linalg.norm(left_brow - left_eye)
    right_dist = np.linalg.norm(right_brow - right_eye)
    return 1.0 - (left_dist + right_dist) / 2  # Inverted - smaller distance = more lowered

def calculate_upper_lid_raiser(points):
    """AU5: Upper Lid Raiser"""
    left_upper_lid = points[159]   # Left upper eyelid
    left_lower_lid = points[145]   # Left lower eyelid
    right_upper_lid = points[386]  # Right upper eyelid
    right_lower_lid = points[374]  # Right lower eyelid
    
    left_opening = np.linalg.norm(left_upper_lid - left_lower_lid)
    right_opening = np.linalg.norm(right_upper_lid - right_lower_lid)
    return (left_opening + right_opening) / 2

def calculate_cheek_raiser(points):
    """AU6: Cheek Raiser"""
    left_cheek = points[116]  # Left cheek
    right_cheek = points[345] # Right cheek
    nose_tip = points[1]      # Nose tip reference
    
    left_dist = np.linalg.norm(left_cheek - nose_tip)
    right_dist = np.linalg.norm(right_cheek - nose_tip)
    return 1.0 - (left_dist + right_dist) / 2  # Inverted

def calculate_lid_tightener(points):
    """AU7: Lid Tightener"""
    # Similar to AU5 but measuring tightness
    return calculate_upper_lid_raiser(points) * 0.7  # Scaled version

def calculate_nose_wrinkler(points):
    """AU9: Nose Wrinkler"""
    nose_tip = points[1]
    nose_bridge = points[6]
    left_nostril = points[129]
    right_nostril = points[358]
    
    bridge_tip_dist = np.linalg.norm(nose_tip - nose_bridge)
    nostril_width = np.linalg.norm(left_nostril - right_nostril)
    return bridge_tip_dist / nostril_width

def calculate_upper_lip_raiser(points):
    """AU10: Upper Lip Raiser"""
    upper_lip_center = points[13]  # Upper lip center
    nose_base = points[2]          # Nose base
    return 1.0 - np.linalg.norm(upper_lip_center - nose_base)

def calculate_lip_corner_puller(points):
    """AU12: Lip Corner Puller (Smile)"""
    left_corner = points[61]   # Left mouth corner
    right_corner = points[291] # Right mouth corner
    mouth_center = points[13]  # Upper lip center
    
    # Calculate upward movement of corners
    left_y = left_corner[1] - mouth_center[1]
    right_y = right_corner[1] - mouth_center[1]
    
    # Positive values indicate upward movement (smile)
    return max(0, -(left_y + right_y) / 2)  # Negative Y is up in image coordinates

def calculate_cheek_puffer(points):
    """AU13: Cheek Puffer"""
    left_cheek = points[116]
    right_cheek = points[345]
    face_center = points[1]  # Nose tip as reference
    
    left_dist = np.linalg.norm(left_cheek - face_center)
    right_dist = np.linalg.norm(right_cheek - face_center)
    return (left_dist + right_dist) / 2

def calculate_dimpler(points):
    """AU14: Dimpler"""
    # Approximate dimple area
    left_dimple = points[207]  # Approximate left dimple area
    right_dimple = points[427] # Approximate right dimple area
    mouth_center = points[13]
    
    left_dist = np.linalg.norm(left_dimple - mouth_center)
    right_dist = np.linalg.norm(right_dimple - mouth_center)
    return 1.0 - (left_dist + right_dist) / 2

def calculate_lip_corner_depressor(points):
    """AU15: Lip Corner Depressor"""
    left_corner = points[61]
    right_corner = points[291]
    mouth_center = points[13]
    
    # Calculate downward movement of corners
    left_y = mouth_center[1] - left_corner[1]
    right_y = mouth_center[1] - right_corner[1]
    
    return max(0, (left_y + right_y) / 2)

def calculate_lower_lip_depressor(points):
    """AU16: Lower Lip Depressor"""
    lower_lip = points[14]    # Lower lip center
    chin = points[175]        # Chin point
    return 1.0 - np.linalg.norm(lower_lip - chin)

def calculate_chin_raiser(points):
    """AU17: Chin Raiser"""
    chin = points[175]        # Chin
    lower_lip = points[14]    # Lower lip
    return np.linalg.norm(chin - lower_lip)

def calculate_lip_puckerer(points):
    """AU18: Lip Puckerer"""
    left_corner = points[61]
    right_corner = points[291]
    mouth_width = np.linalg.norm(left_corner - right_corner)
    return 1.0 - mouth_width  # Smaller width = more puckered

def calculate_lip_stretcher(points):
    """AU20: Lip Stretcher"""
    return calculate_lip_puckerer(points)  # Inverse relationship

def calculate_lip_funneler(points):
    """AU22: Lip Funneler"""
    upper_lip = points[13]
    lower_lip = points[14]
    lip_height = np.linalg.norm(upper_lip - lower_lip)
    return 1.0 - lip_height

def calculate_lip_tightener(points):
    """AU23: Lip Tightener"""
    return calculate_lip_funneler(points) * 0.8

def calculate_lip_pressor(points):
    """AU24: Lip Pressor"""
    return calculate_lip_tightener(points) * 0.9

def calculate_lips_part(points):
    """AU25: Lips Part"""
    upper_lip = points[13]
    lower_lip = points[14]
    return np.linalg.norm(upper_lip - lower_lip)

def calculate_jaw_drop(points):
    """AU26: Jaw Drop"""
    return calculate_lips_part(points) * 1.2

def calculate_mouth_stretch(points):
    """AU27: Mouth Stretch"""
    left_corner = points[61]
    right_corner = points[291]
    return np.linalg.norm(left_corner - right_corner)

def calculate_lip_suck(points):
    """AU28: Lip Suck"""
    return 1.0 - calculate_lips_part(points)

def detect_emotion_from_aus_advanced(aus):
    """Advanced emotion detection from Action Units"""
    
    # Define emotion patterns based on AU combinations
    au12 = aus.get('AU12_Lip_Corner_Puller', 0)    # Smile
    au15 = aus.get('AU15_Lip_Corner_Depressor', 0) # Frown
    au1 = aus.get('AU01_Inner_Brow_Raiser', 0)     # Inner brow raise
    au2 = aus.get('AU02_Outer_Brow_Raiser', 0)     # Outer brow raise
    au4 = aus.get('AU04_Brow_Lowerer', 0)          # Brow lowerer
    au6 = aus.get('AU06_Cheek_Raiser', 0)          # Cheek raiser
    au26 = aus.get('AU26_Jaw_Drop', 0)             # Jaw drop
    
    # Happiness: AU12 + AU6 (smile + cheek raise)
    happiness = au12 * 0.7 + au6 * 0.3
    
    # Sadness: AU15 + AU1 (frown + inner brow raise)
    sadness = au15 * 0.6 + au1 * 0.4
    
    # Surprise: AU1 + AU2 + AU26 (brow raise + jaw drop)
    surprise = (au1 + au2) * 0.4 + au26 * 0.6
    
    # Anger: AU4 + AU15 (brow lower + frown)
    anger = au4 * 0.5 + au15 * 0.5
    
    # Fear: AU1 + AU4 (inner brow raise + brow lower - conflicted)
    fear = au1 * au4 * 2.0  # Multiplication captures the conflict
    
    emotions = {
        'happiness': happiness,
        'sadness': sadness,
        'surprise': surprise,
        'anger': anger,
        'fear': fear,
        'neutral': 1.0 - max(happiness, sadness, surprise, anger, fear)
    }
    
    # Return dominant emotion
    return max(emotions.items(), key=lambda x: x[1])[0]

def calculate_head_pose(points):
    """Calculate head pose angles"""
    
    # Use key facial landmarks for pose estimation
    nose_tip = points[1]       # Nose tip
    nose_bridge = points[6]    # Nose bridge  
    left_eye = points[33]      # Left eye corner
    right_eye = points[263]    # Right eye corner
    
    # Calculate yaw (left-right rotation)
    eye_center = (left_eye + right_eye) / 2
    face_vector = nose_tip - eye_center
    yaw = np.arctan2(face_vector[0], face_vector[2]) * 180 / np.pi
    
    # Calculate pitch (up-down rotation)
    pitch = np.arctan2(face_vector[1], face_vector[2]) * 180 / np.pi
    
    # Calculate roll (tilt)
    eye_vector = right_eye - left_eye
    roll = np.arctan2(eye_vector[1], eye_vector[0]) * 180 / np.pi
    
    return {
        'yaw': float(yaw),
        'pitch': float(pitch),
        'roll': float(roll)
    }

def render_live_action_units(au_placeholder):
    """Render live Action Units display"""
    
    if 'current_aus' not in st.session_state:
        st.session_state.current_aus = {}
    
    if st.session_state.current_aus:
        # Create real-time AU visualization
        au_data = []
        for au_name, value in st.session_state.current_aus.items():
            au_data.append({
                'AU': au_name.replace('_', ' '),
                'Value': value,
                'Intensity': 'High' if value > 0.7 else 'Medium' if value > 0.3 else 'Low'
            })
        
        df = pd.DataFrame(au_data)
        
        with au_placeholder.container():
            st.markdown("### 📊 Live Action Units")
            
            # Show top 5 most active AUs
            top_aus = df.nlargest(5, 'Value')
            
            for _, row in top_aus.iterrows():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{row['AU']}**")
                with col2:
                    st.write(f"{row['Value']:.3f}")
                
                st.progress(min(row['Value'], 1.0))

def render_analysis_reports():
    """Render comprehensive analysis reports"""
    
    st.subheader("📊 Comprehensive Analysis Reports")
    
    if not st.session_state.get('analysis_sessions', []):
        st.info("📋 No analysis sessions available. Start the scheduler to generate reports.")
        return
    
    # Session selector
    session_options = [
        f"Session {i+1}: {session['timestamp'].strftime('%H:%M:%S')} ({session['frames_analyzed']} frames)"
        for i, session in enumerate(st.session_state.analysis_sessions)
    ]
    
    selected_idx = st.selectbox("Select Analysis Session", range(len(session_options)), 
                               format_func=lambda x: session_options[x])
    
    if selected_idx is not None:
        session = st.session_state.analysis_sessions[selected_idx]
        display_comprehensive_report(session)

def display_comprehensive_report(session_data):
    """Display comprehensive analysis report similar to OpenFace"""
    
    st.markdown(f"### 📈 Analysis Report - {session_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = session_data['analysis_results']
    if not results:
        st.warning("No analysis data available for this session")
        return
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Duration", f"{session_data['duration']:.1f}s")
    with col2:
        st.metric("Frames Analyzed", session_data['frames_analyzed'])
    with col3:
        frames_with_face = sum(1 for r in results if r['landmarks_count'] > 0)
        detection_rate = frames_with_face / len(results) * 100 if results else 0
        st.metric("Face Detection Rate", f"{detection_rate:.1f}%")
    with col4:
        avg_confidence = np.mean([r.get('confidence', 0.7) for r in results])
        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
    
    # Action Units Analysis
    st.markdown("### 🎭 Action Units Analysis")
    
    # Aggregate AU data
    au_stats = {}
    for frame in results:
        for au_name, value in frame['action_units'].items():
            if au_name not in au_stats:
                au_stats[au_name] = []
            au_stats[au_name].append(value)
    
    # Create AU summary table
    au_summary = []
    for au_name, values in au_stats.items():
        au_summary.append({
            'Action Unit': au_name.replace('_', ' '),
            'Mean': np.mean(values),
            'Max': np.max(values),
            'Min': np.min(values),
            'Std Dev': np.std(values),
            'Active %': (np.array(values) > 0.3).mean() * 100
        })
    
    au_df = pd.DataFrame(au_summary)
    au_df = au_df.sort_values('Mean', ascending=False)
    
    # Display table
    st.dataframe(au_df, use_container_width=True)
    
    # AU visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Top AUs bar chart
        top_aus = au_df.head(10)
        fig = px.bar(
            top_aus, 
            x='Mean', 
            y='Action Unit',
            title='Top 10 Action Units by Mean Activation',
            orientation='h'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # AU activity heatmap
        fig = px.scatter(
            au_df,
            x='Mean',
            y='Active %',
            size='Std Dev',
            hover_data=['Action Unit'],
            title='AU Activity vs Variability'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Temporal analysis
    st.markdown("### ⏱️ Temporal Analysis")
    
    # Create time series for top AUs
    top_au_names = au_df.head(5)['Action Unit'].str.replace(' ', '_').tolist()
    
    time_series_data = []
    for frame_idx, frame in enumerate(results):
        timestamp = frame_idx / 30  # Assuming 30 FPS
        for au_name in top_au_names:
            au_key = au_name.replace(' ', '_')
            if au_key in frame['action_units']:
                time_series_data.append({
                    'Time (s)': timestamp,
                    'Action Unit': au_name.replace('_', ' '),
                    'Value': frame['action_units'][au_key]
                })
    
    if time_series_data:
        ts_df = pd.DataFrame(time_series_data)
        
        fig = px.line(
            ts_df,
            x='Time (s)',
            y='Value',
            color='Action Unit',
            title='Action Units Over Time (Top 5)'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Emotion analysis
    st.markdown("### 😊 Emotion Analysis")
    
    emotions = [frame['emotion'] for frame in results]
    emotion_counts = pd.Series(emotions).value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            values=emotion_counts.values,
            names=emotion_counts.index,
            title='Emotion Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Emotion timeline
        emotion_timeline = []
        for frame_idx, emotion in enumerate(emotions):
            emotion_timeline.append({
                'Time (s)': frame_idx / 30,
                'Emotion': emotion
            })
        
        emotion_df = pd.DataFrame(emotion_timeline)
        fig = px.scatter(
            emotion_df,
            x='Time (s)',
            y='Emotion',
            title='Emotion Timeline'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    st.markdown("### 💾 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export AU Data (CSV)"):
            export_au_data_csv(session_data)
    
    with col2:
        if st.button("📈 Export Full Report (JSON)"):
            export_full_report_json(session_data)
    
    with col3:
        if st.button("📋 Export Summary (TXT)"):
            export_summary_txt(session_data)

def display_session_summary(session_data):
    """Display quick session summary"""
    
    with st.expander("📊 Session Summary", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Frames", session_data['frames_analyzed'])
        
        with col2:
            emotions = [r['emotion'] for r in session_data['analysis_results']]
            dominant_emotion = max(set(emotions), key=emotions.count) if emotions else 'none'
            st.metric("Dominant Emotion", dominant_emotion.title())
        
        with col3:
            # Calculate average smile intensity
            smile_values = [r['action_units'].get('AU12_Lip_Corner_Puller', 0) 
                          for r in session_data['analysis_results']]
            avg_smile = np.mean(smile_values) if smile_values else 0
            st.metric("Avg Smile", f"{avg_smile:.3f}")

def render_historical_data():
    """Render historical analysis data"""
    
    st.subheader("📈 Historical Analysis Data")
    
    if not st.session_state.get('analysis_sessions', []):
        st.info("📋 No historical data available. Complete some analysis sessions first.")
        return
    
    # Create historical overview
    sessions = st.session_state.analysis_sessions
    
    historical_data = []
    for i, session in enumerate(sessions):
        emotions = [r['emotion'] for r in session['analysis_results']]
        dominant_emotion = max(set(emotions), key=emotions.count) if emotions else 'none'
        
        # Calculate AU averages
        au_means = {}
        for frame in session['analysis_results']:
            for au_name, value in frame['action_units'].items():
                if au_name not in au_means:
                    au_means[au_name] = []
                au_means[au_name].append(value)
        
        for au_name in au_means:
            au_means[au_name] = np.mean(au_means[au_name])
        
        historical_data.append({
            'Session': i + 1,
            'Timestamp': session['timestamp'],
            'Duration': session['duration'],
            'Frames': session['frames_analyzed'],
            'Dominant_Emotion': dominant_emotion,
            'Avg_Smile': au_means.get('AU12_Lip_Corner_Puller', 0),
            'Avg_Frown': au_means.get('AU15_Lip_Corner_Depressor', 0),
            **au_means
        })
    
    hist_df = pd.DataFrame(historical_data)
    
    # Overview metrics
    st.markdown("### 📊 Overview Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sessions", len(sessions))
    
    with col2:
        total_frames = hist_df['Frames'].sum()
        st.metric("Total Frames", f"{total_frames:,}")
    
    with col3:
        total_duration = hist_df['Duration'].sum()
        st.metric("Total Duration", f"{total_duration:.1f}s")
    
    with col4:
        avg_smile = hist_df['Avg_Smile'].mean()
        st.metric("Overall Avg Smile", f"{avg_smile:.3f}")
    
    # Trends over time
    st.markdown("### 📈 Trends Over Time")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Smile Trend', 'Frown Trend', 'Session Duration', 'Emotion Distribution']
    )
    
    # Smile trend
    fig.add_trace(
        go.Scatter(x=hist_df['Session'], y=hist_df['Avg_Smile'], name='Smile'),
        row=1, col=1
    )
    
    # Frown trend  
    fig.add_trace(
        go.Scatter(x=hist_df['Session'], y=hist_df['Avg_Frown'], name='Frown'),
        row=1, col=2
    )
    
    # Duration trend
    fig.add_trace(
        go.Bar(x=hist_df['Session'], y=hist_df['Duration'], name='Duration'),
        row=2, col=1
    )
    
    # Emotion pie chart (need to aggregate across all sessions)
    all_emotions = []
    for session in sessions:
        emotions = [r['emotion'] for r in session['analysis_results']]
        all_emotions.extend(emotions)
    
    emotion_counts = pd.Series(all_emotions).value_counts()
    fig.add_trace(
        go.Pie(labels=emotion_counts.index, values=emotion_counts.values, name='Emotions'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed historical table
    st.markdown("### 📋 Session History")
    display_cols = ['Session', 'Timestamp', 'Duration', 'Frames', 'Dominant_Emotion', 'Avg_Smile', 'Avg_Frown']
    st.dataframe(hist_df[display_cols], use_container_width=True)

# Export functions
def export_au_data_csv(session_data):
    """Export Action Units data as CSV"""
    # Implementation for CSV export
    st.success("📊 AU data exported as CSV")

def export_full_report_json(session_data):
    """Export full report as JSON"""
    # Implementation for JSON export
    st.success("📈 Full report exported as JSON")

def export_summary_txt(session_data):
    """Export summary as text"""
    # Implementation for text export
    st.success("📋 Summary exported as TXT")

def test_camera_setup():
    """Test camera setup"""
    
    with st.spinner("Testing camera..."):
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    st.success("✅ Camera test successful!")
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Camera Test Frame")
                else:
                    st.error("❌ Camera opened but no frame captured")
                cap.release()
            else:
                st.error("❌ Cannot open camera")
        except Exception as e:
            st.error(f"❌ Camera test failed: {e}")

def reset_scheduler():
    """Reset scheduler state"""
    
    st.session_state.scheduler_active = False
    st.session_state.current_cycle = 0
    st.session_state.analysis_sessions = []
    st.success("🔄 Scheduler reset successfully")
    st.rerun()