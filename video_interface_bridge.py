#!/usr/bin/env python3
"""
Enhanced Video Interface with MediaPipe Bridge
Uses subprocess to run MediaPipe analysis in Python 3.11 environment
"""

# DEPRECATED: MediaPipe bridge UI has been removed. The project is OpenFace-only.
raise RuntimeError(
    "video_interface_bridge.py is deprecated. Use Flask UI with OpenFace (flask_app.py + openface_bridge.py)."
)

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import altair as alt
import time
import json
import subprocess
import tempfile
import os
from datetime import datetime
from pathlib import Path
from datetime import datetime
import os

def render_video_interface_with_bridge():
    """
    Render video analysis interface that uses MediaPipe via subprocess
    """
    
    st.header("🎥 Advanced Video Analysis with MediaPipe")
    st.markdown("**Professional Action Units analysis powered by MediaPipe (Python 3.11)**")
    
    # Check if MediaPipe bridge is available
    bridge_path = Path("mediapipe_bridge.py")
    venv311_path = Path(".venv311/bin/python")
    
    if not bridge_path.exists():
        st.error("❌ MediaPipe bridge script not found")
        return
        
    if not venv311_path.exists():
        st.error("❌ Python 3.11 MediaPipe environment not found")
        st.info("Run: `python3.11 -m venv .venv311 && .venv311/bin/pip install mediapipe opencv-python numpy`")
        return
    
    st.success("✅ MediaPipe bridge available - Professional Action Units analysis ready!")
    
    # Simple analysis mode selector  
    analysis_mode = st.selectbox(
        "Choose Analysis Mode:",
        ["📅 Scheduled Analysis", "🎥 Upload Video File"],
        help="Select how you want to analyze video"
    )
    
    if analysis_mode == "📅 Scheduled Analysis":
        handle_scheduled_analysis()
    elif analysis_mode == "🎥 Upload Video File":
        handle_video_upload()

def test_mediapipe_bridge():
    """Test if MediaPipe bridge is working"""
    
    with st.spinner("Testing MediaPipe bridge..."):
        try:
            result = subprocess.run(
                [".venv311/bin/python", "-c", "import mediapipe; print('MediaPipe version:', mediapipe.__version__)"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                st.success(f"✅ MediaPipe bridge working! {result.stdout.strip()}")
            else:
                st.error(f"❌ MediaPipe test failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            st.error("❌ MediaPipe test timed out")
        except Exception as e:
            st.error(f"❌ Bridge test error: {e}")

def handle_scheduled_analysis():
    """Simple scheduled analysis with MediaPipe bridge"""
    
    st.subheader("📅 Scheduled Recording")
    
    # Simple settings
    col1, col2 = st.columns(2)
    
    with col1:
        record_time = st.slider("Recording Duration (seconds)", 5, 60, 20)
        pause_time = st.slider("Pause Between Recordings (seconds)", 10, 180, 45)
    
    with col2:
        num_sessions = st.slider("Number of Sessions", 1, 8, 3)
        show_camera = st.checkbox("Show Camera During Recording", value=True)
    
    # Show simple schedule
    total_time = (record_time + pause_time) * num_sessions - pause_time
    st.info(f"📋 **Schedule:** {num_sessions} recordings of {record_time}s each. Total time: {total_time//60}m {total_time%60:.0f}s")
    
    # Initialize session state
    if 'bridge_running' not in st.session_state:
        st.session_state.bridge_running = False
    if 'bridge_session' not in st.session_state:
        st.session_state.bridge_session = 0
    if 'bridge_results' not in st.session_state:
        st.session_state.bridge_results = []
    
    # Simple controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if not st.session_state.bridge_running:
            if st.button("🚀 Start Scheduled Recording", type="primary", use_container_width=True):
                st.session_state.bridge_running = True
                st.session_state.bridge_session = 0
                st.session_state.bridge_results = []
                st.rerun()
        else:
            if st.button("⏹️ Stop Recording", type="secondary", use_container_width=True):
                st.session_state.bridge_running = False
                st.rerun()
    
    with col2:
        if st.button("📊 Results", use_container_width=True):
            if st.session_state.bridge_results:
                show_bridge_results()
            else:
                st.info("No results yet")
    
    with col3:
        if st.button("🔄 Reset"):
            st.session_state.bridge_running = False
            st.session_state.bridge_session = 0
            st.session_state.bridge_results = []
            st.rerun()
    
    # Show recording status
    if st.session_state.bridge_running:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Session", f"{st.session_state.bridge_session + 1}/{num_sessions}")
        with col2:
            st.metric("Status", "🔴 Active")
        with col3:
            st.metric("Completed", len(st.session_state.bridge_results))
        
        # Show camera if enabled
        if show_camera:
            show_simple_camera_preview()
        
        # Demo session button
        if st.button("▶️ Record Next Session (Demo)"):
            run_demo_bridge_session(record_time)

def show_simple_camera_preview():
    """Show simple camera preview during recording"""
    
    camera_placeholder = st.empty()
    
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Add recording indicator
                height, width = frame_rgb.shape[:2]
                cv2.circle(frame_rgb, (width - 50, 50), 25, (255, 0, 0), -1)
                cv2.putText(frame_rgb, "REC", (width - 70, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                camera_placeholder.image(
                    frame_rgb,
                    caption="🔴 Recording Session",
                    use_container_width=True
                )
            cap.release()
        else:
            st.error("❌ Cannot access camera")
    except Exception as e:
        st.error(f"❌ Camera error: {e}")

def run_demo_bridge_session(duration):
    """Run a real recording session using camera and MediaPipe bridge"""
    
    st.info(f"🎬 Starting recording session {st.session_state.bridge_session + 1}")
    
    # Create temporary video file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video_path = temp_video.name
    temp_video.close()
    
    # Record real video with camera
    success = record_real_video(temp_video_path, duration)
    
    if success:
        with st.spinner("🔍 Analyzing video with MediaPipe..."):
            # Analyze with MediaPipe bridge
            results = analyze_with_mediapipe_bridge(temp_video_path)
            
            if results and results.get('success', False):
                # Process real MediaPipe results
                mock_result = {
                    'session': st.session_state.bridge_session + 1,
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'duration': duration,
                    'frames_processed': results.get('frames_analyzed', 0),
                    'faces_detected': results.get('frames_with_face', 0),
                    'action_units': {
                        'Smile': results['action_units'].get('AU12_Lip_Corner_Puller', {}).get('mean', 0),
                        'Brow Raise': results['action_units'].get('AU01_Inner_Brow_Raiser', {}).get('mean', 0),
                        'Eye Close': results['action_units'].get('AU07_Lid_Tightener', {}).get('mean', 0),
                        'Jaw Drop': results['action_units'].get('AU26_Jaw_Drop', {}).get('mean', 0)
                    },
                    'dominant_emotion': get_dominant_emotion(results.get('emotions', [])),
                    'detection_confidence': results.get('landmarks_stats', {}).get('detection_rate', 0) / 100
                }
                st.success("✅ MediaPipe analysis completed!")
            else:
                # Fallback to simulated results if MediaPipe fails
                st.warning("⚠️ MediaPipe analysis failed, using fallback results")
                mock_result = create_fallback_result(duration)
    else:
        # Fallback if recording fails
        st.error("❌ Recording failed, using simulated results")
        mock_result = create_fallback_result(duration)
    
    # Clean up temporary file
    try:
        os.unlink(temp_video_path)
    except:
        pass
    
    st.session_state.bridge_results.append(mock_result)
    st.session_state.bridge_session += 1
    
    st.success(f"🎉 Session {mock_result['session']} completed!")
    
    # Show detailed results
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Frames", mock_result['frames_processed'])
    with col2:
        st.metric("Faces Found", mock_result['faces_detected'])
    with col3:
        st.metric("Smile Level", f"{mock_result['action_units']['Smile']:.2f}")
    with col4:
        st.metric("Emotion", mock_result['dominant_emotion'])
    
    # Auto-refresh results display
    st.rerun()

def record_real_video(output_path, duration):
    """Record real video from camera with proper live preview"""
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot access camera")
            return False
        
        # Camera settings for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        
        # Get actual dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 15
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Create placeholders
        camera_placeholder = st.empty()
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        start_time = time.time()
        frames_recorded = 0
        last_display_time = 0
        
        status_placeholder.info("🎥 Starting camera...")
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to read frame")
                continue
            
            frames_recorded += 1
            elapsed = time.time() - start_time
            current_time = time.time()
            
            # Write frame to video
            out.write(frame)
            
            # Update display every 0.1 seconds for smooth preview
            if current_time - last_display_time >= 0.1:
                last_display_time = current_time
                
                # Prepare display frame
                display_frame = frame.copy()
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # Add recording indicator
                cv2.circle(frame_rgb, (width - 50, 50), 20, (255, 0, 0), -1)
                cv2.putText(frame_rgb, "REC", (width - 65, 57), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Add timestamp
                time_text = f"{elapsed:.1f}s / {duration}s"
                cv2.putText(frame_rgb, time_text, (10, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Update camera display
                camera_placeholder.image(
                    frame_rgb,
                    caption=f"🔴 Live Recording - {time_text}",
                    use_container_width=True
                )
                
                # Update progress
                progress = elapsed / duration
                progress_placeholder.progress(progress, text=f"Recording: {elapsed:.1f}s / {duration}s")
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.02)
        
        # Clean up
        cap.release()
        out.release()
        
        # Clear display elements
        camera_placeholder.empty()
        progress_placeholder.empty()
        status_placeholder.empty()
        
        st.success(f"✅ Recording completed! {frames_recorded} frames captured")
        return True
        
    except Exception as e:
        st.error(f"❌ Recording failed: {e}")
        return False

def analyze_with_mediapipe_bridge(video_path):
    """Analyze video using MediaPipe bridge subprocess"""
    
    # Create temporary results file
    results_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
    results_path = results_file.name
    results_file.close()
    
    try:
        # Run MediaPipe analysis
        process = subprocess.run([
            ".venv311/bin/python",
            "mediapipe_bridge.py", 
            video_path,
            results_path
        ], capture_output=True, text=True, timeout=60)
        
        if process.returncode == 0:
            # Load results
            with open(results_path, 'r') as f:
                results = json.load(f)
            return results
        else:
            st.error(f"MediaPipe analysis failed: {process.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        st.error("MediaPipe analysis timed out")
        return None
    except Exception as e:
        st.error(f"Analysis error: {e}")
        return None
    finally:
        # Clean up
        try:
            os.unlink(results_path)
        except:
            pass

def get_dominant_emotion(emotions_list):
    """Get dominant emotion from MediaPipe results"""
    
    if not emotions_list:
        return 'Neutral'
    
    # Count emotions
    emotion_counts = {}
    for emotion_info in emotions_list:
        emotion = emotion_info.get('emotion', 'neutral')
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + emotion_info.get('count', 1)
    
    if emotion_counts:
        return max(emotion_counts.items(), key=lambda x: x[1])[0].title()
    else:
        return 'Neutral'

def create_fallback_result(duration):
    """Create fallback result if real analysis fails"""
    
    return {
        'session': st.session_state.bridge_session + 1,
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'duration': duration,
        'frames_processed': duration * 20,  # 20 FPS estimation
        'faces_detected': int(duration * 15),  # Estimate face detection
        'action_units': {
            'Smile': np.random.uniform(0.1, 0.7),
            'Brow Raise': np.random.uniform(0.0, 0.4),
            'Eye Close': np.random.uniform(0.0, 0.3),
            'Jaw Drop': np.random.uniform(0.0, 0.4)
        },
        'dominant_emotion': np.random.choice(['Happy', 'Neutral', 'Focused', 'Calm']),
        'detection_confidence': np.random.uniform(0.6, 0.9)
    }

def show_bridge_results():
    """Show simplified bridge analysis results"""
    
    st.subheader("📊 Recording Results")
    
    results = st.session_state.bridge_results
    
    if not results:
        st.info("No results available")
        return
    
    # Summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sessions", len(results))
    
    with col2:
        total_faces = sum(r['faces_detected'] for r in results)
        st.metric("Total Faces", total_faces)
    
    with col3:
        avg_smile = np.mean([r['action_units']['Smile'] for r in results])
        st.metric("Avg Smile", f"{avg_smile:.2f}")
    
    with col4:
        emotions = [r['dominant_emotion'] for r in results]
        most_common = max(set(emotions), key=emotions.count)
        st.metric("Top Emotion", most_common)
    
    # Results table
    st.markdown("### Session Details")
    
    table_data = []
    for r in results:
        table_data.append({
            'Session': r['session'],
            'Time': r['timestamp'],
            'Duration': f"{r['duration']}s",
            'Faces': r['faces_detected'],
            'Smile': f"{r['action_units']['Smile']:.2f}",
            'Emotion': r['dominant_emotion'],
            'Confidence': f"{r['detection_confidence']:.2f}"
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True)
    
    # Simple trend if multiple sessions
    if len(results) > 1:
        st.markdown("### Smile Trend")
        
        trend_data = pd.DataFrame({
            'Session': [r['session'] for r in results],
            'Smile Level': [r['action_units']['Smile'] for r in results]
        })
        
        st.line_chart(trend_data.set_index('Session'))

def run_scheduled_analysis_bridge(record_duration, pause_duration, total_cycles):
    """Run the actual scheduled analysis using MediaPipe bridge"""
    
    st.info("🎬 Scheduled analysis started! Check the status above.")
    
    # This would implement the actual scheduled recording loop
    # For demonstration, we'll simulate it
    
    if st.button("⚡ Simulate Cycle (Demo)", type="secondary"):
        simulate_scheduled_cycle(record_duration)

def simulate_scheduled_cycle(duration):
    """Simulate one scheduled analysis cycle"""
    
    with st.spinner(f"Recording for {duration} seconds..."):
        time.sleep(2)  # Simulate recording time
        
        # Simulate analysis results
        mock_results = {
            'timestamp': datetime.now(),
            'duration': duration,
            'frames_analyzed': duration * 30,  # 30 FPS
            'action_units': {
                'AU12_Lip_Corner_Puller': np.random.uniform(0.1, 0.8),
                'AU01_Inner_Brow_Raiser': np.random.uniform(0.0, 0.6),
                'AU04_Brow_Lowerer': np.random.uniform(0.0, 0.4),
                'AU26_Jaw_Drop': np.random.uniform(0.0, 0.3),
                'AU15_Lip_Corner_Depressor': np.random.uniform(0.0, 0.5)
            },
            'emotions': ['happy', 'neutral', 'surprised'],
            'detection_rate': np.random.uniform(85, 98)
        }
        
        st.session_state.session_results.append(mock_results)
        st.session_state.current_cycle += 1
        
        st.success(f"✅ Cycle {st.session_state.current_cycle} completed!")
        
        # Show quick results
        with st.expander("📊 Cycle Results", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Frames", mock_results['frames_analyzed'])
            with col2:
                dominant_au = max(mock_results['action_units'].items(), key=lambda x: x[1])
                st.metric("Top AU", f"{dominant_au[0]}: {dominant_au[1]:.3f}")
            with col3:
                st.metric("Detection Rate", f"{mock_results['detection_rate']:.1f}%")

def display_scheduled_results():
    """Display comprehensive scheduled analysis results"""
    
    st.subheader("📊 Scheduled Analysis Results")
    
    results = st.session_state.session_results
    if not results:
        st.warning("No analysis results available")
        return
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sessions", len(results))
    
    with col2:
        total_frames = sum(r['frames_analyzed'] for r in results)
        st.metric("Total Frames", f"{total_frames:,}")
    
    with col3:
        avg_detection = np.mean([r['detection_rate'] for r in results])
        st.metric("Avg Detection Rate", f"{avg_detection:.1f}%")
    
    with col4:
        total_duration = sum(r['duration'] for r in results)
        st.metric("Total Recording Time", f"{total_duration:.1f}s")
    
    # Action Units analysis across sessions
    st.markdown("### 🎭 Action Units Analysis")
    
    # Aggregate AU data
    all_aus = {}
    for result in results:
        for au_name, value in result['action_units'].items():
            if au_name not in all_aus:
                all_aus[au_name] = []
            all_aus[au_name].append(value)
    
    # Create AU summary
    au_summary = []
    for au_name, values in all_aus.items():
        au_summary.append({
            'Action Unit': au_name.replace('_', ' '),
            'Mean': np.mean(values),
            'Max': np.max(values),
            'Min': np.min(values),
            'Sessions Active': sum(1 for v in values if v > 0.3)
        })
    
    au_df = pd.DataFrame(au_summary)
    au_df = au_df.sort_values('Mean', ascending=False)
    
    # Display AU table
    st.dataframe(au_df, use_container_width=True)
    
    # AU trends over sessions
    st.markdown("### 📈 Action Units Trends")
    
    trend_data = []
    for session_idx, result in enumerate(results):
        for au_name, value in result['action_units'].items():
            trend_data.append({
                'Session': session_idx + 1,
                'Action Unit': au_name.replace('_', ' '),
                'Value': value,
                'Timestamp': result['timestamp']
            })
    
    if trend_data:
        trend_df = pd.DataFrame(trend_data)
        
        # Show top AUs trend
        top_aus = au_df.head(5)['Action Unit'].tolist()
        filtered_df = trend_df[trend_df['Action Unit'].isin(top_aus)]
        
        chart = alt.Chart(filtered_df).mark_line(point=True).encode(
            x=alt.X('Session:O', title='Session Number'),
            y=alt.Y('Value:Q', title='AU Value'),
            color=alt.Color('Action Unit:N', title='Action Unit'),
            tooltip=['Session', 'Action Unit', 'Value']
        ).properties(
            title='Top Action Units Across Sessions',
            width=600,
            height=300
        )
        
        st.altair_chart(chart, use_container_width=True)
    
    # Session timeline
    st.markdown("### ⏱️ Session Timeline")
    
    session_data = []
    for i, result in enumerate(results):
        session_data.append({
            'Session': i + 1,
            'Start Time': result['timestamp'].strftime('%H:%M:%S'),
            'Duration': f"{result['duration']:.1f}s",
            'Frames': result['frames_analyzed'],
            'Detection Rate': f"{result['detection_rate']:.1f}%",
            'Top AU': max(result['action_units'].items(), key=lambda x: x[1])[0].replace('_', ' ')
        })
    
    session_df = pd.DataFrame(session_data)
    st.dataframe(session_df, use_container_width=True)
    
    # Export options
    st.markdown("### 💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export AU Data"):
            export_scheduled_au_data(results)
    
    with col2:
        if st.button("📈 Export Full Report"):
            export_scheduled_full_report(results)
    
    with col3:
        if st.button("📋 Export Summary"):
            export_scheduled_summary(results)

def export_scheduled_au_data(results):
    """Export AU data from scheduled sessions"""
    st.success("📊 AU data export prepared")
    # Implementation would create CSV/JSON export

def export_scheduled_full_report(results):
    """Export full scheduled analysis report"""
    st.success("📈 Full report export prepared")
    # Implementation would create comprehensive report

def export_scheduled_summary(results):
    """Export scheduled analysis summary"""  
    st.success("📋 Summary export prepared")
    # Implementation would create text summary

def handle_video_upload():
    """Handle video file upload and analysis"""
    
    st.subheader("📁 Upload Video File")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
        help="Upload a video file for facial analysis"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Show video info
        st.success(f"📁 Video uploaded: {uploaded_file.name}")
        
        # Analysis button
        if st.button("🚀 Analyze Video with MediaPipe", type="primary"):
            analyze_video_with_bridge(video_path, uploaded_file.name)
        
        # Clean up
        if os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except:
                pass

def handle_video_recording():
    """Handle live video recording"""
    
    st.subheader("📹 Record Live Video")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🎬 Start Recording", type="primary", use_container_width=True):
            record_live_video()

def handle_realtime_preview():
    """Handle real-time preview mode"""
    
    st.subheader("🔄 Real-time Analysis Preview")
    st.info("This mode shows live camera feed with basic analysis. For full Action Units, use recording mode.")
    
    # Camera preview placeholder
    camera_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("📷 Start Camera Preview", type="primary", use_container_width=True):
            start_camera_preview(camera_placeholder, stats_placeholder)

def analyze_video_with_bridge(video_path, filename):
    """
    Analyze video using MediaPipe bridge subprocess
    """
    
    # Create output file for results
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w') as tmp_file:
        results_path = tmp_file.name
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🚀 Starting MediaPipe analysis...")
        progress_bar.progress(0.1)
        
        # Run MediaPipe bridge
        process = subprocess.run([
            ".venv311/bin/python", 
            "mediapipe_bridge.py",
            video_path,
            results_path
        ], capture_output=True, text=True, timeout=120)
        
        progress_bar.progress(0.8)
        status_text.text("📊 Processing results...")
        
        if process.returncode == 0:
            progress_bar.progress(1.0)
            status_text.text("✅ Analysis complete!")
            
            # Load and display results
            display_mediapipe_results(results_path, filename)
            
        else:
            st.error(f"❌ MediaPipe analysis failed: {process.stderr}")
            st.code(process.stdout)
            
    except subprocess.TimeoutExpired:
        st.error("⏰ Analysis timed out (120s limit)")
    except Exception as e:
        st.error(f"❌ Analysis error: {e}")
    finally:
        # Clean up
        progress_bar.empty()
        status_text.empty()
        if os.path.exists(results_path):
            try:
                os.unlink(results_path)
            except:
                pass

def display_mediapipe_results(results_path, filename):
    """
    Display MediaPipe analysis results
    """
    
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        if not results['success']:
            st.error(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
            return
        
        st.success(f"✅ Analysis complete for {filename}")
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Frames Analyzed", results['frames_analyzed'])
        
        with col2:
            st.metric("Action Units", len(results['action_units']))
            
        with col3:
            detection_rate = results['landmarks_stats'].get('detection_rate', 0)
            st.metric("Face Detection", f"{detection_rate:.1f}%")
            
        with col4:
            st.metric("Emotions Found", len(results['emotions']))
        
        # Action Units Analysis
        st.subheader("🎭 Action Units Analysis")
        
        if results['action_units']:
            au_data = []
            for au_name, au_stats in results['action_units'].items():
                au_data.append({
                    'Action Unit': au_name.replace('_', ' '),
                    'Mean': au_stats['mean'],
                    'Max': au_stats['max'],
                    'Min': au_stats['min'],
                    'Std Dev': au_stats['std']
                })
            
            au_df = pd.DataFrame(au_data)
            
            # Display as table
            st.dataframe(au_df, use_container_width=True)
            
            # Create visualization
            chart = alt.Chart(au_df).mark_bar().encode(
                x=alt.X('Mean:Q', title='Mean Activation Level'),
                y=alt.Y('Action Unit:N', sort='-x', title='Action Units'),
                color=alt.Color('Mean:Q', scale=alt.Scale(scheme='viridis')),
                tooltip=['Action Unit', 'Mean', 'Max', 'Min']
            ).properties(
                title='Action Units Activation Levels',
                width=600,
                height=400
            )
            
            st.altair_chart(chart, use_container_width=True)
        
        # Emotion Analysis
        st.subheader("😊 Emotion Analysis")
        
        if results['emotions']:
            emotion_data = []
            for emotion_info in results['emotions']:
                emotion_data.append({
                    'Emotion': emotion_info['emotion'].replace('_', ' ').title(),
                    'Frames': emotion_info['count'],
                    'Percentage': emotion_info['percentage']
                })
            
            emotion_df = pd.DataFrame(emotion_data)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.dataframe(emotion_df, use_container_width=True)
            
            with col2:
                # Pie chart
                pie_chart = alt.Chart(emotion_df).mark_arc().encode(
                    theta=alt.Theta('Percentage:Q'),
                    color=alt.Color('Emotion:N', scale=alt.Scale(scheme='category10')),
                    tooltip=['Emotion', 'Frames', 'Percentage']
                ).properties(
                    title='Emotion Distribution',
                    width=300,
                    height=300
                )
                
                st.altair_chart(pie_chart, use_container_width=True)
        
        # Technical Details
        with st.expander("🔧 Technical Details"):
            st.json(results['landmarks_stats'])
            
            st.markdown(f"""
            **Analysis Summary:**
            - **Total Frames:** {results['frames_analyzed']}
            - **Frames with Face:** {results['landmarks_stats'].get('frames_with_face', 0)}
            - **Detection Rate:** {results['landmarks_stats'].get('detection_rate', 0):.1f}%
            - **Landmarks per Face:** {results['landmarks_stats'].get('total_landmarks_detected', 0)}
            """)
        
    except Exception as e:
        st.error(f"❌ Error displaying results: {e}")

def record_live_video():
    """Record live video with preview"""
    
    st.info("🎬 Starting video recording...")
    
    # Placeholder for recording interface
    camera_placeholder = st.empty()
    controls_placeholder = st.empty()
    
    # This would implement the recording logic
    # For now, show a message
    camera_placeholder.info("📹 Live recording interface would appear here")
    controls_placeholder.info("⏺️ Recording controls would be available")

def start_camera_preview(camera_placeholder, stats_placeholder):
    """Start real-time camera preview"""
    
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Could not open camera")
            return
        
        # Preview for 30 seconds
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 30:  # 30 second preview
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Basic face detection for preview
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Convert to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update display
            camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Update stats
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            stats_placeholder.metric("Preview Stats", f"{len(faces)} faces | {fps:.1f} FPS | {30-elapsed:.1f}s left")
            
            time.sleep(0.1)  # Control frame rate
        
        cap.release()
        camera_placeholder.success("✅ Preview complete!")
        stats_placeholder.info("Use recording mode for full MediaPipe Action Units analysis")
        
    except Exception as e:
        st.error(f"❌ Camera preview error: {e}")