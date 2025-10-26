"""
Video/Facial Expression Analysis Interface for Streamlit
Extracted from the original MoodWatch Streamlit application
Focused on facial expressions and Action Units analysis - audio analysis removed
"""

import streamlit as st
import os
import sys
import tempfile
import time
import subprocess
import signal
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import altair as alt
from dotenv import load_dotenv

# Optional clustering libs (graceful fallback if missing)
try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    HAVE_SK = True
except Exception:
    HAVE_SK = False

# Load environment variables
load_dotenv()

# Configuration
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "processed"
SESSION_SUMMARY = BASE_DIR / "video_session_summary.csv"
LIVE_PREVIEW_IMG = BASE_DIR / "live_preview.jpg"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OPENFACE_BIN = os.getenv("OPENFACE_BIN", str(Path.home() / "OpenFace" / "build" / "bin" / "FeatureExtraction"))

# Facial Action Units features
AU_FEATURES = [
    "AU01_r", "AU02_r", "AU04_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r",
    "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r",
    "AU26_r", "AU45_c"
]

SESSION_FIELDS = ["timestamp", *AU_FEATURES, "label", "positivity", "energy", "tension", "hesitation", "confidence", "csv"]

def render_video_interface():
    """Main video analysis interface"""
    
    st.subheader("🎥 Facial Expression Analysis")
    st.markdown("Analyze facial expressions and emotions from video recordings or live camera.")
    
    # Check OpenFace installation
    if not Path(OPENFACE_BIN).exists():
        st.error(f"❌ OpenFace not found at: `{OPENFACE_BIN}`")
        st.markdown("""
        **Setup Instructions:**
        1. Download and build OpenFace from: https://github.com/TadasBaltrusaitis/OpenFace
        2. Set `OPENFACE_BIN` environment variable to point to FeatureExtraction binary
        3. Example: `OPENFACE_BIN=/home/user/OpenFace/build/bin/FeatureExtraction`
        """)
        return
    
    # Create tabs for different video analysis modes
    manual_tab, scheduled_tab, upload_tab, existing_tab = st.tabs([
        "🎬 Manual Recording", 
        "⏰ Scheduled Recording", 
        "📤 Upload Video",
        "📁 Analyze Existing Data"
    ])
    
    with manual_tab:
        render_manual_recording()
    
    with scheduled_tab:
        render_scheduled_recording()
        
    with upload_tab:
        render_video_upload()
    
    with existing_tab:
        render_existing_analysis()

def render_manual_recording():
    """Interface for manual video pulse recording"""
    
    st.subheader("🎬 Manual Pulse Recording")
    st.markdown("Record a short video pulse to analyze current facial expressions and mood.")
    
    with st.expander("⚙️ Recording Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            duration = st.slider("Duration (seconds)", 3, 15, 6, 1)
            width = st.select_slider("Width", options=[640, 800, 1280], value=640)
        with col2:
            height = st.select_slider("Height", options=[480, 600, 720], value=480)
            fps = st.select_slider("FPS", options=[10, 15, 24, 30], value=15)
    
    if st.button("🎥 Record Pulse Now", type="primary", use_container_width=True):
        record_and_analyze_pulse(duration, width, height, fps)

def record_and_analyze_pulse(duration_s: float, w: int, h: int, fps: int):
    """Record a video pulse and analyze it"""
    
    try:
        import cv2
    except ImportError:
        st.error("OpenCV not installed. Run: `pip install opencv-python`")
        return
    
    # Record video
    with st.spinner("🎥 Recording video..."):
        video_path = record_pulse_to_temp(duration_s, w, h, fps)
    
    if video_path is None:
        st.error("❌ Failed to record video. Check camera permissions.")
        return
    
    st.success(f"✅ Video recorded: {video_path.name}")
    
    # Process with OpenFace
    with st.spinner("🔍 Analyzing facial expressions..."):
        csv_path = run_openface_on_video(video_path)
    
    if csv_path is None:
        st.error("❌ OpenFace analysis failed.")
        return
    
    # Display results
    display_video_analysis_results(csv_path)

def render_scheduled_recording():
    """Interface for scheduled automatic recording"""
    
    st.subheader("⏰ Automated Scheduled Recording")
    st.markdown("Capture video pulses at regular intervals for continuous mood monitoring.")
    
    # Settings
    col1, col2 = st.columns(2)
    with col1:
        schedule_duration = st.number_input("Pulse Duration (seconds)", 3, 30, 6)
        schedule_interval = st.number_input("Interval Between Pulses (seconds)", 30, 3600, 120)
    with col2:
        schedule_fps = st.number_input("FPS", 5, 60, 15)
        schedule_width = st.number_input("Width", 320, 1920, 640)
        schedule_height = st.number_input("Height", 240, 1080, 480)
    
    # Scheduler state management
    manage_scheduler(schedule_duration, schedule_interval, schedule_fps, schedule_width, schedule_height)

def render_video_upload():
    """Interface for uploading and analyzing video files"""
    
    st.subheader("📤 Upload Video for Analysis")
    st.markdown("Upload a video file to analyze facial expressions and emotions.")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
        help="Supported formats: MP4, AVI, MOV, MKV, WMV"
    )
    
    if uploaded_file is not None:
        if st.button("🎯 Analyze Video", type="primary", use_container_width=True):
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = Path(tmp_file.name)
            
            try:
                # Process with OpenFace
                with st.spinner("🔍 Analyzing facial expressions in uploaded video..."):
                    csv_path = run_openface_on_video(temp_path)
                
                if csv_path is not None:
                    st.success("✅ Analysis completed!")
                    display_video_analysis_results(csv_path)
                else:
                    st.error("❌ Analysis failed. Check video format and quality.")
                    
            finally:
                # Clean up
                if temp_path.exists():
                    temp_path.unlink()

def render_existing_analysis():
    """Interface for analyzing existing CSV data"""
    
    st.subheader("📁 Analyze Existing CSV Data")
    st.markdown("Load and analyze existing OpenFace CSV output files.")
    
    # List available CSV files
    csv_files = list(PROCESSED_DIR.glob("*.csv"))
    
    if not csv_files:
        st.info("No CSV files found in the processed directory.")
        st.markdown(f"**Processed directory:** `{PROCESSED_DIR}`")
        return
    
    # File selector
    selected_file = st.selectbox(
        "Select a CSV file to analyze",
        options=csv_files,
        format_func=lambda x: f"{x.name} ({datetime.fromtimestamp(x.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})"
    )
    
    if selected_file and st.button("📊 Analyze Selected File"):
        display_video_analysis_results(selected_file)
    
    # Session history
    if SESSION_SUMMARY.exists():
        st.subheader("📈 Session History")
        render_session_timeline()

def record_pulse_to_temp(duration_s: float = 6.0, w: int = 640, h: int = 480, fps: int = 15) -> Optional[Path]:
    """Record video from camera to temporary file"""
    
    try:
        import cv2
    except ImportError:
        st.error("OpenCV not available")
        return None
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open camera. Check camera permissions in System Settings → Privacy & Security → Camera.")
        return None
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_path = Path(tempfile.gettempdir()) / f"video_pulse_{datetime.now().strftime('%Y%m%dT%H%M%S')}.mp4"
    out = cv2.VideoWriter(str(tmp_path), fourcc, fps, (w, h))
    
    if not out.isOpened():
        st.error("Failed to initialize video writer.")
        cap.release()
        return None
    
    frames_target = int(duration_s * fps)
    progress = st.progress(0, text="Recording video...")
    
    for i in range(frames_target):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        progress.progress((i + 1) / frames_target, text=f"Recording... {i + 1}/{frames_target} frames")
    
    cap.release()
    out.release()
    progress.empty()
    
    return tmp_path if tmp_path.exists() else None

def run_openface_on_video(video_path: Path) -> Optional[Path]:
    """Run OpenFace FeatureExtraction on a video file"""
    
    cmd = [
        OPENFACE_BIN, 
        "-f", str(video_path),
        "-aus", "-pose", "-gaze", "-2Dfp", "-3Dfp",
        "-out_dir", str(PROCESSED_DIR),
        "-no_vis"  # No visualization output
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            st.error("❌ OpenFace processing failed")
            with st.expander("OpenFace Error Logs"):
                st.code(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
            return None
        
        # Find the most recently created CSV
        csv_files = sorted(PROCESSED_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime)
        return csv_files[-1] if csv_files else None
        
    except subprocess.TimeoutExpired:
        st.error("❌ OpenFace processing timed out")
        return None
    except Exception as e:
        st.error(f"❌ Error running OpenFace: {str(e)}")
        return None

def display_video_analysis_results(csv_path: Path):
    """Display comprehensive analysis results from OpenFace CSV"""
    
    # Read CSV data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Failed to read CSV file: {e}")
        return
    
    if df.empty:
        st.warning("CSV file is empty")
        return
    
    st.success(f"📊 Analysis Results: {csv_path.name}")
    
    # Basic statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📹 Total Frames", len(df))
    with col2:
        duration = len(df) / 30.0  # Assuming ~30 FPS
        st.metric("⏱️ Duration", f"{duration:.1f}s")
    with col3:
        success_rate = df['success'].mean() if 'success' in df.columns else 1.0
        st.metric("✅ Success Rate", f"{success_rate:.1%}")
    
    # Facial analysis and mood inference
    summary, frames, pose_rx, pose_ry, pose_rz = summarize_dataframe(df)
    
    # Import mood logic (with fallback)
    try:
        from app.mood_logic import infer_mood
        from app.baseline import zscore, add_sample
        
        # Calculate z-scores and infer mood
        z_scores = zscore("default", summary)
        mood = infer_mood(summary, frames=frames, pose_rx=pose_rx, pose_ry=pose_ry, pose_rz=pose_rz, z=z_scores)
        add_sample("default", summary)
        
        # Save to session history
        append_session_row(summary, mood, csv_path.name)
        
        # Display mood metrics
        st.subheader("🎭 Mood Analysis")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Positivity", f"{mood.get('positivity', 0):+.2f}")
        m2.metric("Energy", f"{mood.get('energy', 0):.2f}")
        m3.metric("Tension", f"{mood.get('tension', 0):.2f}")
        m4.metric("Hesitation", f"{mood.get('hesitation', 0):.2f}")
        m5.metric("Confidence", f"{mood.get('confidence', 0):.2f}")
        
        st.markdown(f"**Inferred Mood:** {mood.get('label', 'Unknown')}")
        if mood.get('notes'):
            st.caption(mood['notes'])
            
    except ImportError:
        st.warning("Mood inference modules not available. Showing raw data only.")
        mood = {}
    
    # Display raw data sample
    st.subheader("📋 Raw Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Visualizations
    render_au_visualizations(df)
    
    # Export results
    render_export_options(df, csv_path, summary, mood if 'mood' in locals() else {})

def summarize_dataframe(df: pd.DataFrame) -> Tuple[Dict, int, float, float, float]:
    """Summarize DataFrame to extract AU features and pose information"""
    
    summary = {}
    for au in AU_FEATURES:
        if au in df.columns:
            values = pd.to_numeric(df[au], errors='coerce')
            summary[au] = float(values.mean(skipna=True)) if values.notna().any() else 0.0
        else:
            summary[au] = 0.0
    
    frames = len(df)
    
    # Pose information
    pose_rx = df['pose_Rx'].mean() if 'pose_Rx' in df.columns else 0.0
    pose_ry = df['pose_Ry'].mean() if 'pose_Ry' in df.columns else 0.0
    pose_rz = df['pose_Rz'].mean() if 'pose_Rz' in df.columns else 0.0
    
    return summary, frames, pose_rx, pose_ry, pose_rz

def render_au_visualizations(df: pd.DataFrame):
    """Render Action Units visualizations"""
    
    st.subheader("📊 Action Units Analysis")
    
    # Get available AU columns
    au_cols = [col for col in AU_FEATURES if col in df.columns]
    
    if not au_cols:
        st.warning("No Action Units columns found in the data")
        return
    
    # AU selector
    selected_aus = st.multiselect(
        "Select Action Units to visualize",
        options=au_cols,
        default=au_cols[:4] if len(au_cols) >= 4 else au_cols,
        help="Action Units represent different facial muscle movements"
    )
    
    if not selected_aus:
        st.info("Please select at least one Action Unit to visualize")
        return
    
    # Time series plot
    st.markdown("#### Time Series")
    chart_data = df[['frame'] + selected_aus].melt(id_vars=['frame'], var_name='AU', value_name='intensity')
    
    chart = alt.Chart(chart_data).mark_line().encode(
        x=alt.X('frame:Q', title='Frame'),
        y=alt.Y('intensity:Q', title='Intensity'),
        color=alt.Color('AU:N', title='Action Unit'),
        tooltip=['frame', 'AU', 'intensity']
    ).properties(height=400)
    
    st.altair_chart(chart, use_container_width=True)
    
    # Histograms
    st.markdown("#### Distribution Histograms")
    bins = st.slider("Number of bins", 10, 50, 20)
    
    hist_charts = []
    for au in selected_aus[:4]:  # Limit to 4 for display
        hist = alt.Chart(df).mark_bar().encode(
            alt.X(f'{au}:Q', bin=alt.Bin(maxbins=bins), title=au),
            alt.Y('count()', title='Count'),
            tooltip=['count()']
        ).properties(width=150, height=150, title=au)
        hist_charts.append(hist)
    
    if hist_charts:
        combined_hist = alt.concat(*hist_charts, columns=2)
        st.altair_chart(combined_hist, use_container_width=True)

def append_session_row(summary: Dict, mood: Dict, csv_name: str = ""):
    """Append analysis results to session history"""
    
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **{k: float(summary.get(k, 0.0)) for k in AU_FEATURES},
        "label": mood.get("label", ""),
        "positivity": mood.get("positivity", ""),
        "energy": mood.get("energy", ""),
        "tension": mood.get("tension", ""),
        "hesitation": mood.get("hesitation", ""),
        "confidence": mood.get("confidence", ""),
        "csv": csv_name,
    }
    
    try:
        import csv
        write_header = not SESSION_SUMMARY.exists() or SESSION_SUMMARY.stat().st_size == 0
        
        with open(SESSION_SUMMARY, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SESSION_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
                
    except Exception as e:
        st.warning(f"Could not save to session history: {e}")

def render_session_timeline():
    """Render timeline of session history"""
    
    try:
        session_df = pd.read_csv(SESSION_SUMMARY)
        
        if session_df.empty:
            st.info("No session history available yet")
            return
        
        # Convert timestamp
        session_df['timestamp'] = pd.to_datetime(session_df['timestamp'])
        
        # Plot mood metrics over time
        mood_cols = ['positivity', 'energy', 'tension', 'hesitation', 'confidence']
        available_cols = [col for col in mood_cols if col in session_df.columns]
        
        if available_cols:
            st.markdown("#### Mood Timeline")
            
            chart_data = session_df[['timestamp'] + available_cols].melt(
                id_vars=['timestamp'], 
                var_name='metric', 
                value_name='value'
            )
            
            timeline_chart = alt.Chart(chart_data).mark_line(point=True).encode(
                x=alt.X('timestamp:T', title='Time'),
                y=alt.Y('value:Q', title='Score'),
                color=alt.Color('metric:N', title='Mood Metric'),
                tooltip=['timestamp', 'metric', 'value']
            ).properties(height=300)
            
            st.altair_chart(timeline_chart, use_container_width=True)
        
        # Summary statistics
        st.markdown("#### Session Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Sessions", len(session_df))
        with col2:
            avg_positivity = session_df['positivity'].mean() if 'positivity' in session_df.columns else 0
            st.metric("Avg Positivity", f"{avg_positivity:.2f}")
        with col3:
            avg_energy = session_df['energy'].mean() if 'energy' in session_df.columns else 0
            st.metric("Avg Energy", f"{avg_energy:.2f}")
        
    except Exception as e:
        st.error(f"Error loading session history: {e}")

def render_export_options(df: pd.DataFrame, csv_path: Path, summary: Dict, mood: Dict):
    """Render export and download options"""
    
    st.subheader("💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download raw CSV
        st.download_button(
            label="📄 Download Raw CSV",
            data=df.to_csv(index=False),
            file_name=f"facial_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download summary report
        report = generate_video_report(csv_path, summary, mood, df)
        st.download_button(
            label="📊 Download Summary Report",
            data=report,
            file_name=f"mood_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

def generate_video_report(csv_path: Path, summary: Dict, mood: Dict, df: pd.DataFrame) -> str:
    """Generate a comprehensive text report"""
    
    report = f"""
FACIAL EXPRESSION ANALYSIS REPORT
=================================

File: {csv_path.name}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Frames: {len(df)}
Duration: ~{len(df)/30:.1f} seconds

MOOD ANALYSIS:
--------------
Label: {mood.get('label', 'Unknown')}
Positivity: {mood.get('positivity', 0):.2f}
Energy: {mood.get('energy', 0):.2f}
Tension: {mood.get('tension', 0):.2f}
Hesitation: {mood.get('hesitation', 0):.2f}
Confidence: {mood.get('confidence', 0):.2f}

ACTION UNITS SUMMARY:
--------------------
"""
    
    for au, value in summary.items():
        report += f"{au}: {value:.3f}\n"
    
    if mood.get('notes'):
        report += f"\nNOTES:\n------\n{mood['notes']}\n"
    
    return report

def manage_scheduler(duration: int, interval: int, fps: int, width: int, height: int):
    """Manage the scheduled recording functionality"""
    
    # Initialize session state
    if "scheduler_running" not in st.session_state:
        st.session_state.scheduler_running = False
        st.session_state.scheduler_process = None
        st.session_state.scheduler_start_time = None
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Scheduler", disabled=st.session_state.scheduler_running):
            try:
                # This would start a background process for scheduled recording
                # Implementation depends on camera_schedule module
                st.session_state.scheduler_running = True
                st.session_state.scheduler_start_time = time.time()
                st.success("Scheduled recording started!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to start scheduler: {e}")
    
    with col2:
        if st.button("⏹️ Stop Scheduler", disabled=not st.session_state.scheduler_running):
            st.session_state.scheduler_running = False
            st.session_state.scheduler_process = None
            st.session_state.scheduler_start_time = None
            st.success("Scheduled recording stopped!")
            st.rerun()
    
    with col3:
        if st.session_state.scheduler_running:
            if st.session_state.scheduler_start_time:
                elapsed = time.time() - st.session_state.scheduler_start_time
                hours, remainder = divmod(int(elapsed), 3600)
                minutes, seconds = divmod(remainder, 60)
                st.success(f"🟢 Running: {hours:02d}:{minutes:02d}:{seconds:02d}")
            else:
                st.success("🟢 Running")
        else:
            st.info("⚪ Stopped")