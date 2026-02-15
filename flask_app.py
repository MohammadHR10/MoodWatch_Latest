#!/usr/bin/env python3
"""
Flask App for Audio and Video Mood Analysis
Optimized for GCP Cloud Run deployment
"""

import os
import sys
import cv2
import json
import tempfile
import time
import subprocess
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, send_file
from werkzeug.utils import secure_filename
from collections import deque, defaultdict
import numpy as np
from threading import Thread, Event
import logging
from typing import Optional
from groq import Groq

# Load configuration
from config import load_dotenv
load_dotenv()

# OpenFace support (optional - only for local development)
find_feature_extraction_binary = None
_openface_env = None
try:
    from openface_bridge import find_feature_extraction_binary
    from openface_bridge import _augmented_env as _openface_env
except ImportError:
    pass  # OpenFace not available (expected in cloud)

# Optional: py-feat backend diagnostics and canonical AU names
try:
    from pyfeat_bridge import check_backend as pyfeat_check
    try:
        from pyfeat_bridge import AU_NAME_FULL as _AU_NAME_FULL
        CANONICAL_AU_NAMES = list(_AU_NAME_FULL.values())
    except Exception:
        pyfeat_check = pyfeat_check
        # Py-feat detects 20 AUs (AU27 not included)
        CANONICAL_AU_NAMES = [
            'AU01_Inner_Brow_Raiser','AU02_Outer_Brow_Raiser','AU04_Brow_Lowerer',
            'AU05_Upper_Lid_Raiser','AU06_Cheek_Raiser','AU07_Lid_Tightener',
            'AU09_Nose_Wrinkler','AU10_Upper_Lip_Raiser','AU11_Nasolabial_Deepener',
            'AU12_Lip_Corner_Puller','AU14_Dimpler','AU15_Lip_Corner_Depressor','AU17_Chin_Raiser',
            'AU20_Lip_Stretcher','AU23_Lip_Tightener','AU24_Lip_Pressor','AU25_Lips_Part',
            'AU26_Jaw_Drop','AU28_Lip_Suck','AU43_Eyes_Closed'
        ]
except Exception:
    pyfeat_check = None
    # Py-feat detects 20 AUs (AU27 not included)
    CANONICAL_AU_NAMES = [
        'AU01_Inner_Brow_Raiser','AU02_Outer_Brow_Raiser','AU04_Brow_Lowerer',
        'AU05_Upper_Lid_Raiser','AU06_Cheek_Raiser','AU07_Lid_Tightener',
        'AU09_Nose_Wrinkler','AU10_Upper_Lip_Raiser','AU11_Nasolabial_Deepener',
        'AU12_Lip_Corner_Puller','AU14_Dimpler','AU15_Lip_Corner_Depressor','AU17_Chin_Raiser',
        'AU20_Lip_Stretcher','AU23_Lip_Tightener','AU24_Lip_Pressor','AU25_Lips_Part',
        'AU26_Jaw_Drop','AU28_Lip_Suck','AU43_Eyes_Closed'
    ]

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['SESSIONS_DIR'] = 'sessions'

# Audio analyzer configuration
app.config['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
print(f"[STARTUP] GROQ_API_KEY from env: {bool(os.getenv('GROQ_API_KEY'))}, in config: {bool(app.config.get('GROQ_API_KEY'))}")
# Allowed audio extensions (include webm for microphone recording uploads)
app.config['ALLOWED_EXTENSIONS'] = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.webm'}

# Select face/emotion backend: 'pyfeat' (cloud default), 'openface' (local only), 'worker' (ML worker service)
EMOTION_BACKEND = os.getenv('EMOTION_BACKEND', 'pyfeat').lower()

# ML Worker URL for remote py-feat analysis
ML_WORKER_URL = os.getenv('ML_WORKER_URL', '')

def _call_ml_worker(frame):
    """Call ML worker service for AU analysis. Returns dict with aus, emotions."""
    import base64
    import io
    import requests
    from PIL import Image
    
    if not ML_WORKER_URL:
        return None
    
    try:
        # Convert frame to base64 JPEG
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = frame[:, :, ::-1]  # BGR to RGB
        img = Image.fromarray(frame)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=80)
        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        resp = requests.post(
            f"{ML_WORKER_URL}/analyze",
            json={'image': img_b64},
            timeout=60
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logging.warning(f"ML Worker error: {e}")
    return None

def _bridge_order():
    """Return list of bridge scripts in order to try based on EMOTION_BACKEND."""
    # If backend is 'none', return empty list (audio-only mode for cloud)
    if EMOTION_BACKEND in ('none', 'worker'):
        return []
    root = os.path.abspath(os.path.dirname(__file__))
    of = os.path.join(root, 'openface_bridge.py')
    pf = os.path.join(root, 'pyfeat_bridge.py')
    if EMOTION_BACKEND in ('both', 'auto'):
        return [pf, of]
    if EMOTION_BACKEND == 'pyfeat':
        return [pf]
    return [of]

# Global variables for video recording
recording = False
video_writer = None
video_file_path = None
start_time = None
duration = 10
recorded_frames = 0

# Storage for session results
session_results = []
session_count = 0

# Real-time face analysis data storage
current_realtime_data = {
    'recording': False,
    'elapsed': 0,
    'action_units': {},
    'emotion': 'Neutral',
    'frame_count': 0
}

# Per-session timeline of real-time emotions (appended during recording)
# Each entry: { t: seconds_since_start, label: str, confidence: float, quality: float, scores: {emotion:score} }
current_timeline = []

def _compute_emotion_summary_and_segments(timeline, min_conf: float = 0.15, neutral_is_empty: bool = True):
    """Compute summary stats and segments from a timeline list.
    - timeline: list of {t, label, confidence, quality, scores, aus}
    - min_conf: minimum confidence to consider an emotion "real" for segmentation
    - neutral_is_empty: if True, treat 'Neutral' labels as empty/background, not an emotion segment
    Returns: (summary_dict, segments_list)
    """
    if not isinstance(timeline, list) or not timeline:
        return {
            'dominant_emotion': 'Neutral',
            'avg_confidence': 0.0,
            'total_duration': 0.0,
            'changes': 0,
            'low_quality_pct': 0.0,
            'time_by_emotion': {}
        }, []

    # Sort by time to be safe and filter for valid entries
    tl = sorted([e for e in timeline if isinstance(e, dict) and isinstance(e.get('t'), (int, float))], key=lambda x: x['t'])
    if not tl:
        return {
            'dominant_emotion': 'Neutral',
            'avg_confidence': 0.0,
            'total_duration': 0.0,
            'changes': 0,
            'low_quality_pct': 0.0,
            'time_by_emotion': {}
        }, []

    total_duration = (tl[-1]['t'] - tl[0]['t']) if len(tl) > 1 else 0.0
    if total_duration < 0: total_duration = 0.0

    # --- Metrics based on non-Neutral, sufficient-confidence entries ---
    valid_entries = [e for e in tl if (e.get('label') != 'Neutral' or not neutral_is_empty) and float(e.get('confidence', 0.0) or 0.0) >= min_conf]
    
    # Compute changes, average confidence, low-quality percentage
    changes = 0
    conf_sum = 0.0
    lowq_count = 0
    last_label = None
    
    # Use all entries for quality and change detection, but only valid ones for confidence stats
    for e in tl:
        q = float(e.get('quality') or 0.0)
        if q < 0.45:
            lowq_count += 1
        
        lbl = str(e.get('label') or 'Neutral')
        conf = float(e.get('confidence') or 0.0)

        # A change is a transition *to* a confident, non-neutral state
        if last_label is not None and lbl != last_label:
            if conf >= min_conf and (lbl != 'Neutral' or not neutral_is_empty):
                changes += 1
        last_label = lbl

    if valid_entries:
        avg_confidence = sum(float(e.get('confidence', 0.0) or 0.0) for e in valid_entries) / len(valid_entries)
    else:
        avg_confidence = 0.0

    low_quality_pct = (lowq_count / max(1, len(tl))) * 100.0

    # --- Build segments (merge contiguous entries with same label above threshold) ---
    segments = []
    cur = None
    for e in tl:
        t = float(e.get('t') or 0.0)
        lbl = str(e.get('label') or 'Neutral')
        conf = float(e.get('confidence') or 0.0)
        
        # Determine effective label for segmentation
        lbl_eff = lbl if conf >= min_conf and (lbl != 'Neutral' or not neutral_is_empty) else 'Neutral'

        if cur is None:
            cur = { 'label': lbl_eff, 'start_t': t, 'end_t': t, 'conf_sum': conf, 'count': 1 }
        else:
            if lbl_eff == cur['label']:
                cur['end_t'] = t
                cur['conf_sum'] += conf
                cur['count'] += 1
            else:
                # Finalize previous segment if it's not just background Neutral
                if cur['label'] != 'Neutral' or not neutral_is_empty:
                    dur = max(0.0, cur['end_t'] - cur['start_t'])
                    if dur > 0: # Only add segments with duration
                        segments.append({
                            'label': cur['label'],
                            'start_t': round(cur['start_t'], 2),
                            'end_t': round(cur['end_t'], 2),
                            'duration': round(dur, 2),
                            'avg_confidence': round(cur['conf_sum'] / max(1, cur['count']), 3)
                        })
                # Start new segment
                cur = { 'label': lbl_eff, 'start_t': t, 'end_t': t, 'conf_sum': conf, 'count': 1 }
    
    # Add the last segment
    if cur is not None and (cur['label'] != 'Neutral' or not neutral_is_empty):
        dur = max(0.0, cur['end_t'] - cur['start_t'])
        if dur > 0:
            segments.append({
                'label': cur['label'],
                'start_t': round(cur['start_t'], 2),
                'end_t': round(cur['end_t'], 2),
                'duration': round(dur, 2),
                'avg_confidence': round(cur['conf_sum'] / max(1, cur['count']), 3)
            })

    # --- Final Summary from Segments ---
    time_by_emotion = {}
    for s in segments:
        time_by_emotion[s['label']] = time_by_emotion.get(s['label'], 0.0) + float(s.get('duration') or 0.0)

    dominant_emotion = 'Neutral'
    if time_by_emotion:
        dominant_emotion = max(time_by_emotion.items(), key=lambda x: x[1])[0]

    summary = {
        'dominant_emotion': dominant_emotion,
        'avg_confidence': round(avg_confidence, 3),
        'total_duration': round(total_duration, 2),
        'changes': int(changes),
        'low_quality_pct': round(low_quality_pct, 1),
        'time_by_emotion': {k: round(v, 2) for k, v in time_by_emotion.items()},
    }
    
    return summary, segments

# Keep a short history of recent emotion predictions to stabilize output
emotion_history = deque(maxlen=12)

# The audio blueprint is no longer needed as logic is integrated directly
# try:
#     from audio_analyzer.routes import audio_bp
#     app.register_blueprint(audio_bp, url_prefix='/audio')
# except ImportError as e:
#     app.logger.warning(f"Could not import or register audio blueprint: {e}")
#     # Define a placeholder if import fails, so app doesn't crash
#     audio_bp = None

# Background analysis pipeline (decouple heavy analysis from streaming loop)
analysis_queue = deque(maxlen=1)
analysis_thread = None
analysis_stop = Event()
pyfeat_warmed_up = False

# --- Constants and Configuration ---
# Define the mapping of Action Units to emotions based on Ekman's FACS
# Standardized to lowercase keys to match frontend expectations and raw score dictionaries
AU_EMOTION_RULES = {
    "happiness": {"required": ["AU06", "AU12"], "optional": [], "forbidden": []},
    "sadness": {"required": ["AU01", "AU04", "AU15"], "optional": [], "forbidden": ["AU12"]},
    "surprise": {"required": ["AU01", "AU02", "AU05", "AU26"], "optional": [], "forbidden": []},
    "fear": {"required": ["AU01", "AU02", "AU04", "AU05", "AU20"], "optional": ["AU26"], "forbidden": []},
    "anger": {"required": ["AU04", "AU05", "AU07", "AU23"], "optional": [], "forbidden": []},
    "disgust": {"required": ["AU09", "AU10"], "optional": ["AU15"], "forbidden": ["AU12"]},
}
AU_INTENSITY_THRESHOLD = 0.12  # Min intensity to consider an AU 'active' (slightly relaxed)
MIN_REQUIRED_AUS_FOR_EMOTION = 1 # Min number of required AUs to trigger an emotion

# --- Emotion Analysis Helpers ---

def get_emotion_from_aus(processed_aus):
    """
    Determines emotions from active Action Units using rule-based mapping.
    Returns (best_emotion, best_confidence, detected_emotions_dict).
    detected_emotions_dict contains scores for all emotions that meet the rule, not just the best.
    """
    if not processed_aus:
        return 'Neutral', 0.0, {}

    active_aus = {k: v for k, v in processed_aus.items() if v > AU_INTENSITY_THRESHOLD}
    detected_emotions = {}

    for emotion, rules in AU_EMOTION_RULES.items():
        # Check for forbidden AUs first
        is_forbidden = False
        for au_code in rules.get('forbidden', []):
            if any(key.startswith(au_code + '_') or key == au_code for key in active_aus):
                is_forbidden = True
                break
        if is_forbidden:
            continue

        # Check for required AUs
        active_required_aus_intensities = []
        for au_code in rules.get('required', []):
            for key, intensity in active_aus.items():
                if key.startswith(au_code + '_') or key == au_code:
                    active_required_aus_intensities.append(float(intensity))
                    break  # Found a match for this required AU, move to the next one

        required_found_count = len(active_required_aus_intensities)

        # Check if the minimum number of required AUs are active
        if required_found_count >= MIN_REQUIRED_AUS_FOR_EMOTION:
            # Simple average confidence of active AUs
            total_confidence = sum(active_required_aus_intensities)
            avg_confidence = total_confidence / required_found_count if required_found_count > 0 else 0.0
            detected_emotions[emotion] = max(0.0, min(1.0, avg_confidence))

    if not detected_emotions:
        return 'Neutral', 0.0, {}

    # Return the emotion with the highest average confidence among detected ones
    best_emotion = max(detected_emotions, key=detected_emotions.get)
    return best_emotion.capitalize() if best_emotion.lower() != 'neutral' else 'Neutral', float(detected_emotions[best_emotion]), detected_emotions


def smooth_emotion(new_emotion, new_confidence):
    """Smooths emotion labels over a rolling window to reduce flicker."""
    global emotion_history
    emotion_history.append((new_emotion, new_confidence))

    # Aggregate confidences for each emotion in the history
    agg = {}
    # Use exponential decay so older frames contribute less
    weights = np.exp(-0.1 * np.arange(len(emotion_history)))[::-1]
    
    for (label, conf), weight in zip(emotion_history, weights):
        agg[label] = agg.get(label, 0.0) + conf * weight

    if not agg:
        return 'Neutral', 0.0

    # Find the emotion with the highest weighted score
    best_emotion = max(agg, key=agg.get)
    best_weight = agg[best_emotion]

    # Only return the emotion if its weighted score is above a threshold
    # This prevents fleeting, low-confidence emotions from dominating
    if best_weight > 0.5:
        return best_emotion, best_weight
    else:
        return 'Neutral', 0.0

# Force OpenFace backend (MediaPipe removed per request)

@app.route('/')
def index():
    """Main page with audio and video options"""
    return render_template('index.html')

def process_frame_realtime(frame):
    """Process a single frame with analysis backend for real-time analysis"""
    global current_realtime_data, current_timeline
    
    try:
        # If using ML Worker backend, call it directly via HTTP
        if EMOTION_BACKEND == 'worker' and ML_WORKER_URL:
            worker_result = _call_ml_worker(frame)
            if worker_result and worker_result.get('success'):
                if worker_result.get('face_detected'):
                    aus = worker_result.get('aus', {})
                    emotions = worker_result.get('emotions', {})
                    
                    # Update realtime data
                    current_realtime_data['action_units'] = aus
                    current_realtime_data['frame_count'] = current_realtime_data.get('frame_count', 0) + 1
                    
                    # Find dominant emotion
                    if emotions:
                        dom_emo = max(emotions, key=emotions.get)
                        dom_conf = emotions[dom_emo]
                        current_realtime_data['emotion'] = dom_emo.capitalize()
                        
                        # Add to timeline
                        elapsed = current_realtime_data.get('elapsed', 0)
                        current_timeline.append({
                            't': elapsed,
                            'label': dom_emo.capitalize(),
                            'confidence': dom_conf,
                            'quality': 1.0,
                            'scores': emotions,
                            'aus': aus
                        })
                else:
                    current_realtime_data['emotion'] = 'No Face'
                    current_realtime_data['action_units'] = {}
            return
        
        # Save frame temporarily for analysis
        temp_frame = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        cv2.imwrite(temp_frame.name, frame)
        temp_frame.close()
        
        # Use analysis bridge(s) in priority order
        python3_path = sys.executable
        bridge_scripts = _bridge_order()
        
        try:
            analysis_result = None
            last_err = None
            for bridge_script in bridge_scripts:
                if not bridge_script or not os.path.exists(bridge_script):
                    last_err = f"Bridge not found: {bridge_script}"
                    continue
                # Allow longer timeout for py-feat on first run (model downloads/warm-up)
                is_pyfeat = os.path.basename(bridge_script).startswith('pyfeat')
                per_timeout = 10.0
                if is_pyfeat:
                    per_timeout = 12.0 if pyfeat_warmed_up else 25.0
                result = subprocess.run([
                    python3_path, bridge_script, temp_frame.name
                ], capture_output=True, text=True, timeout=per_timeout)
                if result.returncode != 0 or not (result.stdout or '').strip():
                    last_err = f"Bridge failed rc={result.returncode}: {bridge_script} ({(result.stderr or '')[:120]})"
                    continue
                # Parse JSON from stdout
                try:
                    stdout_lines = result.stdout.strip().split('\n')
                    json_lines = []
                    in_json = False
                    for line in stdout_lines:
                        if line.strip().startswith('{'):
                            in_json = True
                        if in_json:
                            json_lines.append(line)
                    if not json_lines:
                        last_err = f"No JSON in output from {os.path.basename(bridge_script)}"
                        continue
                    json_str = '\n'.join(json_lines)
                    analysis_result = json.loads(json_str)
                    if isinstance(analysis_result, dict) and 'error' not in analysis_result:
                        # Mark py-feat warmed after a successful call
                        if is_pyfeat:
                            try:
                                globals()['pyfeat_warmed_up'] = True
                            except Exception:
                                pass
                        break
                    else:
                        last_err = str(analysis_result.get('error') if isinstance(analysis_result, dict) else 'unknown error')
                        analysis_result = None
                except Exception as pe:
                    last_err = f"JSON parse error: {pe}"
                    analysis_result = None
            
            if analysis_result is not None:
                try:
                    if not isinstance(analysis_result, dict):
                        print(f"Unexpected analysis_result type: {type(analysis_result)}, value: {str(analysis_result)[:200]}")
                        return
                    
                    # Extract real-time Action Units data
                    if 'error' in analysis_result:
                        # Surface backend installation/config issues clearly
                        err = analysis_result.get('error') or last_err or 'unknown error'
                        print(f"Analysis bridge reported error: {err}")
                        current_realtime_data.update({
                            'recording': recording,
                            'elapsed': time.time() - start_time if recording and start_time else 0,
                            'action_units': {},
                            'emotion': 'Neutral',
                            'emotion_confidence': 0.0,
                            'frame_count': current_realtime_data.get('frame_count', 0),
                            'last_update': time.time(),
                            'landmark_quality': 0.0,
                            'error': str(err)[:300]
                        })
                        return
                    action_units = analysis_result.get('action_units', {})
                    emotions = analysis_result.get('emotions', {})
                    # Quality may be float (single image) or dict (video stats elsewhere)
                    quality_raw = analysis_result.get('landmarks_stats', None)
                    landmark_quality = 0.0
                    if isinstance(quality_raw, (int, float)):
                        landmark_quality = float(quality_raw)
                    elif isinstance(quality_raw, dict):
                        # Fallbacks if a dict is provided: try mean_confidence or detection_rate
                        landmark_quality = float(quality_raw.get('mean_confidence', 0.0) or 0.0)
                    
                    # Process AUs to get mean values (ensure all canonical AUs present)
                    processed_aus = {}
                    for au_name, au_data in action_units.items():
                        if isinstance(au_data, dict) and 'mean' in au_data:
                            processed_aus[au_name] = au_data['mean']
                        elif isinstance(au_data, (int, float)):
                            processed_aus[au_name] = au_data
                    # Fill missing AUs with 0.0 so visualizations can render all rows/series efficiently
                    try:
                        for au_full in CANONICAL_AU_NAMES:
                            if au_full not in processed_aus:
                                processed_aus[au_full] = 0.0
                    except Exception:
                        pass
                    
                    # Get dominant emotion - DEPRECATED direct classifier output
                    # dominant_emotion = 'Neutral'
                    # emotion_confidence = 0
                    # is_significant = False
                    # all_scores = None
                    # if emotions and isinstance(emotions, dict):
                    #     dominant_emotion = emotions.get('emotion', 'Neutral')
                    #     emotion_confidence = emotions.get('confidence', 0)
                    #     is_significant = bool(emotions.get('is_significant', False))
                    #     if isinstance(emotions.get('all_scores'), dict):
                    #         all_scores = emotions.get('all_scores')
                    
                    # NEW: Get emotion from AU rules for higher accuracy
                    dominant_emotion, emotion_confidence, detected = get_emotion_from_aus(processed_aus)

                    # Build a comprehensive score dictionary for all 7 emotions
                    all_scores = {
                        "happiness": 0.0, "sadness": 0.0, "surprise": 0.0,
                        "anger": 0.0, "fear": 0.0, "disgust": 0.0, "neutral": 0.0
                    }
                    # Populate scores for all detected emotions
                    for emo, score in (detected or {}).items():
                        key = str(emo).lower()
                        if key in all_scores:
                            all_scores[key] = float(max(0.0, min(1.0, score)))
                    # Set neutral as residual confidence (1 - max other), but not below 0
                    max_other = max([v for k, v in all_scores.items() if k != 'neutral'] or [0.0])
                    all_scores['neutral'] = max(0.0, 1.0 - max_other)

                    # Gate by quality to reduce residual false positives
                    if landmark_quality < 0.25: # Relaxed quality gate
                        dominant_emotion = 'Neutral'
                        emotion_confidence = 0.0
                        all_scores = {k: 0.0 for k in all_scores}
                        all_scores["neutral"] = 1.0

                    # Smooth emotion to reduce flicker
                    smoothed_emotion, smoothed_conf = smooth_emotion(dominant_emotion, emotion_confidence)

                    # Update current real-time data
                    current_realtime_data.update({
                        'recording': recording,
                        'elapsed': time.time() - start_time if recording and start_time else 0,
                        'action_units': processed_aus,
                        'emotion': smoothed_emotion,
                        'emotion_confidence': smoothed_conf,
                        'emotion_scores': (all_scores if isinstance(all_scores, dict) else None),
                        'frame_count': current_realtime_data.get('frame_count', 0) + 1,
                        'last_update': time.time(),
                        'landmark_quality': landmark_quality
                    })

                    # Append to per-session timeline (bounded)
                    try:
                        if recording and start_time:
                            t = max(0.0, float(time.time() - start_time))
                            entry = {
                                't': round(t, 2),
                                'label': str(smoothed_emotion),
                                'confidence': float(smoothed_conf),
                                'quality': float(landmark_quality),
                                'scores': all_scores if isinstance(all_scores, dict) else None,
                                # Include current AU snapshot for AU-over-time charting
                                'aus': processed_aus if isinstance(processed_aus, dict) else None,
                            }
                            current_timeline.append(entry)
                            # Keep reasonable bound to avoid unbounded growth
                            if len(current_timeline) > 2000:
                                del current_timeline[:len(current_timeline) - 2000]
                    except Exception:
                        pass
                    
                    print(f"analysis processed: {len(processed_aus)} AUs, quality: {landmark_quality:.2f}, emotion: {smoothed_emotion}")
                    
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
            else:
                print(f"Analysis bridge(s) failed. Last error: {last_err}")
                
        except subprocess.TimeoutExpired:
            print("Analysis processing timeout")
        except Exception as e:
            import traceback
            print(f"Analysis subprocess error: {e}")
            print(traceback.format_exc())
        
        # Clean up temp file
        try:
            os.unlink(temp_frame.name)
        except:
            pass
            
    except Exception as e:
        print(f"Real-time processing error: {e}")
    
    return frame

def _ensure_analysis_thread():
    global analysis_thread
    if analysis_thread and analysis_thread.is_alive():
        return
    analysis_stop.clear()

    def worker():
        while not analysis_stop.is_set():
            if analysis_queue:
                try:
                    frame = analysis_queue.pop()
                except Exception:
                    frame = None
                if frame is not None:
                    try:
                        process_frame_realtime(frame)
                    except Exception as _e:
                        print(f"Analysis worker error: {_e}")
                    continue
            # Avoid busy spin
            time.sleep(0.01)

    analysis_thread = Thread(target=worker, name='analysis-worker', daemon=True)
    analysis_thread.start()

@app.route('/analyze_audio', methods=['POST'])
def analyze_audio():
    """Handle audio analysis - preserving existing functionality"""
    try:
        # Debug: Log API key presence
        groq_key = app.config.get('GROQ_API_KEY')
        print(f"[DEBUG] GROQ_API_KEY present: {bool(groq_key)}, length: {len(groq_key) if groq_key else 0}")
        
        # Accept both 'audio_file' (frontend upload) and 'audio' (legacy)
        upload = None
        if 'audio_file' in request.files:
            upload = request.files['audio_file']
        elif 'audio' in request.files:
            upload = request.files['audio']

        if upload:
            # File upload analysis
            if upload.filename and upload.filename.strip():
                filename = secure_filename(upload.filename)
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                upload.save(filepath)
                try:
                    # Use the integrated audio processor
                    result = process_audio_file(filepath, app.config)
                    print(f"[DEBUG] process_audio_file result: {result}")
                    return jsonify({'success': True, 'result': result})
                finally:
                    try:
                        os.remove(filepath)
                    except Exception:
                        pass
        
        # Real-time recording analysis (placeholder for now)
        data = request.get_json(silent=True) or {}
        if 'recording_data' in data:
            # Real-time recording analysis - simplified for demo
            return jsonify({
                'success': True, 
                'result': {
                    'transcription': 'Live recording transcription...',
                    'emotion_analysis': 'Positive emotions detected',
                    'sentiment': 'Happy',
                    'confidence': 0.85
                }
            })
        
        return jsonify({'success': False, 'error': 'No audio data provided'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def _open_camera(device_index: int = 0):
    """Try to open a camera with sensible defaults on macOS. Returns (cap, backend_name)."""
    backend = 'default'
    cap = None
    try:
        # Prefer AVFoundation first on macOS for stability
        if sys.platform == 'darwin':
            cap = cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)
            backend = 'AVFOUNDATION'
        else:
            cap = cv2.VideoCapture(device_index)
    except Exception:
        cap = None

    # Fallback to default if first attempt failed
    if not cap or not cap.isOpened():
        try:
            if cap:
                cap.release()
        except Exception:
            pass
        try:
            cap = cv2.VideoCapture(device_index)
            backend = 'default'
        except Exception:
            cap = None
    return cap, backend

def _error_frame(message: str, width: int = 640, height: int = 480):
    """Create a JPEG-encoded error frame with the given message."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    cv2.putText(img, 'Camera unavailable', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    y = 120
    for line in message.split('\n')[:5]:
        cv2.putText(img, line[:60], (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
        y += 30
    ok, buf = cv2.imencode('.jpg', img)
    return buf.tobytes() if ok else b''

def generate_video_feed(device_index: int = 0):
    """Generate video feed for live preview"""
    global recording, video_writer, start_time, duration, recorded_frames, video_file_path
    
    cap, backend = _open_camera(device_index)
    if not cap or not cap.isOpened():
        err = f"Failed to open camera index {device_index}"
        print(err)
        frame_bytes = _error_frame(err + "\n- Close other apps using the camera\n- Check macOS > Privacy & Security > Camera\n- Try another device index (1, 2)\n- If on VM/remote, camera may be unavailable")
        # Stream a few error frames so the <img> shows feedback
        for _ in range(60):
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return

    # Configure capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 20)
    
    try:
        frame_counter = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Camera read failed; sending placeholder frame")
                frame_bytes = _error_frame("Camera read failed — device disconnected?")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.5)
                continue
            
            # Enqueue frame for background analysis at a controlled rate
            if recording and frame_counter % 5 == 0:
                if len(analysis_queue) == 0:
                    try:
                        analysis_queue.append(frame.copy())
                    except Exception:
                        pass
            
            frame_counter += 1
            
            # Add recording indicator if recording
            if recording and frame is not None:
                elapsed = time.time() - start_time
                remaining = duration - elapsed
                
                # Recording indicator
                cv2.circle(frame, (frame.shape[1] - 50, 50), 20, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (frame.shape[1] - 65, 57), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Timer
                cv2.putText(frame, f"{elapsed:.1f}s / {duration}s", 
                           (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Write frame if recording
                # Lazily initialize the writer using the first valid frame to avoid backend segfaults
                if video_writer is None:
                    try:
                        h, w = frame.shape[:2]
                        # Try H.264 codec first (most compatible), fallback to MJPG
                        try:
                            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
                            video_writer = cv2.VideoWriter(video_file_path, fourcc, 20, (w, h))
                            if not video_writer.isOpened():
                                raise Exception("H.264 (avc1) failed to open")
                            print(f"VideoWriter initialized with H.264 codec: {w}x{h} @ 20fps")
                        except:
                            # Fallback to MJPG (most reliable)
                            try:
                                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                                video_writer = cv2.VideoWriter(video_file_path, fourcc, 20, (w, h))
                                if not video_writer.isOpened():
                                    raise Exception("MJPG codec failed")
                                print(f"VideoWriter initialized with MJPG codec: {w}x{h} @ 20fps")
                            except:
                                # Last resort: use default codec (-1)
                                video_writer = cv2.VideoWriter(video_file_path, -1, 20, (w, h))
                                if not video_writer.isOpened():
                                    raise Exception("Default codec also failed")
                                print(f"VideoWriter initialized with default codec: {w}x{h} @ 20fps")
                    except Exception as _e:
                        print(f"VideoWriter init error: {_e}")
                        video_writer = None
                
                if video_writer is not None and video_writer.isOpened():
                    try:
                        video_writer.write(frame)
                        recorded_frames += 1
                    except Exception as _e:
                        print(f"VideoWriter write error: {_e}")
                
                # Stop recording if duration reached
                if elapsed >= duration:
                    stop_recording()
            
            # Encode frame (fallback to placeholder if encoding fails)
            if frame is None:
                frame_bytes = _error_frame("No frame available for encoding")
            else:
                ok, buffer = cv2.imencode('.jpg', frame)
                if not ok:
                    frame_bytes = _error_frame("Encoding error")
                else:
                    frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
    finally:
        cap.release()

@app.route('/analyze_browser_frame', methods=['POST'])
def analyze_browser_frame():
    """
    Analyze a frame sent from browser webcam.
    Expects JSON: {"image": "<base64 jpeg>"}
    Returns AU and emotion data.
    """
    global current_realtime_data, current_timeline
    
    # Read ML_WORKER_URL at request time (not module load time)
    ml_worker_url = os.getenv('ML_WORKER_URL', '')
    emotion_backend = os.getenv('EMOTION_BACKEND', 'pyfeat').lower()
    
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        # If using ML Worker backend
        if emotion_backend == 'worker' and ml_worker_url:
            import requests as req
            try:
                resp = req.post(
                    f"{ml_worker_url}/analyze",
                    json={'image': data['image']},
                    timeout=60
                )
                if resp.status_code == 200:
                    result = resp.json()
                    if result.get('success') and result.get('face_detected'):
                        aus = result.get('aus', {})
                        emotions = result.get('emotions', {})
                        
                        # Update realtime data
                        current_realtime_data['action_units'] = aus
                        current_realtime_data['frame_count'] = current_realtime_data.get('frame_count', 0) + 1
                        
                        # Find dominant emotion
                        dom_emo = 'Neutral'
                        dom_conf = 0.0
                        if emotions:
                            dom_emo = max(emotions, key=emotions.get)
                            dom_conf = emotions[dom_emo]
                        
                        current_realtime_data['emotion'] = dom_emo.capitalize()
                        
                        return jsonify({
                            'success': True,
                            'face_detected': True,
                            'aus': aus,
                            'emotions': emotions,
                            'dominant_emotion': dom_emo.capitalize(),
                            'confidence': dom_conf
                        })
                    else:
                        return jsonify({
                            'success': True,
                            'face_detected': False,
                            'aus': {},
                            'emotions': {}
                        })
                else:
                    return jsonify({'success': False, 'error': f'ML Worker returned {resp.status_code}'}), 500
            except Exception as e:
                logging.error(f"ML Worker call failed: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        # Fallback: no ML worker configured
        return jsonify({
            'success': False,
            'error': f'No ML backend configured. backend={emotion_backend}, url={ml_worker_url[:20] if ml_worker_url else "empty"}',
            'backend': emotion_backend
        }), 503
        
    except Exception as e:
        logging.error(f"analyze_browser_frame error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    """Video streaming route - only works with server-side camera (local dev)"""
    # Allow selecting device index via query param
    try:
        device = int(request.args.get('device', '0'))
    except Exception:
        device = 0
    resp = Response(generate_video_feed(device),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    # Prevent caching to reduce freeze in some browsers
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp
                    
    # Note: set no-cache headers for better stability on some browsers
    

@app.route('/camera_status')
def camera_status():
    """Probe a few camera indices and report which ones are openable."""
    indices = [0, 1, 2, 3]
    available = []
    details = {}
    for idx in indices:
        cap, backend = _open_camera(idx)
        ok = bool(cap and cap.isOpened())
        details[idx] = {'open': ok, 'backend': backend}
        if cap:
            try:
                cap.release()
            except Exception:
                pass
        if ok:
            available.append(idx)
    return jsonify({
        'available_indices': available,
        'details': details,
        'default': 0,
    })

@app.route('/start_recording', methods=['POST'])
def start_recording():
    """Start video recording"""
    global recording, video_writer, video_file_path, start_time, duration, recorded_frames, current_realtime_data, current_timeline
    
    if recording:
        return jsonify({'success': False, 'error': 'Already recording'})
    
    try:
        # Get duration from request
        data = request.get_json()
        duration = data.get('duration', 10)

        # Preflight: ensure backend is available and responsive
        try:
            if EMOTION_BACKEND in ('pyfeat', 'both', 'auto'):
                # Call pyfeat bridge --check
                root = os.path.abspath(os.path.dirname(__file__))
                bridge_script = os.path.join(root, 'pyfeat_bridge.py')
                if not os.path.exists(bridge_script):
                    raise RuntimeError('pyfeat_bridge.py missing')
                # Allow generous timeout for initial model downloads
                proc = subprocess.run([sys.executable, bridge_script, '--check'], capture_output=True, text=True, timeout=30)
                if proc.returncode != 0:
                    raise RuntimeError(f'Py-Feat check failed (code {proc.returncode}). {proc.stderr[:200]}')
                data = json.loads(proc.stdout or '{}')
                if not data.get('success'):
                    raise RuntimeError(data.get('error') or 'Py-Feat not ready')
                try:
                    globals()['pyfeat_warmed_up'] = True
                except Exception:
                    pass
            # Always confirm OpenFace as well when available
            if EMOTION_BACKEND in ('openface', 'both', 'auto'):
                if not find_feature_extraction_binary:
                    raise RuntimeError('openface_bridge not importable')
                fe = find_feature_extraction_binary()
                if not fe:
                    raise RuntimeError('FeatureExtraction binary not found. Use Diagnostics → Check Backend.')
                env = _openface_env() if _openface_env else os.environ.copy()
                proc = subprocess.run([fe, '-h'], capture_output=True, text=True, timeout=3, env=env)
                if proc.returncode != 0:
                    raise RuntimeError(f'OpenFace not ready (code {proc.returncode}). See Diagnostics.\n{(proc.stderr or proc.stdout)[:200]}')
        except Exception as pre_err:
            return jsonify({'success': False, 'error': f'Face backend not ready: {pre_err}'}), 500
        
        # Reset real-time data for new recording
        current_realtime_data = {
            'recording': True,
            'elapsed': 0,
            'action_units': {},
            'emotion': 'Neutral',
            'frame_count': 0
        }
        # Reset timeline buffer for this session
        current_timeline = []
        
        # Create temporary video file (AVI container avoids mp4 PTS warnings)
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
        video_file_path = temp_video.name
        temp_video.close()
        
        # Defer VideoWriter init until first successful frame (avoids PTS and segfault issues)
        video_writer = None
        
        recording = True
        start_time = time.time()
        recorded_frames = 0
        _ensure_analysis_thread()
        
        return jsonify({
            'success': True, 
            'video_path': video_file_path,
            'duration': duration
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def stop_recording():
    """Stop video recording, finalize analysis, and save results."""
    global recording, video_writer, video_file_path, session_count, current_timeline, session_results
    
    if not recording:
        logging.warning("Stop recording called, but not currently recording.")
        return
    
    logging.info("Stopping recording and starting final analysis.")
    recording = False
    
    # Stop the analysis worker thread and wait for it to finish
    try:
        analysis_stop.set()
        if analysis_thread and analysis_thread.is_alive():
            logging.info("Waiting for analysis thread to join...")
            analysis_thread.join(timeout=2.0)
            if analysis_thread.is_alive():
                logging.warning("Analysis thread did not join in time.")
    except Exception as e:
        logging.error(f"Error stopping analysis thread: {e}", exc_info=True)

    # Release the video writer
    if video_writer is not None:
        logging.info(f"Releasing video writer for {video_file_path}")
        video_writer.release()
        video_writer = None
    
    # --- Final Analysis from Timeline ---
    if not current_timeline:
        logging.warning("No timeline data was captured. Cannot generate a session summary.")
        # Reset for next session
        current_timeline = []
        analysis_stop.clear()
        return

    # 1. Compute emotion summary and segments
    emotion_summary, emotion_segments = _compute_emotion_summary_and_segments(current_timeline)
    
    # 2. Compute AU summary
    au_analysis_result = analyze_recorded_video(video_file_path, current_timeline)

    # 3. Construct the final session result object
    session_id = f"session_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    session_result = {
        'session_id': session_id,
        'timestamp_utc': datetime.utcnow().isoformat() + 'Z',
        'video_file': os.path.basename(video_file_path) if video_file_path else None,
        'dominant_emotion': emotion_summary.get('dominant_emotion', 'Neutral'),
        'detection_confidence': emotion_summary.get('avg_confidence', 0.0),
        'emotion_summary': emotion_summary,
        'emotion_segments': emotion_segments,
        'au_summary': au_analysis_result.get('au_summary', {}),
        'timeline': current_timeline,
    }

    # 4. Save the result to a JSON file
    session_filename = os.path.join(app.config['SESSIONS_DIR'], f"{session_id}.json")
    try:
        with open(session_filename, 'w') as f:
            json.dump(session_result, f, indent=4)
        logging.info(f"Successfully saved session summary to {session_filename}")
    except Exception as e:
        logging.error(f"Failed to save session JSON to {session_filename}: {e}", exc_info=True)

    # 5. Append to in-memory list of sessions
    session_results.append(session_result)
    session_count += 1
    
    # 6. Reset state for the next recording
    current_timeline = []
    video_file_path = None
    analysis_stop.clear() # Reset the stop event
    
    logging.info("Recording stopped and final analysis complete.")

@app.route('/stop_recording', methods=['POST'])
def stop_recording_route():
    """Stop recording endpoint"""
    global recording
    
    if not recording:
        return jsonify({'success': False, 'error': 'Not recording'})
    
    stop_recording()
    return jsonify({'success': True})

def analyze_recorded_video(video_path: str, timeline: list) -> dict:
    """Analyzes the timeline data to produce a summary of Action Units (AUs).
    The emotion analysis is now handled by _compute_emotion_summary_and_segments.
    This function is kept for detailed AU summary if needed.
    
    Args:
        video_path (str): Path to the video file (currently unused, kept for API consistency).
        timeline (list): The timeline data captured during the session.
        
    Returns:
        dict: A dictionary containing the summary of Action Units.
    """
    logging.info(f"Starting AU summary analysis for timeline with {len(timeline)} entries.")
    if not timeline:
        return {'au_summary': {}}

    # --- AU Summary Calculation from Timeline ---
    au_intensity = defaultdict(float)
    au_presence = defaultdict(int)
    valid_frames = 0

    for entry in timeline:
        aus = entry.get('aus')
        if isinstance(aus, dict):
            valid_frames += 1
            # Summarize intensity (r) and presence (c)
            for key, value in aus.items():
                if key.endswith('_r'): # Intensity
                    au_intensity[key] += float(value)
                elif key.endswith('_c'): # Presence
                    if float(value) > 0.5: # Consider present if confidence > 0.5
                        au_presence[key] += 1
    
    if valid_frames == 0:
        logging.warning("No valid AU data found in timeline for summary.")
        return {'au_summary': {}}

    avg_au_intensity = {key: val / valid_frames for key, val in au_intensity.items()}
    percent_au_presence = {key: (val / valid_frames) * 100.0 for key, val in au_presence.items()}

    # Get top 5 most intense and most frequent AUs
    top_5_intensity = sorted(avg_au_intensity.items(), key=lambda item: item[1], reverse=True)[:5]
    top_5_presence = sorted(percent_au_presence.items(), key=lambda item: item[1], reverse=True)[:5]

    au_summary = {
        'avg_intensity': {k: round(v, 3) for k, v in avg_au_intensity.items()},
        'percent_presence': {k: round(v, 1) for k, v in percent_au_presence.items()},
        'top_5_intensity': {k: round(v, 3) for k, v in top_5_intensity},
        'top_5_presence': {k: round(v, 1) for k, v in top_5_presence},
        'frames_with_au_data': valid_frames,
    }
    
    logging.info(f"Generated AU summary from {valid_frames} timeline entries.")
    
    return {'au_summary': au_summary}

@app.route('/export_last_session_csv')
def export_last_session_csv():
    """Export the last session's timeline as CSV (t,label,confidence,quality, per-emotion scores, AUs)."""
    try:
        if not session_results:
            return jsonify({'success': False, 'error': 'No sessions yet'}), 404
        last = session_results[-1]
        timeline = last.get('timeline') or []
        if not timeline:
            return jsonify({'success': False, 'error': 'No timeline available for last session'}), 404

        # Build columns: base + union of score keys + union of AU keys
        score_keys = set()
        au_keys = set()
        for e in timeline:
            s = e.get('scores') or {}
            if isinstance(s, dict):
                score_keys.update(s.keys())
            aus = e.get('aus') or {}
            if isinstance(aus, dict):
                au_keys.update(aus.keys())
        score_cols = sorted(list(score_keys))
        au_cols = sorted(list(au_keys))

        # Write to a temporary CSV file then send
        import csv as _csv
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        tmp_path = tmp.name
        tmp.close()
        with open(tmp_path, 'w', newline='') as f:
            writer = _csv.writer(f)
            header = ['t', 'label', 'confidence', 'quality'] + [f'score_{k}' for k in score_cols] + au_cols
            writer.writerow(header)
            for e in timeline:
                row = [e.get('t'), e.get('label'), e.get('confidence'), e.get('quality')]
                s = e.get('scores') or {}
                row.extend([s.get(k, '') for k in score_cols])
                aus = e.get('aus') or {}
                row.extend([aus.get(k, '') for k in au_cols])
                writer.writerow(row)
        # Use send_file to return as attachment
        return send_file(tmp_path, as_attachment=True, download_name='session_timeline.csv')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def get_dominant_emotion(emotions_list):
    """Get dominant emotion from aggregated results"""
    if not emotions_list:
        return 'Neutral'
    
    emotion_counts = {}
    for emotion_info in emotions_list:
        emotion = emotion_info.get('emotion', 'neutral')
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + emotion_info.get('count', 1)
    
    if emotion_counts:
        return max(emotion_counts.items(), key=lambda x: x[1])[0].title()
    else:
        return 'Neutral'

def create_fallback_result():
    """Create fallback result if analysis fails with comprehensive AU structure"""
    global session_count, duration
    
    # Generate realistic fallback values for all 17 Action Units
    action_units_fallback = {
        # Upper Face Action Units
        'AU01_Inner_Brow_Raiser': np.random.uniform(0.0, 0.4),
        'AU02_Outer_Brow_Raiser': np.random.uniform(0.0, 0.3),
        'AU04_Brow_Lowerer': np.random.uniform(0.0, 0.3),
        'AU05_Upper_Lid_Raiser': np.random.uniform(0.0, 0.3),
        'AU06_Cheek_Raiser': np.random.uniform(0.0, 0.5),
        'AU07_Lid_Tightener': np.random.uniform(0.0, 0.4),
        'AU09_Nose_Wrinkler': np.random.uniform(0.0, 0.3),
        'AU10_Upper_Lip_Raiser': np.random.uniform(0.0, 0.3),
        # Lower Face Action Units
        'AU12_Lip_Corner_Puller': np.random.uniform(0.1, 0.7),  # Smile - more likely
        'AU14_Dimpler': np.random.uniform(0.0, 0.4),
        'AU15_Lip_Corner_Depressor': np.random.uniform(0.0, 0.3),
        'AU17_Chin_Raiser': np.random.uniform(0.0, 0.3),
        'AU20_Lip_Stretcher': np.random.uniform(0.0, 0.3),
        'AU23_Lip_Tightener': np.random.uniform(0.0, 0.3),
        'AU25_Lips_Part': np.random.uniform(0.0, 0.4),
        'AU26_Jaw_Drop': np.random.uniform(0.0, 0.4),
        'AU27_Mouth_Stretch': np.random.uniform(0.0, 0.3)
    }
    
    return {
        'session': session_count,
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'duration': duration,
        'frames_processed': duration * 20,
        'faces_detected': int(duration * 15),
        'action_units': action_units_fallback,
        'action_units_stats': {
            'total_aus_detected': len([au for au in action_units_fallback.values() if au > 0.1]),
            'max_intensity_au': max(action_units_fallback.items(), key=lambda x: x[1])[0],
            'average_intensity': sum(action_units_fallback.values()) / len(action_units_fallback)
        },
        'emotions': [
            {'emotion': 'Happy', 'confidence': np.random.uniform(0.3, 0.8)},
            {'emotion': 'Neutral', 'confidence': np.random.uniform(0.2, 0.7)},
            {'emotion': 'Focused', 'confidence': np.random.uniform(0.1, 0.5)}
        ],
        'dominant_emotion': np.random.choice(['Happy', 'Neutral', 'Focused', 'Calm']),
        'detection_confidence': np.random.uniform(0.6, 0.9),
        'quality_metrics': {
            'detection_rate': np.random.uniform(70, 95),
            'mean_confidence': np.random.uniform(0.7, 0.9),
            'landmark_stability': np.random.uniform(0.8, 0.95)
        },
        'analysis_metadata': {
            'backend': 'fallback_mode',
            'analysis_method': 'simulated_geometric',
            'total_landmarks': 468
        }
    }

@app.route('/get_recording_status')
def get_recording_status():
    """Get current recording status"""
    global recording, start_time, duration
    
    if recording and start_time:
        elapsed = time.time() - start_time
        return jsonify({
            'recording': recording,
            'elapsed': elapsed,
            'duration': duration,
            'progress': elapsed / duration if duration > 0 else 0
        })
    else:
        return jsonify({
            'recording': recording,
            'elapsed': 0,
            'duration': duration,
            'progress': 0
        })

@app.route('/get_realtime_data')
def get_realtime_data():
    """Get real-time Action Units data and emotion timeline from face analysis"""
    global current_realtime_data, recording, start_time, duration, current_timeline
    
    if not recording or not start_time:
        return jsonify({
            'recording': False,
            'elapsed': 0,
            'action_units': {},
            'emotion': 'Neutral',
            'emotion_scores': None,
            'timeline': [],
            'error': current_realtime_data.get('error')
        })
    
    elapsed = time.time() - start_time
    progress = min(elapsed / duration, 1.0) if duration > 0 else 0
    
    # Use real data from current_realtime_data
    action_units = current_realtime_data.get('action_units', {})
    
    # Process AU values for response
    processed_aus = {}
    for au_name, au_data in action_units.items():
        if isinstance(au_data, dict):
            processed_aus[au_name] = au_data.get('mean', 0)
        else:
            processed_aus[au_name] = au_data
    
    # Get recent timeline data (last 60 seconds or entire session if shorter)
    cutoff_time = elapsed - 60.0
    recent_timeline = [entry for entry in current_timeline if entry.get('t', 0) >= cutoff_time]
    
    # If no recent data, include all
    if not recent_timeline and current_timeline:
        recent_timeline = current_timeline
    
    # Extract AU names and compute top ones
    top_aus = sorted(processed_aus.items(), key=lambda x: x[1], reverse=True)[:8]
    
    return jsonify({
        'recording': True,
        'elapsed': elapsed,
        'progress': progress,
        'action_units': processed_aus,
        'au_count': len(processed_aus),
        'top_aus': dict(top_aus),  # Top 8 AUs for visualization
        'emotion': current_realtime_data.get('emotion', 'Neutral'),
        'emotion_confidence': current_realtime_data.get('emotion_confidence', 0),
        'emotion_scores': current_realtime_data.get('emotion_scores'),
        'landmark_quality': current_realtime_data.get('landmark_quality', None),
        'error': current_realtime_data.get('error'),
        'timestamp': time.time(),
        'frame_count': current_realtime_data.get('frame_count', 0),
        # Timeline data for charting
        'timeline': recent_timeline,
        'timeline_length': len(recent_timeline)
    })

@app.route('/get_session_summary')
def get_session_summary():
    """Get aggregated session summary: emotion distribution, AU statistics, mood changes"""
    global current_timeline, current_realtime_data
    
    if not current_timeline:
        return jsonify({
            'success': False,
            'error': 'No session data available',
            'summary': {}
        })
    
    try:
        # Emotion distribution
        emotion_counts = {}
        emotion_confidences = {}
        total_entries = len(current_timeline)
        
        for entry in current_timeline:
            emotion = entry.get('label', 'Neutral')
            confidence = entry.get('confidence', 0.0)
            
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            if emotion not in emotion_confidences:
                emotion_confidences[emotion] = []
            emotion_confidences[emotion].append(confidence)
        
        # Calculate emotion stats
        emotion_distribution = {}
        emotion_avg_confidence = {}
        for emotion, count in emotion_counts.items():
            emotion_distribution[emotion] = count / total_entries if total_entries > 0 else 0
            confs = emotion_confidences.get(emotion, [0])
            emotion_avg_confidence[emotion] = sum(confs) / len(confs) if confs else 0
        
        # AU statistics (mean, max, variance)
        au_stats = {}
        all_aus = {}  # Collect all AU timeseries
        
        for entry in current_timeline:
            aus = entry.get('aus', {})
            if isinstance(aus, dict):
                for au_name, au_value in aus.items():
                    if au_name not in all_aus:
                        all_aus[au_name] = []
                    all_aus[au_name].append(au_value)
        
        # Compute statistics for each AU
        for au_name, au_values in all_aus.items():
            if au_values:
                au_stats[au_name] = {
                    'mean': sum(au_values) / len(au_values),
                    'max': max(au_values),
                    'min': min(au_values),
                    'count': len(au_values)
                }
        
        # Mood change count (transitions between different emotions)
        mood_changes = 0
        prev_emotion = None
        for entry in current_timeline:
            emotion = entry.get('label', 'Neutral')
            if prev_emotion is not None and prev_emotion != emotion:
                mood_changes += 1
            prev_emotion = emotion
        
        # Quality metrics
        quality_scores = [entry.get('quality', 0) for entry in current_timeline]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        low_quality_count = sum(1 for q in quality_scores if q < 0.3)
        
        # Dominant emotion
        dominant_emotion = max(emotion_distribution.items(), key=lambda x: x[1])[0] if emotion_distribution else 'Neutral'
        
        # Time range
        if current_timeline:
            duration_secs = current_timeline[-1].get('t', 0) - current_timeline[0].get('t', 0)
        else:
            duration_secs = 0
        
        summary = {
            'emotion_distribution': emotion_distribution,
            'emotion_avg_confidence': emotion_avg_confidence,
            'dominant_emotion': dominant_emotion,
            'au_statistics': au_stats,
            'mood_changes': mood_changes,
            'total_frames': total_entries,
            'average_quality': avg_quality,
            'low_quality_frames': low_quality_count,
            'duration_seconds': round(duration_secs, 2),
            'top_aus_overall': dict(sorted(
                [(name, stats['mean']) for name, stats in au_stats.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10])
        }
        
        return jsonify({
            'success': True,
            'summary': summary,
            'timestamp': time.time()
        })
    
    except Exception as e:
        import traceback
        print(f"Session summary error: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'summary': {}
        })

@app.route('/get_timeline_export')
def get_timeline_export():
    """Export timeline data as CSV for analysis"""
    global current_timeline
    
    if not current_timeline:
        return jsonify({
            'success': False,
            'error': 'No timeline data to export'
        }), 404
    
    try:
        # Create CSV-like data
        csv_data = "time_s,emotion,confidence,quality"
        
        # Add all emotion keys if they exist
        all_emotion_keys = set()
        all_au_keys = set()
        for entry in current_timeline:
            if isinstance(entry.get('scores'), dict):
                all_emotion_keys.update(entry['scores'].keys())
            if isinstance(entry.get('aus'), dict):
                all_au_keys.update(entry['aus'].keys())
        
        # Build CSV header
        for key in sorted(all_emotion_keys):
            csv_data += f",emotion_score_{key}"
        for key in sorted(all_au_keys):
            csv_data += f",au_{key}"
        
        csv_data += "\n"
        
        # Add data rows
        for entry in current_timeline:
            t = entry.get('t', 0)
            emotion = entry.get('label', 'Neutral')
            conf = entry.get('confidence', 0)
            quality = entry.get('quality', 0)
            
            csv_data += f"{t},{emotion},{conf:.3f},{quality:.3f}"
            
            # Add emotion scores
            scores = entry.get('scores', {})
            for key in sorted(all_emotion_keys):
                score = scores.get(key, 0) if isinstance(scores, dict) else 0
                csv_data += f",{score:.3f}"
            
            # Add AU values
            aus = entry.get('aus', {})
            for key in sorted(all_au_keys):
                au_val = aus.get(key, 0) if isinstance(aus, dict) else 0
                csv_data += f",{au_val:.3f}"
            
            csv_data += "\n"
        
        # Return as downloadable file
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=emotion_au_timeline.csv"}
        )
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/face_backend_status')
def face_backend_status():
    """Report the selected face analysis backend and whether it's available."""
    backend = EMOTION_BACKEND
    available = False
    note = None
    detail = None
    path = None
    try:
        if backend == 'pyfeat':
            if pyfeat_check:
                res = pyfeat_check()
                available = bool(res.get('success'))
                note = res.get('error')
            else:
                # Fallback: run --check via subprocess
                root = os.path.abspath(os.path.dirname(__file__))
                bridge_script = os.path.join(root, 'pyfeat_bridge.py')
                if os.path.exists(bridge_script):
                    proc = subprocess.run([sys.executable, bridge_script, '--check'], capture_output=True, text=True, timeout=5)
                    available = (proc.returncode == 0) and ('"success": true' in (proc.stdout or '').lower())
                    detail = (proc.stdout or proc.stderr or '')[:4000]
                else:
                    note = 'pyfeat_bridge.py not found'
        else:
            if find_feature_extraction_binary:
                path = find_feature_extraction_binary()
                available = bool(path and os.path.exists(path) and os.access(path, os.X_OK))
            else:
                note = 'openface_bridge not importable'
    except Exception as e:
        note = str(e)
    return jsonify({
        'backend': backend,
        'feature_extraction_path': path,
        'available': available,
        'note': note,
        'detail': detail,
    })

@app.route('/check_backend')
def check_backend():
    """Run a quick backend check and return diagnostic details."""
    if EMOTION_BACKEND == 'pyfeat':
        try:
            root = os.path.abspath(os.path.dirname(__file__))
            bridge_script = os.path.join(root, 'pyfeat_bridge.py')
            if not os.path.exists(bridge_script):
                return jsonify({'success': False, 'error': 'pyfeat_bridge.py not found'}), 404
            proc = subprocess.run([sys.executable, bridge_script, '--check'], capture_output=True, text=True, timeout=8)
            out = (proc.stdout or '').strip()
            err = (proc.stderr or '').strip()
            ok = False
            try:
                data = json.loads(out or '{}')
                ok = bool(data.get('success'))
            except Exception:
                ok = proc.returncode == 0
            return jsonify({
                'success': ok,
                'returncode': proc.returncode,
                'stdout': out[:4000],
                'stderr': err[:4000],
                'backend': 'pyfeat',
            })
        except subprocess.TimeoutExpired:
            return jsonify({'success': False, 'error': 'Timeout running py-feat check'}), 504
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    else:
        try:
            if not find_feature_extraction_binary:
                return jsonify({'success': False, 'error': 'openface_bridge not importable'}), 500
            fe = find_feature_extraction_binary()
            if not fe:
                return jsonify({'success': False, 'error': 'FeatureExtraction binary not found. Install OpenFace or set OPENFACE_BIN.'}), 404
            proc = subprocess.run([fe, '-h'], capture_output=True, text=True, timeout=5)
            out = (proc.stdout or '').strip()
            err = (proc.stderr or '').strip()
            return jsonify({
                'success': proc.returncode == 0,
                'returncode': proc.returncode,
                'stdout': out[:4000],
                'stderr': err[:4000],
                'path': fe,
                'backend': 'openface',
            })
        except subprocess.TimeoutExpired:
            return jsonify({'success': False, 'error': 'Timeout running FeatureExtraction -h'}), 504
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/openface_install_help')
def openface_install_help():
    """Suggest macOS setup commands for OpenFace based on current environment and error messages."""
    suggestions = []
    notes = []
    detected_path = None
    sample_out = {'returncode': None, 'stdout': '', 'stderr': ''}
    try:
        if not find_feature_extraction_binary:
            raise RuntimeError('openface_bridge not importable')
        fe = find_feature_extraction_binary()
        if not fe:
            notes.append('FeatureExtraction binary not found. If you have a local build, set OPENFACE_BIN to its path.')
        else:
            detected_path = fe
            env = _openface_env() if '_openface_env' in globals() and _openface_env else os.environ.copy()
            try:
                proc = subprocess.run([fe, '-h'], capture_output=True, text=True, timeout=4, env=env)
                sample_out = {
                    'returncode': proc.returncode,
                    'stdout': (proc.stdout or '')[:2000],
                    'stderr': (proc.stderr or '')[:2000]
                }
            except Exception as sub_e:
                notes.append(f'Error invoking FeatureExtraction: {sub_e}')

        # Base brew installs
        suggestions.append('brew install boost opencv')
        # DYLD path exports
        suggestions.append('export DYLD_LIBRARY_PATH="/opt/homebrew/opt/boost/lib:/opt/homebrew/opt/opencv/lib:${DYLD_LIBRARY_PATH}"')
        suggestions.append('export DYLD_LIBRARY_PATH="/usr/local/opt/boost/lib:/usr/local/opt/opencv/lib:${DYLD_LIBRARY_PATH}"  # Intel macs')

        # Recommend OPENFACE_BIN if we detected a repo-local build
        repo_local = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'OpenFace', 'build', 'bin', 'FeatureExtraction')
        if detected_path and detected_path == repo_local:
            suggestions.append(f'export OPENFACE_BIN="{repo_local}"')
        elif detected_path:
            # still show a hint so users can persist the choice
            suggestions.append(f'# Optional: persist current binary path\nexport OPENFACE_BIN="{detected_path}"')

        # Tailor suggestions based on stderr hints
        err = (sample_out.get('stderr') or '') + '\n' + (sample_out.get('stdout') or '')
        if 'libboost' in err or 'boost' in err.lower():
            notes.append('Boost runtime appears missing; brew install boost should resolve libboost_*.dylib errors.')
        if 'libopencv' in err or 'opencv' in err.lower():
            notes.append('OpenCV runtime appears missing; brew install opencv should resolve libopencv_*.dylib errors.')

        return jsonify({
            'success': True,
            'feature_extraction_path': detected_path,
            'commands': suggestions,
            'notes': notes,
            'sample': sample_out,
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'commands': ['brew install boost opencv'],
        }), 500

@app.route('/get_results')
def get_results():
    """Get analysis results"""
    return jsonify({'results': session_results})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video file upload for analysis"""
    try:
        if 'video_file' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'})
        
        video_file = request.files['video_file']
        if video_file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save uploaded file
        filename = secure_filename(video_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(filepath)
        
        # Analyze the video
        analyze_recorded_video(filepath)
        
        return jsonify({'success': True, 'message': 'Video analyzed successfully'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# =================================================================
# Audio Analysis - Integrated
# =================================================================

class AudioAnalyzer:
    """Main class for audio analysis operations using Groq API"""
    
    def __init__(self, app_config):
        self.config = app_config
        # Try config first, then fall back to direct env var read
        groq_key = self.config.get('GROQ_API_KEY') or os.getenv('GROQ_API_KEY')
        print(f"[DEBUG AudioAnalyzer] groq_key present: {bool(groq_key)}, len={len(groq_key) if groq_key else 0}")
        try:
            self.groq_client = Groq(api_key=groq_key) if groq_key else None
            print(f"[DEBUG AudioAnalyzer] groq_client created: {bool(self.groq_client)}")
        except Exception as e:
            print(f"[DEBUG AudioAnalyzer] Groq client error: {e}")
            self.groq_client = None
        # Dev fallback flag: enable if no Groq key or when AUDIO_DEV_MODE is truthy
        self.dev_mode = (not groq_key) or (str(os.getenv('AUDIO_DEV_MODE', '0')).lower() in ('1','true','yes'))
        print(f"[DEBUG AudioAnalyzer] dev_mode: {self.dev_mode}")

    def _dev_fallback_analysis(self, transcript: Optional[str]) -> dict:
        t = transcript or "This is a sample transcription used in development mode."
        return {
            "transcript": t,
            "primary_emotion": "Calm",
            "secondary_emotions": ["Content"],
            "mood_category": "Neutral",
            "energy_level": "Medium",
            "tone": "Casual",
            "confidence": 0.72,
            "stress_indicators": ["Even pace"],
            "emotional_intensity": 0.35,
            "key_phrases": ["demo output", "dev mode"],
            "overall_vibe": "Steady and neutral with mild positivity.",
            "explanation": "Development fallback: returning a mocked analysis due to missing API or quota limits."
        }
    
    def transcribe_with_groq(self, audio_path: str) -> str:
        """Uses Groq Whisper to transcribe audio files."""
        if not self.groq_client:
            if self.dev_mode:
                return "Development mode transcription placeholder."
            raise ValueError("Groq API key missing. Set GROQ_API_KEY for audio transcription.")

        try:
            with open(audio_path, "rb") as audio_file:
                response = self.groq_client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=audio_file
                )
            return response.text
        except Exception as e:
            msg = str(e).lower()
            if 'quota' in msg or '429' in msg or 'rate' in msg:
                return "Development mode transcription (quota exceeded)."
            if self.dev_mode:
                return "Development mode transcription (error fallback)."
            raise
    
    def analyze_emotions(self, transcript: str) -> dict:
        """Analyzes emotions and mood from transcript using Groq."""
        system_prompt = self._get_emotion_analysis_prompt()
        user_message = f"Transcript:\n\n{transcript}\n\nProvide comprehensive emotional analysis."
        
        if not self.groq_client:
            if self.dev_mode:
                return self._dev_fallback_analysis(transcript)
            raise ValueError("Groq API key missing. Set GROQ_API_KEY for audio analysis.")

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3
            )
            text = response.choices[0].message.content
            return self._parse_emotion_response(text, transcript)
        except Exception as e:
            msg = str(e).lower()
            if 'quota' in msg or '429' in msg or 'rate' in msg:
                return self._dev_fallback_analysis(transcript)
            if self.dev_mode:
                return self._dev_fallback_analysis(transcript)
            raise

    def _get_emotion_analysis_prompt(self) -> str:
        """Returns the system prompt for emotion analysis"""
        return """You are an expert AI audio analyst. Analyze the provided transcript and return a JSON object with these fields:
- transcript: The original text
- primary_emotion: Main emotion (e.g., "Happy", "Sad", "Angry", "Calm")
- secondary_emotions: List of up to 2 other emotions
- mood_category: Overall mood ("Positive", "Negative", "Neutral", "Mixed")
- energy_level: "High", "Medium", or "Low"
- tone: Communication tone (e.g., "Formal", "Casual", "Professional")
- confidence: Float 0.0-1.0
- stress_indicators: List of stress signs
- emotional_intensity: Float 0.0-1.0
- key_phrases: List of 2-3 key phrases
- overall_vibe: One-sentence summary
- explanation: Brief justification with specific examples

Return ONLY valid JSON."""

    def _parse_emotion_response(self, text: str, original_transcript: str) -> dict:
        """Parse Groq response and return structured emotion data"""
        try:
            result = json.loads(text)
            # Always ensure transcript is included
            if 'transcript' not in result or not result['transcript']:
                result['transcript'] = original_transcript
            return result
        except json.JSONDecodeError:
            # Try to extract JSON block
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    result = json.loads(text[start:end+1])
                    # Always ensure transcript is included
                    if 'transcript' not in result or not result['transcript']:
                        result['transcript'] = original_transcript
                    return result
                except json.JSONDecodeError:
                    pass
            
            # Fallback to basic structure
            return {
                "transcript": original_transcript,
                "primary_emotion": "Unknown",
                "secondary_emotions": [],
                "mood_category": "Unknown",
                "energy_level": "Unknown",
                "tone": "Unknown",
                "confidence": 0.0,
                "stress_indicators": [],
                "emotional_intensity": 0.0,
                "key_phrases": [],
                "overall_vibe": "Could not analyze",
                "explanation": "Failed to parse emotion analysis response"
            }

def process_audio_file(filepath: str, app_config) -> dict:
    """
    Orchestrates the full audio analysis pipeline using Groq.
    """
    analyzer = AudioAnalyzer(app_config)

    # Prepare audio (optional resample/convert to 16k mono WAV using pydub if available)
    prepped_path = filepath
    tmp_to_cleanup = None

    def _prepare_audio_file(path: str) -> str:
        nonlocal tmp_to_cleanup
        try:
            from pydub import AudioSegment  # optional
            # Load any input format and convert
            seg = AudioSegment.from_file(path)
            seg = seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            fd, outpath = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            seg.export(outpath, format='wav')
            tmp_to_cleanup = outpath
            return outpath
        except Exception:
            return path

    prepped_path = _prepare_audio_file(filepath)

    try:
        # 1. Transcribe audio using Groq
        transcript = analyzer.transcribe_with_groq(prepped_path)

        # 2. Analyze emotions from transcript using Groq
        analysis = analyzer.analyze_emotions(transcript)

        return analysis

    except ValueError as e:
        # Handle specific errors like missing API keys or quota issues gracefully
        return {"error": str(e)}
    except Exception as e:
        # Catch-all for other unexpected errors
        return {"error": f"An unexpected error occurred: {e}"}
    finally:
        if tmp_to_cleanup and os.path.exists(tmp_to_cleanup):
            try:
                os.remove(tmp_to_cleanup)
            except Exception:
                pass


# =================================================================
# App setup and routes
# =================================================================

# --- Globals ---
video_writer = None
analysis_thread = None
analysis_queue = deque(maxlen=1)
emotion_history = deque(maxlen=10) # Rolling window for smoothing
last_analysis_time = 0

if __name__ == '__main__':
    # Create upload directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['SESSIONS_DIR'], exist_ok=True)

    # Respect env flags for debug/port and avoid reloader to keep a single stable process
    debug_flag = os.getenv('FLASK_DEBUG', os.getenv('DEBUG', '0')) in ('1', 'true', 'True')
    port = int(os.getenv('PORT', '5002'))
    app.run(host='0.0.0.0', port=port, debug=debug_flag, threaded=True, use_reloader=False)