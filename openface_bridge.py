#!/usr/bin/env python3
"""
OpenFace Bridge - Extract Action Units and a simple emotion estimate from an image using OpenFace CLI.
Requires OpenFace FeatureExtraction binary installed and available on PATH.
Outputs JSON similar to mediapipe_bridge single-image mode:
{
  success: bool,
  action_units: { AUxx_name: float in [0,1], ... },
  emotions: { emotion: str, confidence: float, is_significant: bool, all_scores: {...} },
  landmarks_stats: float (proxy quality if available)
}
"""
import os
import sys
import json
import csv
import tempfile
import subprocess
import shutil
from pathlib import Path

# Mapping from OpenFace AU columns (e.g., AU01_r intensity) to our AU names
AU_NAME_MAP = {
    'AU01_r': 'AU01_Inner_Brow_Raiser',
    'AU02_r': 'AU02_Outer_Brow_Raiser',
    'AU04_r': 'AU04_Brow_Lowerer',
    'AU05_r': 'AU05_Upper_Lid_Raiser',
    'AU06_r': 'AU06_Cheek_Raiser',
    'AU07_r': 'AU07_Lid_Tightener',
    'AU09_r': 'AU09_Nose_Wrinkler',
    'AU10_r': 'AU10_Upper_Lip_Raiser',
    'AU12_r': 'AU12_Lip_Corner_Puller',
    'AU14_r': 'AU14_Dimpler',
    'AU15_r': 'AU15_Lip_Corner_Depressor',
    'AU17_r': 'AU17_Chin_Raiser',
    'AU20_r': 'AU20_Lip_Stretcher',
    'AU23_r': 'AU23_Lip_Tightener',
    'AU25_r': 'AU25_Lips_Part',
    'AU26_r': 'AU26_Jaw_Drop',
    'AU27_r': 'AU27_Mouth_Stretch',
}

# OpenFace intensities are typically 0..5; normalize to 0..1
def normalize_intensity(v: float) -> float:
    try:
        return max(0.0, min(1.0, float(v) / 5.0))
    except Exception:
        return 0.0

# Simple emotion detection reusing similar rules as mediapipe bridge
def detect_emotion_from_aus(au):
    def g(k):
        return float(au.get(k, 0.0) or 0.0)
    au12, au6 = g('AU12_Lip_Corner_Puller'), g('AU06_Cheek_Raiser')
    au15, au1, au2 = g('AU15_Lip_Corner_Depressor'), g('AU01_Inner_Brow_Raiser'), g('AU02_Outer_Brow_Raiser')
    au4, au5 = g('AU04_Brow_Lowerer'), g('AU05_Upper_Lid_Raiser')
    au26, au9, au10 = g('AU26_Jaw_Drop'), g('AU09_Nose_Wrinkler'), g('AU10_Upper_Lip_Raiser')

    scores = {}
    scores['happiness'] = au12 * 0.7 + au6 * 0.3
    scores['sadness'] = au15 * 0.6 + au1 * 0.4
    scores['surprise'] = (au1 + au2) * 0.4 + au26 * 0.6
    scores['anger'] = au4 * 0.5 + au15 * 0.5
    scores['fear'] = (au1 + au4) * 0.5 + au5 * 0.3
    # Disgust needs AU9 strongly + AU10 support and not smiling
    disgust = 0.0
    if au9 > 0.4 and au10 > 0.3 and au12 < 0.2:
        disgust = 0.4 + 0.6 * min(1.0, (au9 + au10) / 2)
    scores['disgust'] = disgust

    # Pick max and scale confidence to be more responsive in UI
    emo, conf = max(scores.items(), key=lambda x: x[1])
    conf_scaled = min(1.0, conf * 1.5)
    is_sig = conf_scaled >= 0.18
    if not is_sig:
        return { 'emotion': 'neutral', 'confidence': 0.0, 'is_significant': False, 'all_scores': scores }
    return { 'emotion': emo, 'confidence': float(conf_scaled), 'is_significant': True, 'all_scores': scores }


def find_feature_extraction_binary():
    # 1) Env var override
    env_bin = os.getenv('OPENFACE_BIN')
    if env_bin and os.path.exists(env_bin) and os.access(env_bin, os.X_OK):
        return env_bin

    # 2) Try PATH first
    path = shutil.which('FeatureExtraction')
    if path:
        return path

    # 3) Common local build locations
    candidates = [
        '/opt/homebrew/bin/FeatureExtraction',
        '/usr/local/bin/FeatureExtraction',
        # Home-folder build
        os.path.expanduser('~/OpenFace/build/bin/FeatureExtraction'),
    ]

    # 4) Project-local build (OpenFace bundled in repo)
    try:
        root = Path(__file__).resolve().parent
        candidates.append(str(root / 'OpenFace' / 'build' / 'bin' / 'FeatureExtraction'))
    except Exception:
        pass

    for c in candidates:
        if c and os.path.exists(c) and os.access(c, os.X_OK):
            return c
    return None

def _augmented_env():
    """Return a copy of os.environ with DYLD_LIBRARY_PATH augmented for common macOS Homebrew installs.
    Honors colon-separated OPENFACE_LIB_DIRS if provided.
    """
    env = os.environ.copy()
    extra_dirs = []
    # User-provided additional lib dirs (colon-separated)
    user_dirs = env.get('OPENFACE_LIB_DIRS')
    if user_dirs:
        extra_dirs.extend([p for p in user_dirs.split(':') if p])
    # Common Homebrew locations (Apple Silicon and Intel)
    candidates = [
        '/opt/homebrew/opt/boost/lib',
        '/opt/homebrew/opt/boost@1.85/lib',
        '/opt/homebrew/opt/boost@1.76/lib',
        '/opt/homebrew/opt/opencv/lib',
        '/usr/local/opt/boost/lib',
        '/usr/local/opt/opencv/lib',
    ]
    for c in candidates:
        if os.path.isdir(c):
            extra_dirs.append(c)
    # Build DYLD_LIBRARY_PATH
    existing = env.get('DYLD_LIBRARY_PATH', '')
    parts = [p for p in existing.split(':') if p]
    for d in extra_dirs:
        if d not in parts:
            parts.append(d)
    if parts:
        env['DYLD_LIBRARY_PATH'] = ':'.join(parts)
    return env


def analyze_with_openface(image_path: Path):
    fe_bin = find_feature_extraction_binary()
    if not fe_bin:
        return { 'error': 'OpenFace FeatureExtraction binary not found on PATH. Install OpenFace and ensure FeatureExtraction is available.' }

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        try:
            cmd = [
                fe_bin,
                '-q',            # quiet
                '-f', str(image_path),
                '-aus',          # extract AUs
                '-out_dir', str(out_dir),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=8, env=_augmented_env())
            if proc.returncode != 0:
                return { 'error': f'OpenFace failed (code {proc.returncode})', 'stderr': proc.stderr[:500] }
        except Exception as e:
            return { 'error': f'Failed to run OpenFace: {e}' }

        # OpenFace writes a CSV with same base name
        csv_path = out_dir / (Path(image_path).stem + '.csv')
        if not csv_path.exists():
            # Some builds use "processed" prefix
            alt = out_dir / (Path(image_path).stem + '_of_details.txt')
            return { 'error': f'OpenFace output not found at {csv_path}' }

        # Parse first data row
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            row = None
            for r in reader:
                row = r
                break
            if not row:
                return { 'error': 'OpenFace CSV had no rows' }

        # Build AU dict
        aus = {}
        for col, name in AU_NAME_MAP.items():
            if col in row:
                aus[name] = normalize_intensity(row[col])

        # Proxy quality if available
        quality = 0.0
        if 'confidence' in row:
            try:
                c = float(row['confidence'])
                # OpenFace confidence is roughly [0..1]
                quality = max(0.0, min(1.0, c))
            except Exception:
                pass
        if 'success' in row:
            try:
                if int(float(row['success'])) == 0:
                    quality = min(quality, 0.3)
            except Exception:
                pass

        emotions = detect_emotion_from_aus(aus)
        return {
            'success': True,
            'action_units': aus,
            'emotions': emotions,
            'landmarks_stats': quality,
            'metadata': {
                'backend': 'openface',
            }
        }


def main():
    if len(sys.argv) < 2:
        print(json.dumps({ 'error': 'Usage: openface_bridge.py <image_path>' }))
        sys.exit(1)
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(json.dumps({ 'error': f'Image not found: {image_path}' }))
        sys.exit(1)
    res = analyze_with_openface(image_path)
    print(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()
