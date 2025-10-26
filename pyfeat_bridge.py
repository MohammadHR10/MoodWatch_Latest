#!/usr/bin/env python3
"""
Py-Feat Bridge - Extract Action Units and emotion probabilities from an image.
Outputs JSON compatible with openface_bridge:
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
import traceback
from pathlib import Path
import cv2
import numpy as np

AU_NAME_FULL = {
    'AU01': 'AU01_Inner_Brow_Raiser',
    'AU02': 'AU02_Outer_Brow_Raiser',
    'AU04': 'AU04_Brow_Lowerer',
    'AU05': 'AU05_Upper_Lid_Raiser',
    'AU06': 'AU06_Cheek_Raiser',
    'AU07': 'AU07_Lid_Tightener',
    'AU09': 'AU09_Nose_Wrinkler',
    'AU10': 'AU10_Upper_Lip_Raiser',
    'AU12': 'AU12_Lip_Corner_Puller',
    'AU14': 'AU14_Dimpler',
    'AU15': 'AU15_Lip_Corner_Depressor',
    'AU17': 'AU17_Chin_Raiser',
    'AU20': 'AU20_Lip_Stretcher',
    'AU23': 'AU23_Lip_Tightener',
    'AU25': 'AU25_Lips_Part',
    'AU26': 'AU26_Jaw_Drop',
    'AU27': 'AU27_Mouth_Stretch',
}

EMO_LABEL_MAP = {
    'happiness': 'happiness',
    'happy': 'happiness',
    'anger': 'anger',
    'angry': 'anger',
    'sadness': 'sadness',
    'sad': 'sadness',
    'surprise': 'surprise',
    'fear': 'fear',
    'disgust': 'disgust',
    'neutral': 'neutral',
}


def _load_detector():
    try:
        from feat import Detector
    except Exception as e:
        raise RuntimeError(f"py-feat not installed or import failed: {e}")
    # Configure with robust defaults and layered fallbacks for older py-feat versions
    # Tier 1: retinaface + mobilefacenet + xgb + resmasknet
    try:
        return Detector(
            face_model="retinaface",
            landmark_model="mobilefacenet",
            au_model="xgb",
            emotion_model="resmasknet",
            device="cpu",
        )
    except Exception:
        pass
    # Tier 2: retinaface + mobilefacenet + rf + resmasknet
    try:
        return Detector(
            face_model="retinaface",
            landmark_model="mobilefacenet",
            au_model="rf",
            emotion_model="resmasknet",
            device="cpu",
        )
    except Exception:
        pass
    # Tier 3: mediapipe (widely available) + rf; emotion model best-effort
    try:
        return Detector(
            face_model="mediapipe",
            landmark_model="mobilefacenet",
            au_model="rf",
            emotion_model="resmasknet",
            device="cpu",
        )
    except Exception:
        # Final fallback minimal config
        return Detector(
            face_model="mediapipe",
            au_model="rf",
            device="cpu",
        )
    return det


def _read_and_preprocess(image_path: Path):
    # Load and apply light normalization (LAB histogram equalization)
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError('Failed to read image')
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception:
        pass
    return img


def analyze_with_pyfeat(image_path: Path):
    try:
        det = _load_detector()
    except Exception as e:
        return { 'error': str(e) }

    try:
        # Detect on preprocessed image; some py-feat versions expect a file-like/path
        # for detect_image. To be compatible across versions, write a temp PNG and pass its path.
        img = _read_and_preprocess(image_path)
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        try:
            cv2.imwrite(tmp.name, img)
            tmp.flush()
            tmp.close()
            fex = det.detect_image(tmp.name)
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
    except Exception as e:
        return { 'error': f"py-feat detection failed: {e}" }

    try:
        if fex is None or len(fex) == 0:
            return { 'error': 'No face detected' }

        row = fex.iloc[0]

        # Extract AU intensities (columns often like AU01, AU02, ...)
        aus = {}
        for short, full in AU_NAME_FULL.items():
            val = row.get(short)
            if val is None:
                # Some versions use intensity suffixes; try alternatives
                val = row.get(f"{short}_r") or row.get(f"{short}_intensity")
            try:
                v = float(val) if val is not None else 0.0
                if np.isnan(v):
                    v = 0.0
                # Py-Feat intensities typically in 0..1 already; clamp
                v = max(0.0, min(1.0, v))
            except Exception:
                v = 0.0
            aus[full] = v

    # Extract emotion scores (columns e.g. anger, disgust, fear, happiness, sadness, surprise, neutral)
        scores = {}
        for k in EMO_LABEL_MAP.keys():
            # Build unique keys to probe; but dataframe uses canonical names
            pass
        # Prefer canonical keys
        for key in ['anger','disgust','fear','happiness','sadness','surprise','neutral']:
            if key in fex.columns:
                try:
                    val = row.get(key)
                    v = float(val) if val is not None else 0.0
                    if np.isnan(v):
                        v = 0.0
                    scores[key] = v
                except Exception:
                    scores[key] = 0.0
        if not scores:
            # Attempt alternative prefix
            for key in ['anger','disgust','fear','happiness','sadness','surprise','neutral']:
                try:
                    val = row.get(f"emotion_{key}")
                    v = float(val) if val is not None else 0.0
                    if np.isnan(v):
                        v = 0.0
                    scores[key] = v
                except Exception:
                    scores[key] = 0.0

        # Fallback: derive scores from AUs if emotion model wasn't available
        if not any(scores.values()):
            # Simple AU-based heuristic (same as in openface bridge)
            def g(name):
                return float(aus.get(AU_NAME_FULL.get(name, name), 0.0))
            au12, au6 = g('AU12'), g('AU06')
            au15, au1, au2 = g('AU15'), g('AU01'), g('AU02')
            au4, au5 = g('AU04'), g('AU05')
            au26, au9, au10 = g('AU26'), g('AU09'), g('AU10')
            scores = {
                'happiness': au12 * 0.7 + au6 * 0.3,
                'sadness': au15 * 0.6 + au1 * 0.4,
                'surprise': (au1 + au2) * 0.4 + au26 * 0.6,
                'anger': au4 * 0.5 + au15 * 0.5,
                'fear': (au1 + au4) * 0.5 + au5 * 0.3,
                'disgust': 0.4 + 0.6 * min(1.0, (au9 + au10) / 2) if (au9 > 0.4 and au10 > 0.3 and au12 < 0.2) else 0.0,
                'neutral': 0.0,
            }

        # Normalize scores to 0..1 and pick best
        total = sum(max(0.0, s) for s in scores.values()) or 1.0
        norm_scores = {k: max(0.0, min(1.0, s/total)) for k,s in scores.items()}
        emo, conf = max(norm_scores.items(), key=lambda x: x[1]) if norm_scores else ('neutral', 0.0)

        # Proxy quality: use detector face score if present
        quality = 0.0
        for cand in ['face_score','detection_confidence','confidence']:
            try:
                q = float(row.get(cand))
                if q and q > 0:
                    quality = max(quality, min(1.0, q))
            except Exception:
                pass
        # If not available, set conservative default
        if quality == 0.0:
            quality = 0.4

        return {
            'success': True,
            'action_units': aus,
            'emotions': {
                'emotion': emo,
                'confidence': float(conf),
                'is_significant': conf >= 0.12,
                'all_scores': norm_scores
            },
            'landmarks_stats': quality,
            'metadata': {
                'backend': 'py-feat'
            }
        }
    except Exception:
        return { 'error': 'py-feat parsing error', 'trace': traceback.format_exc()[:500] }


def check_backend():
    try:
        _ = _load_detector()
        return { 'success': True, 'backend': 'py-feat' }
    except Exception as e:
        return { 'success': False, 'backend': 'py-feat', 'error': str(e) }


def main():
    # CLI usage:
    # pyfeat_bridge.py --check
    # pyfeat_bridge.py <image_path>
    if len(sys.argv) == 2 and sys.argv[1] == '--check':
        print(json.dumps(check_backend(), indent=2))
        return
    if len(sys.argv) < 2:
        print(json.dumps({ 'error': 'Usage: pyfeat_bridge.py <image_path>' }))
        sys.exit(1)
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(json.dumps({ 'error': f'Image not found: {image_path}' }))
        sys.exit(1)
    res = analyze_with_pyfeat(image_path)
    print(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()
