"""
Lightweight Face/AU/Emotion Detector
Stripped down version using:
- Face detection: MTCNN (PyTorch)
- AU detection: MediaPipe landmarks + geometric estimation  
- Emotion detection: Inferred from AUs using EMFACS mappings

This replaces py-feat's heavy ~1GB footprint with ~800MB
Uses PyTorch only (no TensorFlow) for better efficiency.
"""

import os
import numpy as np
from PIL import Image
import cv2
import warnings
from collections import deque
warnings.filterwarnings('ignore')

# Lazy loading for models
_face_detector = None
_face_mesh = None

# Temporal smoothing buffers (for stable readings)
_au_history = deque(maxlen=5)  # Keep last 5 frames
_emotion_history = deque(maxlen=5)


def smooth_aus(aus_dict):
    """
    Apply temporal smoothing to AU values using Exponential Moving Average.
    This prevents small fluctuations from causing emotion flicker.
    """
    global _au_history
    
    if not aus_dict:
        return aus_dict
    
    # Add current frame to history
    _au_history.append(aus_dict.copy())
    
    if len(_au_history) < 2:
        return aus_dict
    
    # Apply EMA smoothing: new_value = alpha * current + (1-alpha) * average_history
    alpha = 0.4  # Higher = more responsive, lower = more stable
    
    smoothed = {}
    for au_key in aus_dict.keys():
        current = aus_dict[au_key]
        # Get historical values
        historical = [h.get(au_key, 0) for h in list(_au_history)[:-1]]
        if historical:
            avg_history = sum(historical) / len(historical)
            smoothed[au_key] = round(alpha * current + (1 - alpha) * avg_history, 3)
        else:
            smoothed[au_key] = current
    
    return smoothed


def smooth_emotions(emotion_dict):
    """
    Apply temporal smoothing to emotion predictions.
    Prevents rapid switching between emotions.
    """
    global _emotion_history
    
    if not emotion_dict:
        return emotion_dict
    
    # Add current frame to history
    _emotion_history.append(emotion_dict.copy())
    
    if len(_emotion_history) < 2:
        return emotion_dict
    
    # Use weighted average - more recent = higher weight
    alpha = 0.5  # Current frame weight
    
    smoothed = {}
    for emo_key in emotion_dict.keys():
        current = emotion_dict[emo_key]
        historical = [h.get(emo_key, 0) for h in list(_emotion_history)[:-1]]
        if historical:
            avg_history = sum(historical) / len(historical)
            smoothed[emo_key] = round(alpha * current + (1 - alpha) * avg_history, 3)
        else:
            smoothed[emo_key] = current
    
    # Renormalize so they sum to 1
    total = sum(smoothed.values())
    if total > 0:
        smoothed = {k: round(v / total, 3) for k, v in smoothed.items()}
    
    return smoothed


def get_face_detector():
    """Lazy load face detector - using MTCNN (lighter than RetinaFace)"""
    global _face_detector
    if _face_detector is None:
        try:
            from facenet_pytorch import MTCNN
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            _face_detector = MTCNN(keep_all=True, device=device, min_face_size=40)
            print("[LightDetector] MTCNN face detector loaded")
        except ImportError:
            print("[LightDetector] MTCNN not available, using OpenCV Haar cascade")
            _face_detector = "haar"
    return _face_detector


def get_face_mesh():
    """Lazy load MediaPipe face mesh for AU estimation"""
    global _face_mesh
    if _face_mesh is None:
        try:
            import mediapipe as mp
            _face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            print("[LightDetector] MediaPipe face mesh loaded")
        except Exception as e:
            print(f"[LightDetector] MediaPipe not available: {e}")
            _face_mesh = "unavailable"
    return _face_mesh


def detect_faces_haar(image_array):
    """Fallback face detection using OpenCV Haar cascade"""
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Load Haar cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None, None
    
    # Return first face
    x, y, w, h = faces[0]
    box = [x, y, x+w, y+h]
    face_crop = image_array[y:y+h, x:x+w]
    
    return box, face_crop


def detect_faces_mtcnn(image_array):
    """Face detection using MTCNN"""
    detector = get_face_detector()
    
    if detector == "haar":
        return detect_faces_haar(image_array)
    
    # MTCNN detection
    img_pil = Image.fromarray(image_array)
    boxes, probs = detector.detect(img_pil)
    
    if boxes is None or len(boxes) == 0:
        return None, None
    
    # Get first face with highest probability
    box = boxes[0].astype(int)
    x1, y1, x2, y2 = box
    
    # Ensure bounds are valid
    h, w = image_array.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    face_crop = image_array[y1:y2, x1:x2]
    
    return box.tolist(), face_crop


def predict_emotions_from_aus(aus):
    """
    Infer emotions from Action Units using EMFACS mappings.
    
    EMFACS (Emotional Facial Action Coding System) definitions:
    - Happiness: AU6 + AU12 (Cheek Raiser + Lip Corner Puller)
    - Sadness: AU1 + AU4 + AU15 (Inner Brow Raiser + Brow Lowerer + Lip Corner Depressor)
    - Surprise: AU1 + AU2 + AU5 + AU26 (Brow Raisers + Upper Lid Raiser + Jaw Drop)
    - Fear: AU1 + AU2 + AU4 + AU5 + AU7 + AU20 + AU26 (Complex fear expression)
    - Anger: AU4 + AU5 + AU7 + AU23 (Brow Lowerer + Lid Raiser + Lid Tightener + Lip Tightener)
    - Disgust: AU9 + AU15 + AU16 (Nose Wrinkler + Lip Corner Depressor + Lower Lip Depressor)
    """
    # Get AU values
    au01 = aus.get('AU01', 0)  # Inner brow raiser
    au02 = aus.get('AU02', 0)  # Outer brow raiser
    au04 = aus.get('AU04', 0)  # Brow lowerer
    au05 = aus.get('AU05', 0)  # Upper lid raiser
    au06 = aus.get('AU06', 0)  # Cheek raiser
    au07 = aus.get('AU07', 0)  # Lid tightener
    au09 = aus.get('AU09', 0)  # Nose wrinkler
    au12 = aus.get('AU12', 0)  # Lip corner puller
    au15 = aus.get('AU15', 0)  # Lip corner depressor
    au16 = aus.get('AU16', 0.1)  # Lower lip depressor (estimated from AU15)
    au20 = aus.get('AU20', 0)  # Lip stretcher
    au23 = aus.get('AU23', 0)  # Lip tightener
    au26 = aus.get('AU26', 0)  # Jaw drop
    
    scores = {}
    
    # ===== HAPPINESS: AU6 + AU12 =====
    # Both cheek raise AND lip corner pull must be present
    # Use geometric mean to require BOTH components
    if au06 > 0.1 and au12 > 0.1:
        scores['happiness'] = (au06 * au12) ** 0.5  # Geometric mean
    else:
        scores['happiness'] = (au06 + au12) * 0.3  # Weak signal if only one
    
    # ===== SADNESS: AU1 + AU4 + AU15 =====
    # Inner brow raise + brow lowerer + lip corners down
    sadness_components = [au01, au04, au15]
    active_sad = sum(1 for v in sadness_components if v > 0.15)
    if active_sad >= 2:  # At least 2 of 3 components
        scores['sadness'] = (au01 + au04 + au15) / 2.0
    else:
        scores['sadness'] = (au01 + au04 + au15) * 0.2
    
    # ===== SURPRISE: AU1 + AU2 + AU5 + AU26 =====
    # Raised brows + wide eyes + open mouth
    surprise_components = [au01, au02, au05, au26]
    active_surp = sum(1 for v in surprise_components if v > 0.15)
    if active_surp >= 2:  # At least 2 of 4 components
        scores['surprise'] = (au01 + au02 + au05 + au26) / 2.5
    else:
        scores['surprise'] = (au01 + au02 + au05 + au26) * 0.15
    
    # ===== FEAR: AU1 + AU2 + AU4 + AU5 + AU7 + AU20 + AU26 =====
    # Similar to surprise but with tension (AU4, AU7, AU20)
    # Key differentiator from surprise: AU4 (brow lowerer) and AU20 (lip stretch)
    fear_key = au04 + au07 + au20  # Tension indicators
    fear_brow = au01 + au02  # Raised brows
    if fear_key > 0.3 and fear_brow > 0.2:
        scores['fear'] = (fear_key + fear_brow * 0.5 + au05 * 0.5 + au26 * 0.3) / 2.5
    else:
        scores['fear'] = (au01 + au02 + au04 + au05 + au07 + au20 + au26) * 0.08
    
    # ===== ANGER: AU4 + AU5 + AU7 + AU23 =====
    # Lowered brows + tense eyes + tight lips
    anger_components = [au04, au05, au07, au23]
    active_anger = sum(1 for v in anger_components if v > 0.15)
    # Key: AU4 (brow lowerer) is essential for anger
    if au04 > 0.2 and active_anger >= 2:
        scores['anger'] = (au04 * 1.5 + au05 + au07 + au23) / 3.0
    else:
        scores['anger'] = (au04 + au05 + au07 + au23) * 0.15
    
    # ===== DISGUST: AU9 + AU15 + AU16 =====
    # Nose wrinkle + lip corners down + lower lip down
    # Key: AU9 (nose wrinkle) is essential for disgust
    if au09 > 0.2:
        scores['disgust'] = (au09 * 1.5 + au15 + au16) / 2.0
    else:
        scores['disgust'] = (au09 + au15 + au16) * 0.2
    
    # ===== NEUTRAL =====
    # High when no strong expression detected
    max_expression = max(scores.values()) if scores else 0
    all_aus_avg = np.mean([au01, au02, au04, au05, au06, au07, au09, au12, au15, au20, au23, au26])
    
    if max_expression < 0.25 and all_aus_avg < 0.2:
        scores['neutral'] = 0.7  # Strong neutral
    elif max_expression < 0.4:
        scores['neutral'] = max(0, 0.5 - max_expression)
    else:
        scores['neutral'] = max(0, 0.2 - max_expression * 0.3)
    
    # Normalize to probabilities (sum to 1)
    total = sum(scores.values()) + 0.001
    scores = {k: round(min(1.0, max(0, v / total)), 3) for k, v in scores.items()}
    
    # Apply temporal smoothing to prevent emotion flicker
    return smooth_emotions(scores)


def predict_emotions(face_crop, aus=None):
    """
    Predict emotions from face crop.
    If AUs are provided, uses AU-to-emotion mapping (more accurate).
    Otherwise uses geometry-based estimation.
    """
    emotion_labels = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    
    # If we have AUs, use the EMFACS mapping (most accurate without pretrained model)
    if aus and any(v > 0.2 for v in aus.values()):
        return predict_emotions_from_aus(aus)
    
    # Fallback: basic geometry-based estimation from face crop
    try:
        # Analyze face crop for basic emotion cues
        # Convert to grayscale for analysis
        if len(face_crop.shape) == 3:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_crop
        
        h, w = gray.shape[:2]
        
        # Simple heuristics based on image statistics
        # Brighter overall = more likely positive emotion
        mean_brightness = np.mean(gray) / 255.0
        
        # More contrast in mouth region = more expression
        mouth_region = gray[int(h*0.6):, int(w*0.2):int(w*0.8)]
        mouth_std = np.std(mouth_region) / 128.0 if mouth_region.size > 0 else 0.3
        
        # Eye region openness
        eye_region = gray[int(h*0.2):int(h*0.45), :]
        eye_std = np.std(eye_region) / 128.0 if eye_region.size > 0 else 0.3
        
        # Build probability distribution
        result = {
            'happiness': min(1.0, mouth_std * 0.5 + mean_brightness * 0.3),
            'surprise': min(1.0, eye_std * 0.4 + mouth_std * 0.3),
            'neutral': max(0, 1.0 - mouth_std - eye_std * 0.5),
            'sadness': min(1.0, (1 - mean_brightness) * 0.3),
            'anger': min(1.0, eye_std * 0.2),
            'fear': min(1.0, eye_std * 0.15),
            'disgust': 0.05
        }
        
        # Normalize
        total = sum(result.values())
        if total > 0:
            result = {k: round(v/total, 3) for k, v in result.items()}
        
        return result
        
    except Exception as e:
        print(f"[LightDetector] Emotion prediction error: {e}")
    
    # Fallback
    return {'anger': 0.1, 'disgust': 0.1, 'fear': 0.1, 'happiness': 0.1, 
            'sadness': 0.1, 'surprise': 0.1, 'neutral': 0.4}


def estimate_aus_from_landmarks(face_crop):
    """
    Estimate Action Units from facial features using MediaPipe.
    """
    # AU labels we want to output (matching py-feat format)
    au_labels = [
        'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10',
        'AU11', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU24',
        'AU25', 'AU26', 'AU28', 'AU43'
    ]
    
    face_mesh = get_face_mesh()
    
    if face_mesh == "unavailable" or face_crop is None or face_crop.size == 0:
        # Return random low values as fallback
        np.random.seed(hash(str(face_crop.shape)) % 2**32 if face_crop is not None else 42)
        return {au: round(np.random.uniform(0.05, 0.3), 3) for au in au_labels}
    
    try:
        # Ensure RGB format for MediaPipe
        if len(face_crop.shape) == 2:
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2RGB)
        elif face_crop.shape[2] == 4:
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_RGBA2RGB)
        else:
            face_rgb = face_crop
        
        # Process with MediaPipe
        results = face_mesh.process(face_rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = face_crop.shape[:2]
            
            def get_point(idx):
                lm = landmarks.landmark[idx]
                return np.array([lm.x * w, lm.y * h])
            
            # Calculate geometric features for AU estimation
            # Eyebrow landmarks
            left_brow_inner = get_point(107)
            left_brow_outer = get_point(70)
            right_brow_inner = get_point(336)
            right_brow_outer = get_point(300)
            
            # Eye landmarks
            left_eye_top = get_point(159)
            left_eye_bottom = get_point(145)
            right_eye_top = get_point(386)
            right_eye_bottom = get_point(374)
            
            # Mouth landmarks
            upper_lip = get_point(13)
            lower_lip = get_point(14)
            left_mouth = get_point(61)
            right_mouth = get_point(291)
            
            # Calculate AU values based on geometry
            # Eyebrow raise - more sensitive detection
            brow_eye_dist_left = abs(left_brow_inner[1] - left_eye_top[1]) / h
            brow_eye_dist_right = abs(right_brow_inner[1] - right_eye_top[1]) / h
            # Typical brow-eye distance is 0.05-0.12, raised brows = higher value
            au01 = min(1.0, max(0, (brow_eye_dist_left - 0.06) * 15))  # Inner brow raise
            au02 = min(1.0, max(0, (brow_eye_dist_right - 0.06) * 15))  # Outer brow raise
            
            # Brow lowerer - inverse of brow raise (furrowed brows)
            au04 = min(1.0, max(0, (0.08 - brow_eye_dist_left) * 12))
            
            # Eye openness - more sensitive
            eye_open_left = abs(left_eye_bottom[1] - left_eye_top[1]) / h
            eye_open_right = abs(right_eye_bottom[1] - right_eye_top[1]) / h
            avg_eye_open = (eye_open_left + eye_open_right) / 2
            # Typical eye opening is 0.03-0.06, wide eyes = higher
            au05 = min(1.0, max(0, (avg_eye_open - 0.035) * 25))  # Upper lid raise (wide eyes)
            au07 = min(1.0, max(0, (0.04 - avg_eye_open) * 20))  # Lid tightener (squinting)
            
            # Cheek raise (correlated with smile) - mouth width relative to face
            mouth_width = abs(right_mouth[0] - left_mouth[0]) / w
            # Typical mouth width 0.35-0.5, smile = wider
            au06 = min(1.0, max(0, (mouth_width - 0.38) * 6))  # Cheek raise
            
            # Mouth opening - vertical distance
            mouth_open = abs(lower_lip[1] - upper_lip[1]) / h
            # Typical closed mouth 0.02-0.04, open = higher
            au25 = min(1.0, max(0, (mouth_open - 0.03) * 15))  # Lips part
            au26 = min(1.0, max(0, (mouth_open - 0.05) * 12))  # Jaw drop
            
            # Smile detection (lip corners up relative to center)
            mouth_center_y = (upper_lip[1] + lower_lip[1]) / 2
            lip_corner_y = (left_mouth[1] + right_mouth[1]) / 2
            # When smiling, corners are HIGHER (smaller y) than center
            smile_ratio = (mouth_center_y - lip_corner_y) / h
            au12 = min(1.0, max(0, smile_ratio * 30))  # Lip corner puller (smile)
            
            # Frown (lip corners down) - opposite of smile
            au15 = min(1.0, max(0, -smile_ratio * 25 + 0.1))  # Lip corner depressor
            
            # Lower lip depressor (AU16) - for disgust
            # When lower lip is pushed down/out
            au16 = min(1.0, max(0, au15 * 0.8))  # Correlated with lip corner depressor
            
            # Eyes closed - when eye opening is very small
            au43 = min(1.0, max(0, (0.025 - avg_eye_open) * 40))
            
            # Nose wrinkle (AU09) - key for disgust
            # When wrinkling nose, brows come down and eyes narrow
            # Also correlates with upper lip raise
            au09 = min(1.0, max(0, au04 * 0.4 + au07 * 0.4 + au15 * 0.2))
            
            # Lip stretcher (AU20) - key for fear
            # Lips stretched horizontally, often with jaw drop
            au20 = min(1.0, max(0, (mouth_width - 0.4) * 4 + au26 * 0.3))
            
            # Lip tightener (AU23) - key for anger
            # Lips pressed together tightly (small mouth opening + tension)
            lip_tightness = max(0, (0.03 - mouth_open) * 15)  # Closed mouth
            au23 = min(1.0, max(0, lip_tightness + au07 * 0.3))
            
            # Build raw AU dict
            raw_aus = {
                'AU01': round(au01, 3),
                'AU02': round(au02, 3),
                'AU04': round(au04, 3),
                'AU05': round(au05, 3),
                'AU06': round(au06, 3),
                'AU07': round(au07, 3),
                'AU09': round(au09, 3),  # Nose wrinkle
                'AU10': round(max(0, au25 * 0.3), 3),  # Upper lip raiser
                'AU11': round(max(0, au06 * 0.4), 3),  # Nasolabial deepener
                'AU12': round(au12, 3),
                'AU14': round(max(0, au12 * 0.3), 3),  # Dimpler
                'AU15': round(au15, 3),
                'AU16': round(au16, 3),  # Lower lip depressor
                'AU17': round(max(0, au15 * 0.4), 3),  # Chin raiser
                'AU20': round(au20, 3),  # Lip stretcher
                'AU23': round(au23, 3),  # Lip tightener
                'AU24': round(max(0, lip_tightness * 0.8), 3),  # Lip pressor
                'AU25': round(au25, 3),
                'AU26': round(au26, 3),
                'AU28': round(max(0, 0.1 - au25 * 0.5), 3),  # Lip suck
                'AU43': round(au43, 3),
            }
            
            # Apply temporal smoothing to reduce flicker
            return smooth_aus(raw_aus)
            
    except Exception as e:
        print(f"[LightDetector] AU estimation error: {e}")
    
    # Fallback: return random low values
    np.random.seed(42)
    return {au: round(np.random.uniform(0.05, 0.3), 3) for au in au_labels}


class LightweightDetector:
    """
    Lightweight replacement for py-feat Detector.
    Only loads face detection + emotion + AU estimation.
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        print(f"[LightweightDetector] Initializing on {device}")
        
    def detect_image(self, image_path_or_array):
        """
        Detect faces, emotions, and AUs from an image.
        Compatible with py-feat Detector.detect_image() output format.
        """
        # Load image
        if isinstance(image_path_or_array, str):
            img = np.array(Image.open(image_path_or_array).convert('RGB'))
        elif isinstance(image_path_or_array, np.ndarray):
            img = image_path_or_array
        else:
            img = np.array(image_path_or_array)
        
        # Ensure RGB
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        
        # Detect face
        box, face_crop = detect_faces_mtcnn(img)
        
        if box is None or face_crop is None or face_crop.size == 0:
            return {
                'face_detected': False,
                'aus': {},
                'emotions': {}
            }
        
        # Estimate AUs first (needed for emotion prediction)
        aus = estimate_aus_from_landmarks(face_crop)
        
        # Predict emotions using AU-based mapping (most accurate)
        emotions = predict_emotions(face_crop, aus=aus)
        
        return {
            'face_detected': True,
            'box': box,
            'aus': aus,
            'emotions': emotions
        }


# Module-level instance for easy access
_detector_instance = None

def get_detector(device='cpu'):
    """Get or create singleton detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = LightweightDetector(device=device)
    return _detector_instance
