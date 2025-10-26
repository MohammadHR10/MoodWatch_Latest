#!/usr/bin/env python3
"""
MediaPipe Bridge - Comprehensive Action Units Analysis with ML Calibration
This script provides accurate facial analysis using MediaPipe's 468 landmarks
with proper geometric calculations, comprehensive AU detection, and ML-based emotion calibration.
"""

# DEPRECATED: MediaPipe support has been removed. This file remains only to avoid confusion.
# Any attempt to run or import this module will raise a clear error.
raise RuntimeError(
    "MediaPipe path is deprecated. The project is OpenFace-only now. "
    "Use openface_bridge.py and run Flask via flask_app.py."
)

import sys
import json
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import argparse
import math
import traceback
import csv

# ML and data science imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
from collections import defaultdict, Counter
import pickle
from datetime import datetime
import os

# For ML-based emotion recognition
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. ML features disabled.")

# MediaPipe Face Mesh landmark indices for accurate AU calculations
LANDMARK_INDICES = {
    # Eyebrows
    'left_eyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    'right_eyebrow': [296, 334, 293, 300, 276, 285, 295, 282, 283, 276],
    
    # Eyes
    'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    
    # Nose
    'nose_bridge': [6, 51, 48, 115, 131, 134, 102, 49, 220, 305, 279, 360, 344],
    'nose_tip': [1, 2, 5, 4, 19, 94, 168, 8, 9, 10, 151, 195, 197, 236, 3, 51],
    'nostrils': [79, 82, 13, 312, 308, 317, 14, 17, 18, 175],
    
    # Mouth
    'outer_lips': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
    'inner_lips': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 324],
    'mouth_corners': [61, 291],
    'upper_lip': [0, 11, 12, 13, 14, 15, 16, 17, 18, 200],
    'lower_lip': [0, 17, 18, 175, 199, 200, 16, 15, 14, 13, 12, 11, 0],
    
    # Cheeks
    'left_cheek': [116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207, 213, 192, 147],
    'right_cheek': [345, 346, 347, 348, 349, 350, 355, 371, 266, 425, 426, 427, 436, 416, 376],
    
    # Jaw
    'jawline': [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323]
}

def analyze_video_with_mediapipe(video_path, output_path):
    """
    Comprehensive MediaPipe analysis with accurate Action Units detection
    """
    
    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    results = {
        'success': False,
        'frames_analyzed': 0,
        'frames_with_face': 0,
        'action_units': {},
        'emotions': [],
        'landmarks_stats': {},
        'confidence_scores': {},
        'error': None
    }
    
    try:
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            results['error'] = f"Could not open video file: {video_path}"
            return results
            
        frame_count = 0
        face_frame_count = 0
        all_au_values = []
        all_emotions = []
        confidence_scores = []
        
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5) as face_mesh:
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                mesh_results = face_mesh.process(rgb_frame)
                
                if mesh_results.multi_face_landmarks:
                    face_frame_count += 1
                    
                    for face_landmarks in mesh_results.multi_face_landmarks:
                        # Extract landmark coordinates
                        landmarks = extract_landmark_coordinates(face_landmarks, frame.shape)
                        
                        # Calculate comprehensive Action Units
                        au_frame = calculate_comprehensive_action_units(landmarks)
                        all_au_values.append(au_frame)
                        
                        # Detect emotion with confidence
                        emotion_result = detect_emotion_from_comprehensive_aus(au_frame)
                        all_emotions.append(emotion_result)
                        
                        # Store confidence scores
                        confidence_scores.append(calculate_landmark_quality(landmarks))
        
        cap.release()
        
        # Process results
        if all_au_values and face_frame_count > 0:
            results['success'] = True
            results['frames_analyzed'] = frame_count
            results['frames_with_face'] = face_frame_count
            
            # Calculate comprehensive AU statistics
            results['action_units'] = calculate_au_statistics(all_au_values)
            
            # Process emotion data
            results['emotions'] = process_emotion_data(all_emotions)
            
            # Calculate confidence and quality metrics
            results['confidence_scores'] = {
                'mean_confidence': float(np.mean(confidence_scores)),
                'min_confidence': float(np.min(confidence_scores)),
                'max_confidence': float(np.max(confidence_scores)),
                'frames_high_quality': int(np.sum(np.array(confidence_scores) > 0.8))
            }
            
            results['landmarks_stats'] = {
                'total_landmarks_per_frame': 468,
                'frames_with_face': face_frame_count,
                'detection_rate': (face_frame_count / frame_count * 100) if frame_count > 0 else 0,
                'average_face_size': calculate_average_face_size(all_au_values)
            }
        
        else:
            results['error'] = f"No faces detected in {frame_count} frames analyzed"
            
    except Exception as e:
        results['error'] = f"Error processing video: {str(e)}"
        import traceback
        results['traceback'] = traceback.format_exc()
    
    # Save results to JSON file
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        results['output_saved'] = True
    except Exception as e:
        results['save_error'] = str(e)
    
    return results

def extract_training_features(landmarks):
    """
    Extract normalized feature vector from MediaPipe landmarks for ML training
    Returns a flattened feature vector suitable for machine learning
    """
    # Normalize landmarks by face center and size
    face_center = np.mean(landmarks, axis=0)
    centered_landmarks = landmarks - face_center
    
    # Calculate face size for normalization
    face_width = abs(landmarks[234][0] - landmarks[454][0])  # Left to right face
    face_height = abs(landmarks[10][1] - landmarks[152][1])   # Top to bottom face
    face_size = max(face_width, face_height)
    
    if face_size > 0:
        normalized_landmarks = centered_landmarks / face_size
    else:
        normalized_landmarks = centered_landmarks
    
    # Extract key geometric features (more robust than raw coordinates)
    features = []
    
    # 1. Raw normalized landmark coordinates (468 * 3 = 1404 features)
    features.extend(normalized_landmarks.flatten())
    
    # 2. Inter-landmark distances (key facial ratios)
    key_distances = calculate_key_distances(landmarks)
    features.extend(key_distances)
    
    # 3. Facial angles and slopes
    facial_angles = calculate_facial_angles(landmarks)
    features.extend(facial_angles)
    
    # 4. Action Units as features
    au_features = calculate_comprehensive_action_units(landmarks)
    features.extend(list(au_features.values()))
    
    return np.array(features, dtype=np.float32)

def calculate_key_distances(landmarks):
    """Calculate key inter-landmark distances for robust features"""
    distances = []
    
    # Eye measurements
    left_eye_width = np.linalg.norm(landmarks[33] - landmarks[133])
    right_eye_width = np.linalg.norm(landmarks[362] - landmarks[263])
    inter_eye_distance = np.linalg.norm(landmarks[133] - landmarks[362])
    
    # Mouth measurements  
    mouth_width = np.linalg.norm(landmarks[61] - landmarks[291])
    mouth_height = np.linalg.norm(landmarks[13] - landmarks[14])
    
    # Nose measurements
    nose_width = np.linalg.norm(landmarks[79] - landmarks[308])
    nose_height = np.linalg.norm(landmarks[168] - landmarks[18])
    
    # Face proportions
    face_width = np.linalg.norm(landmarks[234] - landmarks[454])
    face_height = np.linalg.norm(landmarks[10] - landmarks[152])
    
    distances.extend([
        left_eye_width, right_eye_width, inter_eye_distance,
        mouth_width, mouth_height, nose_width, nose_height,
        face_width, face_height,
        # Ratios
        mouth_width / face_width if face_width > 0 else 0,
        mouth_height / mouth_width if mouth_width > 0 else 0,
        inter_eye_distance / face_width if face_width > 0 else 0,
        nose_width / face_width if face_width > 0 else 0
    ])
    
    return distances

def calculate_facial_angles(landmarks):
    """Calculate facial angles and slopes for geometric features"""
    angles = []
    
    # Eyebrow angles
    left_brow_angle = calculate_angle(landmarks[70], landmarks[107], landmarks[55])
    right_brow_angle = calculate_angle(landmarks[296], landmarks[334], landmarks[285])
    
    # Mouth corner angles
    left_mouth_angle = calculate_angle(landmarks[84], landmarks[61], landmarks[17])
    right_mouth_angle = calculate_angle(landmarks[17], landmarks[291], landmarks[314])
    
    # Eye angles
    left_eye_angle = calculate_angle(landmarks[33], landmarks[159], landmarks[133])
    right_eye_angle = calculate_angle(landmarks[362], landmarks[386], landmarks[263])
    
    angles.extend([
        left_brow_angle, right_brow_angle,
        left_mouth_angle, right_mouth_angle,
        left_eye_angle, right_eye_angle
    ])
    
    return angles

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points"""
    v1 = p1 - p2
    v2 = p3 - p2
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    return angle

class EmotionCalibrator:
    """ML-based emotion calibration using extracted features"""
    
    def __init__(self):
        self.model = None
        self.feature_scaler = None
        self.emotion_labels = ['neutral', 'happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'contempt']
        self.is_trained = False
        self.training_data = []
        
    def collect_training_sample(self, landmarks, emotion_label):
        """Collect a training sample with landmarks and emotion label"""
        features = extract_training_features(landmarks)
        self.training_data.append({
            'features': features,
            'emotion': emotion_label,
            'timestamp': datetime.now().isoformat()
        })
        
    def save_training_data(self, filepath):
        """Save collected training data"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.training_data, f)
        print(f"Saved {len(self.training_data)} training samples to {filepath}")
    
    def load_training_data(self, filepath):
        """Load training data"""
        with open(filepath, 'rb') as f:
            self.training_data = pickle.load(f)
        print(f"Loaded {len(self.training_data)} training samples from {filepath}")
    
    def train_model(self, test_size=0.2):
        """Train emotion recognition model"""
        if not SKLEARN_AVAILABLE:
            print("Error: scikit-learn required for training")
            return False
            
        if len(self.training_data) < 10:
            print("Error: Need at least 10 training samples")
            return False
        
        # Prepare training data
        X = np.array([sample['features'] for sample in self.training_data])
        y = np.array([sample['emotion'] for sample in self.training_data])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        
        print(f"Training Accuracy: {train_accuracy:.3f}")
        print(f"Test Accuracy: {test_accuracy:.3f}")
        
        # Print classification report
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        return True
    
    def predict_emotion(self, landmarks):
        """Predict emotion using trained model"""
        if not self.is_trained or self.model is None:
            return self.fallback_emotion_detection(landmarks)
        
        try:
            features = extract_training_features(landmarks)
            features = features.reshape(1, -1)
            
            # Get prediction and confidence
            emotion_pred = self.model.predict(features)[0]
            confidence_scores = self.model.predict_proba(features)[0]
            
            # Get confidence for predicted emotion
            emotion_idx = list(self.model.classes_).index(emotion_pred)
            confidence = confidence_scores[emotion_idx]
            
            # Create detailed scores dictionary
            all_scores = {}
            for i, emotion in enumerate(self.model.classes_):
                all_scores[emotion] = confidence_scores[i]
            
            return {
                'emotion': emotion_pred,
                'confidence': float(confidence),
                'all_scores': all_scores,
                'is_significant': confidence > 0.3,
                'method': 'ml_calibrated'
            }
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            return self.fallback_emotion_detection(landmarks)
    
    def fallback_emotion_detection(self, landmarks):
        """Fallback to rule-based detection if ML fails"""
        au_values = calculate_comprehensive_action_units(landmarks)
        return detect_emotion_from_comprehensive_aus(au_values)
    
    def save_model(self, filepath):
        """Save trained model"""
        if self.model is not None:
            model_data = {
                'model': self.model,
                'emotion_labels': self.emotion_labels,
                'is_trained': self.is_trained,
                'training_samples': len(self.training_data)
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.emotion_labels = model_data['emotion_labels']
            self.is_trained = model_data['is_trained']
            
            print(f"Model loaded from {filepath} (trained on {model_data['training_samples']} samples)")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

# Global emotion calibrator instance
emotion_calibrator = EmotionCalibrator()

def extract_landmark_coordinates(face_landmarks, frame_shape):
    """Extract and normalize landmark coordinates"""
    height, width = frame_shape[:2]
    landmarks = []
    
    for landmark in face_landmarks.landmark:
        x = landmark.x * width
        y = landmark.y * height
        z = landmark.z * width  # Relative depth
        landmarks.append([x, y, z])
    
    return np.array(landmarks)

def calculate_comprehensive_action_units(landmarks):
    """
    Calculate comprehensive Action Units using accurate geometric relationships
    Based on Facial Action Coding System (FACS) and MediaPipe 468 landmarks
    """
    
    au_results = {}
    
    # AU01 - Inner Brow Raiser
    au_results['AU01_Inner_Brow_Raiser'] = calculate_au01_inner_brow_raiser(landmarks)
    
    # AU02 - Outer Brow Raiser  
    au_results['AU02_Outer_Brow_Raiser'] = calculate_au02_outer_brow_raiser(landmarks)
    
    # AU04 - Brow Lowerer
    au_results['AU04_Brow_Lowerer'] = calculate_au04_brow_lowerer(landmarks)
    
    # AU05 - Upper Lid Raiser
    au_results['AU05_Upper_Lid_Raiser'] = calculate_au05_upper_lid_raiser(landmarks)
    
    # AU06 - Cheek Raiser
    au_results['AU06_Cheek_Raiser'] = calculate_au06_cheek_raiser(landmarks)
    
    # AU07 - Lid Tightener
    au_results['AU07_Lid_Tightener'] = calculate_au07_lid_tightener(landmarks)
    
    # AU09 - Nose Wrinkler
    au_results['AU09_Nose_Wrinkler'] = calculate_au09_nose_wrinkler(landmarks)
    
    # AU10 - Upper Lip Raiser
    au_results['AU10_Upper_Lip_Raiser'] = calculate_au10_upper_lip_raiser(landmarks)
    
    # AU12 - Lip Corner Puller (Smile)
    au_results['AU12_Lip_Corner_Puller'] = calculate_au12_lip_corner_puller(landmarks)
    
    # AU14 - Dimpler
    au_results['AU14_Dimpler'] = calculate_au14_dimpler(landmarks)
    
    # AU15 - Lip Corner Depressor
    au_results['AU15_Lip_Corner_Depressor'] = calculate_au15_lip_corner_depressor(landmarks)
    
    # AU17 - Chin Raiser
    au_results['AU17_Chin_Raiser'] = calculate_au17_chin_raiser(landmarks)
    
    # AU20 - Lip Stretcher
    au_results['AU20_Lip_Stretcher'] = calculate_au20_lip_stretcher(landmarks)
    
    # AU23 - Lip Tightener
    au_results['AU23_Lip_Tightener'] = calculate_au23_lip_tightener(landmarks)
    
    # AU25 - Lips Part
    au_results['AU25_Lips_Part'] = calculate_au25_lips_part(landmarks)
    
    # AU26 - Jaw Drop
    au_results['AU26_Jaw_Drop'] = calculate_au26_jaw_drop(landmarks)
    
    # AU27 - Mouth Stretch
    au_results['AU27_Mouth_Stretch'] = calculate_au27_mouth_stretch(landmarks)
    
    return au_results

def calculate_au01_inner_brow_raiser(landmarks):
    """Calculate AU01 - Inner Brow Raiser using accurate landmark geometry"""
    # Inner brow points
    left_inner_brow = landmarks[70]   # Left inner brow
    right_inner_brow = landmarks[296] # Right inner brow
    
    # Reference points (inner eye corners)
    left_inner_eye = landmarks[133]
    right_inner_eye = landmarks[362]
    
    # Calculate vertical distances
    left_distance = abs(left_inner_brow[1] - left_inner_eye[1])
    right_distance = abs(right_inner_brow[1] - right_inner_eye[1])
    
    # Normalize by face size
    face_height = abs(landmarks[10][1] - landmarks[152][1])  # Nose tip to forehead
    
    return ((left_distance + right_distance) / 2) / face_height if face_height > 0 else 0

def calculate_au02_outer_brow_raiser(landmarks):
    """Calculate AU02 - Outer Brow Raiser"""
    # Outer brow points
    left_outer_brow = landmarks[63]   # Left outer brow
    right_outer_brow = landmarks[293] # Right outer brow
    
    # Reference points (outer eye corners)
    left_outer_eye = landmarks[33]
    right_outer_eye = landmarks[263]
    
    # Calculate vertical distances
    left_distance = abs(left_outer_brow[1] - left_outer_eye[1])
    right_distance = abs(right_outer_brow[1] - right_outer_eye[1])
    
    # Normalize by face size
    face_height = abs(landmarks[10][1] - landmarks[152][1])
    
    return ((left_distance + right_distance) / 2) / face_height if face_height > 0 else 0

def calculate_au04_brow_lowerer(landmarks):
    """Calculate AU04 - Brow Lowerer"""
    # Mid brow points
    left_mid_brow = landmarks[105]   # Left mid brow
    right_mid_brow = landmarks[334]  # Right mid brow
    
    # Upper eye points
    left_upper_eye = landmarks[159]
    right_upper_eye = landmarks[386]
    
    # Calculate distances (smaller distance = more lowered)
    left_distance = abs(left_mid_brow[1] - left_upper_eye[1])
    right_distance = abs(right_mid_brow[1] - right_upper_eye[1])
    
    # Invert scale (lower brow = higher AU value)
    face_height = abs(landmarks[10][1] - landmarks[152][1])
    base_distance = face_height * 0.08  # Expected normal distance
    
    left_lowering = max(0, base_distance - left_distance)
    right_lowering = max(0, base_distance - right_distance)
    
    return ((left_lowering + right_lowering) / 2) / base_distance if base_distance > 0 else 0

def calculate_au05_upper_lid_raiser(landmarks):
    """Calculate AU05 - Upper Lid Raiser"""
    # Upper eyelid points
    left_upper_lid = landmarks[159]
    right_upper_lid = landmarks[386]
    
    # Lower eyelid points for reference
    left_lower_lid = landmarks[145]
    right_lower_lid = landmarks[374]
    
    # Calculate eye opening
    left_opening = abs(left_upper_lid[1] - left_lower_lid[1])
    right_opening = abs(right_upper_lid[1] - right_lower_lid[1])
    
    # Normalize by face size
    face_width = abs(landmarks[234][0] - landmarks[454][0])  # Face width
    
    return ((left_opening + right_opening) / 2) / face_width if face_width > 0 else 0

def calculate_au06_cheek_raiser(landmarks):
    """Calculate AU06 - Cheek Raiser (squinting, genuine smile)"""
    # Cheek points
    left_cheek = landmarks[116]
    right_cheek = landmarks[345]
    
    # Reference: lower eyelid points  
    left_lower_lid = landmarks[145]
    right_lower_lid = landmarks[374]
    
    # For cheek raising, cheeks should move UP (smaller Y values in image coordinates)
    # We want to detect when cheek is HIGHER than normal position
    # Calculate baseline - distance from cheek to lower eyelid in neutral expression
    face_height = abs(landmarks[10][1] - landmarks[152][1])
    normal_cheek_distance = face_height * 0.12  # Approximate normal distance
    
    # Current distances
    left_distance = abs(left_cheek[1] - left_lower_lid[1])
    right_distance = abs(right_cheek[1] - right_lower_lid[1])
    
    # Calculate elevation (when current distance is smaller than normal = cheek raised)
    left_elevation = max(0, normal_cheek_distance - left_distance)
    right_elevation = max(0, normal_cheek_distance - right_distance)
    
    # Average and normalize
    average_elevation = (left_elevation + right_elevation) / 2
    normalized_elevation = average_elevation / normal_cheek_distance if normal_cheek_distance > 0 else 0
    
    return min(1.0, normalized_elevation * 3)  # Scale appropriately

def calculate_au07_lid_tightener(landmarks):
    """Calculate AU07 - Lid Tightener (squinting)"""
    # Eye corner points
    left_inner = landmarks[133]
    left_outer = landmarks[33]
    right_inner = landmarks[362]
    right_outer = landmarks[263]
    
    # Calculate eye width
    left_width = abs(left_outer[0] - left_inner[0])
    right_width = abs(right_outer[0] - right_inner[0])
    
    # Compare to expected width (smaller = more tightening)
    face_width = abs(landmarks[234][0] - landmarks[454][0])
    expected_eye_width = face_width * 0.15  # Normal eye width ratio
    
    left_tightening = max(0, expected_eye_width - left_width)
    right_tightening = max(0, expected_eye_width - right_width)
    
    return ((left_tightening + right_tightening) / 2) / expected_eye_width if expected_eye_width > 0 else 0

def calculate_au09_nose_wrinkler(landmarks):
    """Calculate AU09 - Nose Wrinkler (nostril flaring, disgust)"""
    # Nose nostril points
    left_nostril = landmarks[79]   # Left nostril
    right_nostril = landmarks[308] # Right nostril
    
    # Nose bridge center for reference
    nose_bridge = landmarks[168]
    
    # For nose wrinkling, nostrils should flare outward (wider apart)
    current_nostril_width = abs(left_nostril[0] - right_nostril[0])
    
    # Calculate baseline nostril width relative to face
    face_width = abs(landmarks[234][0] - landmarks[454][0])
    normal_nostril_ratio = 0.08  # Normal nostril width as ratio of face width
    expected_nostril_width = face_width * normal_nostril_ratio
    
    # Calculate flaring (current width exceeding normal width)
    if expected_nostril_width > 0:
        flaring_ratio = max(0, (current_nostril_width - expected_nostril_width) / expected_nostril_width)
        return min(1.0, flaring_ratio * 2)  # Scale and cap at 1.0
    else:
        return 0.0

def calculate_au10_upper_lip_raiser(landmarks):
    """Calculate AU10 - Upper Lip Raiser"""
    # Upper lip center
    upper_lip_center = landmarks[13]
    
    # Nose base for reference
    nose_base = landmarks[168]
    
    # Calculate vertical distance
    distance = abs(upper_lip_center[1] - nose_base[1])
    
    # Normalize by face height
    face_height = abs(landmarks[10][1] - landmarks[152][1])
    
    return distance / face_height if face_height > 0 else 0

def calculate_au12_lip_corner_puller(landmarks):
    """Calculate AU12 - Lip Corner Puller (Smile)"""
    # Mouth corners (correct indices)
    left_corner = landmarks[61]   # Left mouth corner
    right_corner = landmarks[291] # Right mouth corner
    
    # Mouth center points for reference
    upper_lip_center = landmarks[13]  # Upper lip center
    lower_lip_center = landmarks[14]  # Lower lip center
    
    # Calculate rest position (neutral mouth)
    mouth_center_y = (upper_lip_center[1] + lower_lip_center[1]) / 2
    
    # For smiling, corners should be ABOVE the mouth center (Y decreases in image coords)
    # So we want negative values (corners higher than center)
    left_elevation = mouth_center_y - left_corner[1]   # Positive when corner is above center
    right_elevation = mouth_center_y - right_corner[1] # Positive when corner is above center
    
    # Also check horizontal stretch (smile widens mouth)
    mouth_width = abs(left_corner[0] - right_corner[0])
    
    # Normalize by face width
    face_width = abs(landmarks[234][0] - landmarks[454][0])
    normalized_width = mouth_width / face_width if face_width > 0 else 0
    
    # Combine elevation and width (both should increase for smile)
    elevation_score = max(0, (left_elevation + right_elevation) / 2) / 20  # Normalize elevation
    width_score = max(0, normalized_width - 0.35)  # Baseline mouth width ~0.35 of face
    
    return min(1.0, (elevation_score + width_score) * 2)  # Scale and cap at 1.0

def calculate_au14_dimpler(landmarks):
    """Calculate AU14 - Dimpler"""
    # Mouth corners
    left_corner = landmarks[61]
    right_corner = landmarks[291]
    
    # Cheek reference points
    left_cheek = landmarks[116]
    right_cheek = landmarks[345]
    
    # Calculate corner to cheek distance (dimpling pulls corners back)
    left_distance = np.linalg.norm(left_corner - left_cheek)
    right_distance = np.linalg.norm(right_corner - right_cheek)
    
    # Normalize
    face_width = abs(landmarks[234][0] - landmarks[454][0])
    
    return ((left_distance + right_distance) / 2) / face_width if face_width > 0 else 0

def calculate_au15_lip_corner_depressor(landmarks):
    """Calculate AU15 - Lip Corner Depressor (Frown/Sadness)"""
    # Mouth corners
    left_corner = landmarks[61]
    right_corner = landmarks[291]
    
    # Mouth center for reference  
    upper_lip_center = landmarks[13]
    lower_lip_center = landmarks[14]
    mouth_center_y = (upper_lip_center[1] + lower_lip_center[1]) / 2
    
    # For frowning, corners should be BELOW the mouth center (Y increases in image coords)
    # So we want positive values when corners are lower than center
    left_depression = max(0, left_corner[1] - mouth_center_y)   # Positive when corner below center
    right_depression = max(0, right_corner[1] - mouth_center_y) # Positive when corner below center
    
    # Calculate overall depression
    depression_score = (left_depression + right_depression) / 2
    
    # Normalize by face height
    face_height = abs(landmarks[10][1] - landmarks[152][1])
    normalized_depression = depression_score / face_height if face_height > 0 else 0
    
    return min(1.0, normalized_depression * 10)  # Scale appropriately

def calculate_au17_chin_raiser(landmarks):
    """Calculate AU17 - Chin Raiser"""
    # Chin point
    chin = landmarks[18]
    
    # Lower lip for reference
    lower_lip = landmarks[14]
    
    # Calculate chin to lip distance
    distance = abs(chin[1] - lower_lip[1])
    
    # Normalize by face height
    face_height = abs(landmarks[10][1] - landmarks[152][1])
    
    return distance / face_height if face_height > 0 else 0

def calculate_au20_lip_stretcher(landmarks):
    """Calculate AU20 - Lip Stretcher"""
    # Mouth corners
    left_corner = landmarks[61]
    right_corner = landmarks[291]
    
    # Calculate mouth width
    mouth_width = abs(left_corner[0] - right_corner[0])
    
    # Normalize by face width
    face_width = abs(landmarks[234][0] - landmarks[454][0])
    
    return mouth_width / face_width if face_width > 0 else 0

def calculate_au23_lip_tightener(landmarks):
    """Calculate AU23 - Lip Tightener"""
    # Upper and lower lip points
    upper_lip_points = [landmarks[13], landmarks[12], landmarks[15]]
    lower_lip_points = [landmarks[14], landmarks[17], landmarks[18]]
    
    # Calculate lip thickness
    total_thickness = 0
    for i in range(len(upper_lip_points)):
        thickness = abs(upper_lip_points[i][1] - lower_lip_points[i][1])
        total_thickness += thickness
    
    average_thickness = total_thickness / len(upper_lip_points)
    
    # Normalize by face height (thinner lips = higher tightening)
    face_height = abs(landmarks[10][1] - landmarks[152][1])
    expected_thickness = face_height * 0.02
    
    tightening = max(0, expected_thickness - average_thickness)
    
    return tightening / expected_thickness if expected_thickness > 0 else 0

def calculate_au25_lips_part(landmarks):
    """Calculate AU25 - Lips Part"""
    # Upper and lower lip centers
    upper_lip = landmarks[13]
    lower_lip = landmarks[14]
    
    # Calculate vertical mouth opening
    opening = abs(upper_lip[1] - lower_lip[1])
    
    # Normalize by face height
    face_height = abs(landmarks[10][1] - landmarks[152][1])
    
    return opening / face_height if face_height > 0 else 0

def calculate_au26_jaw_drop(landmarks):
    """Calculate AU26 - Jaw Drop"""
    # Upper lip and chin
    upper_lip = landmarks[13]
    chin = landmarks[18]
    
    # Calculate jaw opening
    jaw_opening = abs(chin[1] - upper_lip[1])
    
    # Normalize by face height
    face_height = abs(landmarks[10][1] - landmarks[152][1])
    
    return jaw_opening / face_height if face_height > 0 else 0

def calculate_au27_mouth_stretch(landmarks):
    """Calculate AU27 - Mouth Stretch"""
    # Upper and lower lip
    upper_lip = landmarks[13]
    lower_lip = landmarks[14]
    
    # Mouth corners
    left_corner = landmarks[61]
    right_corner = landmarks[291]
    
    # Calculate vertical and horizontal mouth dimensions
    vertical_dimension = abs(upper_lip[1] - lower_lip[1])
    horizontal_dimension = abs(left_corner[0] - right_corner[0])
    
    # Stretch ratio
    if vertical_dimension > 0:
        stretch_ratio = horizontal_dimension / vertical_dimension
    else:
        stretch_ratio = 0
    
    return min(stretch_ratio / 5.0, 1.0)  # Normalize to 0-1 range

def detect_emotion_from_comprehensive_aus(au_values):
    """
    Detect emotions using comprehensive Action Units analysis
    Based on established FACS emotion combinations
    Improved patterns to avoid false positives
    """
    
    # Define emotion patterns based on FACS research
    emotion_patterns = {
        'happiness': {
            'required': ['AU12_Lip_Corner_Puller'],
            'supporting': ['AU06_Cheek_Raiser', 'AU25_Lips_Part'],
            'inhibitors': ['AU04_Brow_Lowerer', 'AU15_Lip_Corner_Depressor'],  # Prevent sad/angry misclassification
            'threshold': 0.3
        },
        'sadness': {
            'required': ['AU15_Lip_Corner_Depressor'],
            'supporting': ['AU01_Inner_Brow_Raiser', 'AU04_Brow_Lowerer', 'AU17_Chin_Raiser'],
            'inhibitors': ['AU12_Lip_Corner_Puller'],  # Prevent happy misclassification
            'threshold': 0.25
        },
        'anger': {
            'required': ['AU04_Brow_Lowerer'],
            'supporting': ['AU07_Lid_Tightener', 'AU23_Lip_Tightener'],
            'inhibitors': ['AU12_Lip_Corner_Puller'],  # Prevent happy misclassification
            'threshold': 0.3
        },
        'fear': {
            'required': ['AU01_Inner_Brow_Raiser', 'AU02_Outer_Brow_Raiser'],
            'supporting': ['AU05_Upper_Lid_Raiser', 'AU20_Lip_Stretcher'],
            'inhibitors': ['AU12_Lip_Corner_Puller', 'AU26_Jaw_Drop'],  # High AU26 suggests surprise
            'threshold': 0.25
        },
        'surprise': {
            'required': ['AU01_Inner_Brow_Raiser', 'AU02_Outer_Brow_Raiser', 'AU05_Upper_Lid_Raiser'],
            'supporting': ['AU26_Jaw_Drop', 'AU27_Mouth_Stretch'],
            'inhibitors': ['AU04_Brow_Lowerer'],
            'threshold': 0.3
        },
        'disgust': {
            'required': ['AU09_Nose_Wrinkler'],
            'supporting': ['AU10_Upper_Lip_Raiser', 'AU15_Lip_Corner_Depressor'],
            'inhibitors': ['AU12_Lip_Corner_Puller', 'AU06_Cheek_Raiser'],  # Prevent happy misclassification
            'threshold': 0.4  # Increased threshold to reduce false positives
        },
        'contempt': {
            'required': ['AU14_Dimpler'],
            'supporting': ['AU23_Lip_Tightener'],  # Changed from AU12 to AU23
            'inhibitors': ['AU12_Lip_Corner_Puller'],  # Prevent happy misclassification
            'threshold': 0.2
        }
    }
    
    emotion_scores = {}
    
    for emotion, pattern in emotion_patterns.items():
        score = 0
        required_count = 0
        
        # Check required AUs
        for au in pattern['required']:
            if au in au_values:
                au_value = au_values[au]
                if au_value >= pattern['threshold']:
                    score += au_value * 2  # Weight required AUs more heavily
                    required_count += 1
                else:
                    score += au_value * 0.5  # Partial credit
        
        # Check supporting AUs
        for au in pattern['supporting']:
            if au in au_values:
                score += au_values[au] * 0.5
        
        # Apply inhibitor penalty (reduce score if inhibitor AUs are present)
        if 'inhibitors' in pattern:
            inhibitor_strength = 0
            for au in pattern['inhibitors']:
                if au in au_values:
                    inhibitor_strength += au_values[au]
            # Reduce score based on inhibitor presence
            score = score * (1 - inhibitor_strength * 0.3)
        
        # Require at least one strong required AU
        if required_count > 0:
            emotion_scores[emotion] = score / (len(pattern['required']) + len(pattern['supporting']))
        else:
            emotion_scores[emotion] = 0
    
    # Find dominant emotion
    if emotion_scores:
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        
        # Only return emotion if it's significantly strong
        if dominant_emotion[1] > 0.25:  # Increased threshold for significance
            return {
                'emotion': dominant_emotion[0],
                'confidence': dominant_emotion[1],
                'all_scores': emotion_scores,
                'is_significant': True
            }
        else:
            # Too weak, default to neutral
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'all_scores': emotion_scores,
                'is_significant': False
            }
    else:
        return {
            'emotion': 'neutral',
            'confidence': 0.0,
            'all_scores': {},
            'is_significant': False
        }

def calculate_landmark_quality(landmarks):
    """Calculate quality score for landmark detection"""
    
    # Check if landmarks form reasonable face proportions
    face_width = abs(landmarks[234][0] - landmarks[454][0])
    face_height = abs(landmarks[10][1] - landmarks[152][1])
    
    if face_width == 0 or face_height == 0:
        return 0.0
    
    # Face aspect ratio should be reasonable
    aspect_ratio = face_height / face_width
    aspect_quality = 1.0 - abs(aspect_ratio - 1.3) / 1.3  # Ideal ratio ~1.3
    aspect_quality = max(0, min(1, aspect_quality))
    
    # Check eye symmetry
    left_eye_width = abs(landmarks[33][0] - landmarks[133][0])
    right_eye_width = abs(landmarks[263][0] - landmarks[362][0])
    
    if left_eye_width > 0 and right_eye_width > 0:
        eye_symmetry = min(left_eye_width, right_eye_width) / max(left_eye_width, right_eye_width)
    else:
        eye_symmetry = 0
    
    # Check mouth position relative to nose
    nose_x = landmarks[1][0]
    mouth_center_x = (landmarks[61][0] + landmarks[291][0]) / 2
    mouth_alignment = 1.0 - abs(nose_x - mouth_center_x) / face_width
    mouth_alignment = max(0, min(1, mouth_alignment))
    
    # Combine quality metrics
    overall_quality = (aspect_quality * 0.4 + eye_symmetry * 0.3 + mouth_alignment * 0.3)
    
    return float(overall_quality)

def calculate_au_statistics(all_au_values):
    """Calculate comprehensive statistics for all Action Units"""
    
    if not all_au_values:
        return {}
    
    au_stats = {}
    au_keys = all_au_values[0].keys()
    
    for au in au_keys:
        values = [frame[au] for frame in all_au_values if au in frame and frame[au] is not None]
        
        if values:
            au_stats[au] = {
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'percentile_75': float(np.percentile(values, 75)),
                'percentile_95': float(np.percentile(values, 95)),
                'activation_rate': float(np.sum(np.array(values) > 0.1) / len(values)),
                'strong_activation_rate': float(np.sum(np.array(values) > 0.3) / len(values)),
                'frames_detected': len(values)
            }
        else:
            au_stats[au] = {
                'mean': 0.0, 'median': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                'percentile_75': 0.0, 'percentile_95': 0.0,
                'activation_rate': 0.0, 'strong_activation_rate': 0.0,
                'frames_detected': 0
            }
    
    return au_stats

def process_emotion_data(all_emotions):
    """Process emotion detection results"""
    
    if not all_emotions:
        return []
    
    # Count emotions
    emotion_counts = {}
    confidence_sum = {}
    significant_emotions = []
    
    for emotion_result in all_emotions:
        emotion = emotion_result['emotion']
        confidence = emotion_result['confidence']
        
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        confidence_sum[emotion] = confidence_sum.get(emotion, 0) + confidence
        
        if emotion_result['is_significant']:
            significant_emotions.append(emotion_result)
    
    # Calculate emotion statistics
    emotion_stats = []
    total_frames = len(all_emotions)
    
    for emotion, count in emotion_counts.items():
        avg_confidence = confidence_sum[emotion] / count if count > 0 else 0
        
        emotion_stats.append({
            'emotion': emotion,
            'count': count,
            'percentage': (count / total_frames) * 100,
            'average_confidence': avg_confidence,
            'frames_detected': count
        })
    
    # Sort by count
    emotion_stats.sort(key=lambda x: x['count'], reverse=True)
    
    return emotion_stats

def calculate_average_face_size(all_au_values):
    """Calculate average face size metric"""
    
    if not all_au_values:
        return 0.0
    
    # Use mouth width as a proxy for face size consistency
    mouth_widths = []
    for au_frame in all_au_values:
        if 'AU20_Lip_Stretcher' in au_frame:
            mouth_widths.append(au_frame['AU20_Lip_Stretcher'])
    
    if mouth_widths:
        return float(np.mean(mouth_widths))
    else:
        return 0.0

def analyze_single_image(image_path):
    """Analyze a single image and return JSON results"""
    try:
        # Load the image
        image = cv2.imread(str(image_path))
        if image is None:
            return {"error": f"Could not load image: {image_path}"}
        
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return {"error": "No face detected in image"}
            
            # Process the first detected face
            landmarks = results.multi_face_landmarks[0]
            
            # Extract landmark coordinates
            landmark_coords = extract_landmark_coordinates(landmarks, image.shape[:2])
            
            # Calculate comprehensive Action Units
            action_units = calculate_comprehensive_action_units(landmark_coords)
            
            # Detect emotions using ML calibrator (falls back to rule-based if not trained)
            emotions = emotion_calibrator.predict_emotion(landmark_coords)
            
            # Calculate quality metrics
            quality_metrics = calculate_landmark_quality(landmark_coords)
            
            # Return results in JSON format
            result = {
                'success': True,
                'action_units': action_units,
                'emotions': emotions,
                'landmarks_stats': quality_metrics,
                'metadata': {
                    'mediapipe_version': mp.__version__,
                    'analysis_method': 'geometric_landmarks',
                    'total_landmarks': 468,
                    'image_dimensions': image.shape[:2]
                }
            }
            
            return result
            
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

def main():
    parser = argparse.ArgumentParser(description='MediaPipe analysis bridge with ML calibration')
    parser.add_argument('input_path', help='Path to video or image file to analyze')
    parser.add_argument('output_path', nargs='?', help='Path to save JSON results (optional for single image)')
    parser.add_argument('--image', action='store_true', help='Analyze single image instead of video')
    
    # ML Training arguments
    parser.add_argument('--train', action='store_true', help='Training mode: collect labeled data')
    parser.add_argument('--emotion', type=str, help='Emotion label for training sample (e.g., happiness, sadness)')
    parser.add_argument('--train-data', type=str, default='training_data.pkl', help='Training data file path')
    parser.add_argument('--model', type=str, default='emotion_model.pkl', help='Model file path')
    parser.add_argument('--load-model', action='store_true', help='Load existing trained model')
    parser.add_argument('--train-model', action='store_true', help='Train model from collected data')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Load existing model if requested
    if args.load_model and Path(args.model).exists():
        emotion_calibrator.load_model(args.model)
        print("✅ Loaded existing emotion model")
    
    # Training mode: collect labeled samples
    if args.train:
        if not args.emotion:
            print("Error: --emotion label required in training mode")
            print("Available emotions: neutral, happiness, sadness, anger, fear, surprise, disgust, contempt")
            sys.exit(1)
        
        if args.emotion not in emotion_calibrator.emotion_labels:
            print(f"Error: Invalid emotion '{args.emotion}'")
            print(f"Available emotions: {', '.join(emotion_calibrator.emotion_labels)}")
            sys.exit(1)
        
        # Load existing training data if available
        if Path(args.train_data).exists():
            emotion_calibrator.load_training_data(args.train_data)
        
        # Analyze image and collect training sample
        results = analyze_single_image(input_path)
        if 'error' not in results:
            # Get landmarks for this image
            image = cv2.imread(str(input_path))
            mp_face_mesh = mp.solutions.face_mesh
            
            with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mesh_results = face_mesh.process(rgb_image)
                
                if mesh_results.multi_face_landmarks:
                    landmarks = extract_landmark_coordinates(mesh_results.multi_face_landmarks[0], image.shape[:2])
                    emotion_calibrator.collect_training_sample(landmarks, args.emotion)
                    emotion_calibrator.save_training_data(args.train_data)
                    
                    print(f"✅ Collected training sample: {args.emotion}")
                    print(f"📊 Total samples: {len(emotion_calibrator.training_data)}")
                else:
                    print("❌ No face detected in training image")
        else:
            print(f"❌ Error analyzing training image: {results['error']}")
        
        return
    
    # Train model from collected data
    if args.train_model:
        if Path(args.train_data).exists():
            emotion_calibrator.load_training_data(args.train_data)
            success = emotion_calibrator.train_model()
            if success:
                emotion_calibrator.save_model(args.model)
                print("✅ Model training completed and saved")
            else:
                print("❌ Model training failed")
        else:
            print(f"❌ Training data file not found: {args.train_data}")
        return
    
    # Regular analysis mode
    if args.image or args.output_path is None:
        # Single image analysis - output JSON to stdout for real-time processing
        results = analyze_single_image(input_path)
        print(json.dumps(results, indent=2))
        return
    
    # Video analysis (original functionality)
    output_path = Path(args.output_path)
    
    print(f"Analyzing video: {input_path}")
    print(f"Output will be saved to: {output_path}")
    
    results = analyze_video_with_mediapipe(input_path, output_path)
    
    if results['success']:
        print(f"✅ Analysis complete! Processed {results['frames_analyzed']} frames")
        print(f"📊 Action Units detected: {len(results['action_units'])}")
        print(f"🎭 Emotions detected: {len(results['emotions'])}")
        print(f"🎯 Average confidence: {results['confidence_scores']['mean_confidence']:.3f}")
    else:
        print(f"❌ Analysis failed: {results['error']}")
        if 'traceback' in results:
            print("Detailed error:")
            print(results['traceback'])
        sys.exit(1)

if __name__ == "__main__":
    main()