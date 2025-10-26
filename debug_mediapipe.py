#!/usr/bin/env python3
"""
Debug MediaPipe Analysis - Test real-time emotion detection
"""

# DEPRECATED: MediaPipe debugging scripts are removed. Use OpenFace instead.
raise RuntimeError(
    "debug_mediapipe.py is deprecated. Verify OpenFace by running FeatureExtraction and using openface_bridge.py."
)

import cv2
import numpy as np
import mediapipe as mp
import json

def test_mediapipe_realtime():
    """Test MediaPipe with live camera feed"""
    
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Starting MediaPipe debug test...")
    print("Press 'q' to quit, 's' to save analysis")
    
    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_faces=1
    ) as face_mesh:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                    
                    # Extract key points and calculate basic AUs
                    landmarks = []
                    for lm in face_landmarks.landmark:
                        landmarks.append([lm.x * frame.shape[1], lm.y * frame.shape[0]])
                    
                    landmarks = np.array(landmarks)
                    
                    # Test basic AU calculations
                    mouth_corners = {
                        'left': landmarks[61],
                        'right': landmarks[291],
                        'center_upper': landmarks[13],
                        'center_lower': landmarks[14]
                    }
                    
                    # Calculate smile (AU12) manually
                    mouth_width = np.linalg.norm(mouth_corners['left'] - mouth_corners['right'])
                    mouth_height = np.linalg.norm(mouth_corners['center_upper'] - mouth_corners['center_lower'])
                    
                    # Simple smile detection
                    left_elevation = mouth_corners['center_upper'][1] - mouth_corners['left'][1]
                    right_elevation = mouth_corners['center_upper'][1] - mouth_corners['right'][1]
                    
                    smile_intensity = (left_elevation + right_elevation) / 2
                    
                    # Display debug info
                    cv2.putText(frame, f"Smile: {smile_intensity:.3f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Mouth W: {mouth_width:.1f}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Left Corner Y: {mouth_corners['left'][1]:.1f}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw key points
                    cv2.circle(frame, tuple(mouth_corners['left'].astype(int)), 3, (255, 0, 0), -1)
                    cv2.circle(frame, tuple(mouth_corners['right'].astype(int)), 3, (255, 0, 0), -1)
                    cv2.circle(frame, tuple(mouth_corners['center_upper'].astype(int)), 3, (0, 255, 0), -1)
                    cv2.circle(frame, tuple(mouth_corners['center_lower'].astype(int)), 3, (0, 255, 0), -1)
                    
            else:
                cv2.putText(frame, "No face detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('MediaPipe Debug', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame analysis
                if results.multi_face_landmarks:
                    print(f"Smile intensity: {smile_intensity:.3f}")
                    print(f"Left corner elevation: {left_elevation:.3f}")
                    print(f"Right corner elevation: {right_elevation:.3f}")
                    print(f"Mouth width: {mouth_width:.1f}")
                    print("---")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_mediapipe_realtime()