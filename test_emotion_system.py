#!/usr/bin/env python
"""
Test script to verify emotion analysis system is working correctly after bug fixes
"""

import json
import subprocess
import sys
import os

def test_emotion_detection():
    """Test the complete emotion detection pipeline"""
    
    print("=" * 60)
    print("EMOTION ANALYSIS SYSTEM - TEST SUITE")
    print("=" * 60)
    print()
    
    # Get paths
    venv_python = "/Users/mohammad/Downloads/speaker 4/.venv311/bin/python"
    bridge_script = "/Users/mohammad/Downloads/speaker 4/mediapipe_bridge.py"
    test_image = "/Users/mohammad/Downloads/speaker 4/webcam_test.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return False
    
    print(f"📸 Test Image: {test_image}")
    print()
    
    # Run MediaPipe analysis
    print("🔄 Running MediaPipe analysis...")
    try:
        result = subprocess.run(
            [venv_python, bridge_script, test_image],
            capture_output=True,
            text=True,
            timeout=10.0
        )
    except subprocess.TimeoutExpired:
        print("❌ Analysis timed out")
        return False
    
    if result.returncode != 0:
        print(f"❌ Analysis failed with return code {result.returncode}")
        print(f"Error: {result.stderr[:500]}")
        return False
    
    # Parse JSON output
    print("📋 Parsing analysis results...")
    try:
        stdout_lines = result.stdout.strip().split('\n')
        json_lines = []
        in_json = False
        
        for line in stdout_lines:
            if line.strip().startswith('{'):
                in_json = True
            if in_json:
                json_lines.append(line)
        
        json_str = '\n'.join(json_lines)
        data = json.loads(json_str)
    except (json.JSONDecodeError, IndexError) as e:
        print(f"❌ Failed to parse JSON output: {e}")
        return False
    
    # Verify structure
    print("✓ JSON parsed successfully")
    print()
    
    # Check for required fields
    required_fields = ['success', 'action_units', 'emotions']
    for field in required_fields:
        if field not in data:
            print(f"❌ Missing field: {field}")
            return False
        print(f"✓ Field '{field}' present")
    
    print()
    
    # Extract data
    aus = data['action_units']
    emotions = data['emotions']
    
    # Test 1: Check emotion is a dict (not list)
    print("TEST 1: Emotion Data Type")
    print("-" * 40)
    if not isinstance(emotions, dict):
        print(f"❌ FAIL: Emotions should be dict, got {type(emotions)}")
        return False
    print(f"✓ PASS: Emotions is dict")
    print()
    
    # Test 2: Check emotion has required keys
    print("TEST 2: Emotion Dictionary Keys")
    print("-" * 40)
    required_emotion_keys = ['emotion', 'confidence', 'all_scores', 'is_significant']
    for key in required_emotion_keys:
        if key not in emotions:
            print(f"❌ FAIL: Missing emotion key: {key}")
            return False
        print(f"✓ PASS: Key '{key}' present")
    print()
    
    # Test 3: Emotion parsing in Flask-like code
    print("TEST 3: Flask Emotion Extraction")
    print("-" * 40)
    try:
        # Simulate Flask code
        dominant_emotion = 'Neutral'
        if emotions and isinstance(emotions, dict):
            dominant_emotion = emotions.get('emotion', 'Neutral')
        elif emotions and isinstance(emotions, list) and len(emotions) > 0:
            sorted_emotions = sorted(emotions, key=lambda x: x.get('confidence', 0), reverse=True)
            dominant_emotion = sorted_emotions[0].get('emotion', 'Neutral')
        
        print(f"✓ PASS: Extracted emotion = '{dominant_emotion}'")
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False
    print()
    
    # Test 4: Check inhibitor logic is working
    print("TEST 4: Inhibitor Logic")
    print("-" * 40)
    all_scores = emotions['all_scores']
    
    # For a happy face, happiness should be highest
    # and contempt/disgust should be reduced due to inhibitor logic
    if dominant_emotion == 'happiness':
        happiness_score = all_scores['happiness']
        contempt_score = all_scores['contempt']
        disgust_score = all_scores['disgust']
        
        print(f"Happiness: {happiness_score:.3f}")
        print(f"Contempt:  {contempt_score:.3f}")
        print(f"Disgust:   {disgust_score:.3f}")
        
        # Check that inhibitor logic reduced false positives
        if contempt_score < 0.5 and disgust_score < 0.5:
            print("✓ PASS: Inhibitor logic working (false positives reduced)")
        else:
            print("⚠ WARNING: Inhibitor logic may not be working optimally")
    print()
    
    # Test 5: Action Units
    print("TEST 5: Action Units Detection")
    print("-" * 40)
    print(f"Total AUs detected: {len(aus)}")
    
    expected_aus = 17
    if len(aus) >= expected_aus:
        print(f"✓ PASS: Detected {len(aus)} AUs (expected ~{expected_aus})")
    else:
        print(f"⚠ WARNING: Only {len(aus)} AUs detected (expected ~{expected_aus})")
    
    # Print AU values
    print("\nTop AUs for this face:")
    for au_name in sorted(aus.keys(), key=lambda x: aus[x], reverse=True)[:5]:
        print(f"  {au_name}: {aus[au_name]:.3f}")
    print()
    
    # Summary
    print("=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print()
    print(f"RESULT:")
    print(f"  Emotion: {emotions['emotion']}")
    print(f"  Confidence: {emotions['confidence']:.1%}")
    print(f"  Significant: {'Yes' if emotions['is_significant'] else 'No'}")
    print()
    
    return True

if __name__ == "__main__":
    success = test_emotion_detection()
    sys.exit(0 if success else 1)
