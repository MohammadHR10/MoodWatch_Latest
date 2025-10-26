#!/usr/bin/env python3
"""
Emotion Model Training Script
This script demonstrates how to collect training data and train the ML emotion model.
"""

import subprocess
import sys
import json
from pathlib import Path

def run_command(cmd):
    """Run a command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def collect_training_samples():
    """Interactive training sample collection."""
    print("🎯 ML Emotion Model Training")
    print("=" * 50)
    print()
    
    # Available emotions
    emotions = ['neutral', 'happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'contempt']
    
    print("Available emotions to train on:")
    for i, emotion in enumerate(emotions, 1):
        print(f"  {i}. {emotion}")
    print()
    
    while True:
        # Get image path
        image_path = input("Enter path to training image (or 'q' to quit): ").strip()
        if image_path.lower() == 'q':
            break
        
        if not Path(image_path).exists():
            print(f"❌ File not found: {image_path}")
            continue
        
        # Get emotion label
        print("\nSelect emotion (1-8):")
        try:
            choice = int(input("Choice: ")) - 1
            if choice < 0 or choice >= len(emotions):
                print("❌ Invalid choice")
                continue
            
            emotion = emotions[choice]
        except ValueError:
            print("❌ Please enter a number")
            continue
        
        # Collect training sample
        cmd = f"python mediapipe_bridge.py --train --emotion {emotion} '{image_path}'"
        success, stdout, stderr = run_command(cmd)
        
        if success:
            print(f"✅ {stdout.strip()}")
        else:
            print(f"❌ Error: {stderr}")
        
        print("-" * 30)

def train_model():
    """Train the emotion model from collected samples."""
    print("\n🔧 Training emotion model...")
    
    cmd = "python mediapipe_bridge.py --train-model"
    success, stdout, stderr = run_command(cmd)
    
    if success:
        print(f"✅ {stdout.strip()}")
        return True
    else:
        print(f"❌ Training failed: {stderr}")
        return False

def test_model():
    """Test the trained model on a sample image."""
    print("\n🧪 Testing trained model...")
    
    test_image = input("Enter path to test image: ").strip()
    if not Path(test_image).exists():
        print(f"❌ File not found: {test_image}")
        return
    
    cmd = f"python mediapipe_bridge.py --load-model --image '{test_image}'"
    success, stdout, stderr = run_command(cmd)
    
    if success:
        try:
            results = json.loads(stdout)
            if 'error' not in results:
                print("🎭 Emotion Analysis Results:")
                emotions = results.get('emotions', {})
                ml_emotions = results.get('ml_emotions', {})
                
                print(f"  Rule-based: {emotions}")
                print(f"  ML-based:   {ml_emotions}")
            else:
                print(f"❌ Analysis error: {results['error']}")
        except json.JSONDecodeError:
            print("❌ Invalid JSON response")
            print(stdout)
    else:
        print(f"❌ Test failed: {stderr}")

def main():
    """Main training workflow."""
    print("🤖 Emotion Recognition ML Training")
    print("=" * 50)
    print()
    print("This script helps you:")
    print("1. Collect labeled training samples")
    print("2. Train an ML emotion model")
    print("3. Test the trained model")
    print()
    
    while True:
        print("\nOptions:")
        print("1. Collect training samples")
        print("2. Train model from collected data")
        print("3. Test trained model")
        print("4. Show training data statistics")
        print("5. Quit")
        
        choice = input("\nChoose option (1-5): ").strip()
        
        if choice == '1':
            collect_training_samples()
        elif choice == '2':
            train_model()
        elif choice == '3':
            test_model()
        elif choice == '4':
            show_statistics()
        elif choice == '5':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")

def show_statistics():
    """Show training data statistics."""
    training_file = Path("training_data.pkl")
    if not training_file.exists():
        print("❌ No training data found. Collect some samples first!")
        return
    
    # Load and analyze training data
    import pickle
    try:
        with open(training_file, 'rb') as f:
            training_data = pickle.load(f)
        
        from collections import Counter
        emotion_counts = Counter([sample['emotion'] for sample in training_data])
        
        print("\n📊 Training Data Statistics:")
        print(f"Total samples: {len(training_data)}")
        print("\nSamples per emotion:")
        for emotion, count in sorted(emotion_counts.items()):
            print(f"  {emotion}: {count}")
        
        # Recommendations
        min_samples = 10
        low_count_emotions = [emotion for emotion, count in emotion_counts.items() if count < min_samples]
        
        if low_count_emotions:
            print(f"\n⚠️  Emotions with < {min_samples} samples:")
            for emotion in low_count_emotions:
                print(f"  {emotion}: {emotion_counts[emotion]} samples")
            print(f"Consider collecting more samples for better accuracy.")
        else:
            print(f"\n✅ All emotions have >= {min_samples} samples - good for training!")
            
    except Exception as e:
        print(f"❌ Error reading training data: {e}")

if __name__ == "__main__":
    main()