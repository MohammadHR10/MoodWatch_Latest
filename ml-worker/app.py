"""
ML Worker - Lightweight AU Detection Service
Uses stripped-down detector instead of heavy py-feat (~1GB → ~200MB)
"""
import os
import sys
import io
import base64
import logging
import time

# Disable tqdm progress bars globally
os.environ["TQDM_DISABLE"] = "1"
os.environ["DISABLE_TQDM"] = "1"

from flask import Flask, request, jsonify
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Temp directory for frame images
TEMP_DIR = '/tmp/detector_images'
os.makedirs(TEMP_DIR, exist_ok=True)

# Global detector instance (lazy loaded)
_detector = None

def get_detector():
    """Lazy load the lightweight detector"""
    global _detector
    if _detector is None:
        logger.info("Loading lightweight detector...")
        from lightweight_detector import LightweightDetector
        _detector = LightweightDetector(device='cpu')
        logger.info("Lightweight detector loaded successfully")
    return _detector

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'type': 'lightweight'})

@app.route('/warmup', methods=['POST'])
def warmup():
    """Warm up the detector by loading models"""
    try:
        detector = get_detector()
        # Create a small test image to warm up all models
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[30:70, 30:70] = 200  # Add a bright square
        detector.detect_image(test_img)
        return jsonify({'status': 'warm', 'type': 'lightweight'})
    except Exception as e:
        logger.error(f"Warmup error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze base64 image, return AUs and emotions."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image'}), 400
        
        # Decode image
        img_data = base64.b64decode(data['image'])
        img = Image.open(io.BytesIO(img_data))
        
        logger.info(f"Received image: mode={img.mode}, size={img.size}")
        
        # Convert to RGB if needed
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        elif img.mode == 'L':
            img = img.convert('RGB')
        
        # Downscale large frames to speed up detection
        max_w, max_h = 640, 480
        if img.size[0] > max_w or img.size[1] > max_h:
            img.thumbnail((max_w, max_h))
            logger.info(f"Resized to {img.size}")
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Get detector and analyze
        detector = get_detector()
        
        start = time.time()
        result = detector.detect_image(img_array)
        elapsed = time.time() - start
        logger.info(f"detect_image completed in {elapsed:.2f}s")
        
        if not result.get('face_detected', False):
            return jsonify({
                'success': True, 
                'face_detected': False, 
                'aus': {}, 
                'emotions': {}
            })
        
        # Extract AUs (already in correct format from lightweight detector)
        aus = result.get('aus', {})
        
        # Round AU values and filter NaN
        aus = {k: round(float(v), 3) for k, v in aus.items() 
               if v is not None and not (isinstance(v, float) and np.isnan(v))}
        
        # Extract emotions (already in correct format)
        emotions = result.get('emotions', {})
        
        # Round emotion values and filter NaN
        emotions = {k: round(float(v), 3) for k, v in emotions.items() 
                    if v is not None and not (isinstance(v, float) and np.isnan(v))}
        
        # Log key AU values for debugging
        key_aus = {k: aus.get(k, 0) for k in ['AU06', 'AU12', 'AU01', 'AU04', 'AU05', 'AU26']}
        logger.info(f"Key AUs: {key_aus}")
        
        # Log top emotion
        if emotions:
            top_emotion = max(emotions.items(), key=lambda x: x[1])
            logger.info(f"Top emotion: {top_emotion[0]}={top_emotion[1]:.2f}")
        
        logger.info(f"Analysis complete: {len(aus)} AUs, {len(emotions)} emotions detected")
        
        return jsonify({
            'success': True,
            'face_detected': True,
            'aus': aus,
            'emotions': emotions
        })
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Pre-load detector on startup for faster first request
    logger.info("Pre-loading detector...")
    try:
        get_detector()
        logger.info("Detector pre-loaded successfully")
    except Exception as e:
        logger.warning(f"Could not pre-load detector: {e}")
    
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
