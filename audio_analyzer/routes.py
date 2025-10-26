"""
Flask routes for VoiceVibe Audio Analyzer
"""
from flask import Blueprint, request, render_template, current_app, jsonify
from .models import process_audio_file
from .utils import save_uploaded_file, cleanup_file, validate_audio_file, get_file_info

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def index():
    """Main route for audio analysis"""
    if request.method == 'GET':
        return render_template('index.html', 
                             enable_diarization=current_app.config['ENABLE_DIARIZATION_DEFAULT'])
    
    # Handle POST request (file upload and analysis)
    try:
        # Validate file
        if 'audio' not in request.files:
            return render_template('index.html', 
                                 error="No file uploaded",
                                 enable_diarization=False)
        
        file = request.files['audio']
        is_valid, error_message = validate_audio_file(file)
        
        if not is_valid:
            return render_template('index.html', 
                                 error=error_message,
                                 enable_diarization=False)
        
        # Save uploaded file
        try:
            tmp_path = save_uploaded_file(file)
        except ValueError as e:
            return render_template('index.html', 
                                 error=str(e),
                                 enable_diarization=False)
        
        # Get analysis options
        enable_diarization = request.form.get('diarize') is not None
        
        try:
            # Process the audio file
            result = process_audio_file(tmp_path, enable_diarization)
            
            return render_template('index.html',
                                 result=result,
                                 diarization=result.get('diarization'),
                                 diarize=enable_diarization,
                                 file_info=get_file_info(file))
        
        except ValueError as e:
            # User-friendly errors (no speech, API issues, etc.)
            return render_template('index.html', 
                                 error=str(e),
                                 enable_diarization=enable_diarization)
        
        except Exception as e:
            # Generic errors with helpful messages
            error_message = _get_user_friendly_error(str(e))
            current_app.logger.error(f"Audio processing failed: {e}")
            
            return render_template('index.html', 
                                 error=error_message,
                                 enable_diarization=enable_diarization)
        
        finally:
            # Always cleanup temporary file
            cleanup_file(tmp_path)
    
    except Exception as e:
        current_app.logger.error(f"Unexpected error in index route: {e}")
        return render_template('index.html', 
                             error="An unexpected error occurred. Please try again.",
                             enable_diarization=False)

@main.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for audio analysis (JSON response)"""
    try:
        # Validate file
        if 'audio' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['audio']
        is_valid, error_message = validate_audio_file(file)
        
        if not is_valid:
            return jsonify({'error': error_message}), 400
        
        # Save uploaded file
        try:
            tmp_path = save_uploaded_file(file)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        
        # Get analysis options
        enable_diarization = request.form.get('diarize', '').lower() == 'true'
        
        try:
            # Process the audio file
            result = process_audio_file(tmp_path, enable_diarization)
            result['file_info'] = get_file_info(file)
            
            return jsonify({
                'success': True,
                'result': result
            })
        
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        
        except Exception as e:
            error_message = _get_user_friendly_error(str(e))
            current_app.logger.error(f"Audio processing failed: {e}")
            return jsonify({'error': error_message}), 500
        
        finally:
            cleanup_file(tmp_path)
    
    except Exception as e:
        current_app.logger.error(f"Unexpected error in API: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@main.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'VoiceVibe Audio Analyzer'
    })

@main.route('/config')
def get_config():
    """Get public configuration info"""
    return jsonify({
        'max_file_size': current_app.config.get('MAX_CONTENT_LENGTH', 100 * 1024 * 1024),
        'allowed_extensions': list(current_app.config.get('ALLOWED_EXTENSIONS', [])),
        'diarization_available': current_app.config.get('HF_TOKEN', 'YOUR_HF_TOKEN') != 'YOUR_HF_TOKEN'
    })

def _get_user_friendly_error(error_str: str) -> str:
    """Convert technical errors to user-friendly messages"""
    error_lower = error_str.lower()
    
    if any(keyword in error_lower for keyword in ['api', 'openai', 'authentication']):
        return "API error: Please check your OpenAI API key and account limits."
    
    elif any(keyword in error_lower for keyword in ['network', 'connection', 'timeout']):
        return "Network error: Please check your internet connection and try again."
    
    elif any(keyword in error_lower for keyword in ['file', 'format', 'codec']):
        return "File error: Please ensure your audio file is in a supported format and try again."
    
    elif 'rate limit' in error_lower:
        return "Rate limit exceeded: Please wait a moment before trying again."
    
    elif 'quota' in error_lower:
        return "API quota exceeded: Please check your OpenAI account usage limits."
    
    else:
        return "An unexpected error occurred. Please check your audio file and try again."
