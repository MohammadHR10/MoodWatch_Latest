"""
Utility functions for audio processing
"""
import os
import tempfile
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import current_app

def allowed_file(filename: str) -> bool:
    """
    Check if file extension is allowed.
    
    Args:
        filename (str): Name of the file
        
    Returns:
        bool: True if file extension is allowed
    """
    if not filename:
        return False
    
    file_ext = Path(filename).suffix.lower()
    return file_ext in current_app.config['ALLOWED_EXTENSIONS']

def save_uploaded_file(file) -> str:
    """
    Save uploaded file to temporary location.
    
    Args:
        file: Flask uploaded file object
        
    Returns:
        str: Path to saved file
        
    Raises:
        ValueError: If file is invalid
    """
    if not file or file.filename == '':
        raise ValueError("No file selected")
    
    if not allowed_file(file.filename):
        allowed_exts = ', '.join(current_app.config['ALLOWED_EXTENSIONS'])
        raise ValueError(f"Invalid file format. Allowed formats: {allowed_exts}")
    
    # Create secure filename
    filename = secure_filename(file.filename)
    if not filename:
        raise ValueError("Invalid filename")
    
    # Create temporary file with original extension
    file_ext = Path(filename).suffix
    with tempfile.NamedTemporaryFile(
        delete=False, 
        suffix=file_ext,
        dir=current_app.config['UPLOAD_FOLDER']
    ) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)
    
    return tmp_path

def cleanup_file(file_path: str) -> None:
    """
    Safely remove temporary file.
    
    Args:
        file_path (str): Path to file to remove
    """
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        current_app.logger.warning(f"Failed to cleanup file {file_path}: {e}")

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes (int): Size in bytes
        
    Returns:
        str: Formatted size string
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f}{size_names[i]}"

def get_file_info(file) -> dict:
    """
    Get information about uploaded file.
    
    Args:
        file: Flask uploaded file object
        
    Returns:
        dict: File information
    """
    if not file:
        return {}
    
    # Get file size
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset position
    
    return {
        'filename': file.filename,
        'size_bytes': size,
        'size_formatted': format_file_size(size),
        'extension': Path(file.filename).suffix.lower() if file.filename else '',
        'is_allowed': allowed_file(file.filename)
    }

def validate_audio_file(file) -> tuple[bool, str]:
    """
    Validate uploaded audio file.
    
    Args:
        file: Flask uploaded file object
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not file:
        return False, "No file provided"
    
    if file.filename == '':
        return False, "No file selected"
    
    file_info = get_file_info(file)
    
    # Check file extension
    if not file_info['is_allowed']:
        allowed_exts = ', '.join(current_app.config['ALLOWED_EXTENSIONS'])
        return False, f"Invalid file format. Allowed formats: {allowed_exts}"
    
    # Check file size
    max_size = current_app.config.get('MAX_CONTENT_LENGTH', 100 * 1024 * 1024)
    if file_info['size_bytes'] > max_size:
        max_size_formatted = format_file_size(max_size)
        return False, f"File too large. Maximum size: {max_size_formatted}"
    
    # Check minimum size (at least 1KB)
    if file_info['size_bytes'] < 1024:
        return False, "File too small. Please upload a valid audio file"
    
    return True, ""
