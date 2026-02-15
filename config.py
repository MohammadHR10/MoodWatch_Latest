"""
Configuration settings for VoiceVibe Audio Analyzer
"""
import os
from pathlib import Path

# Load environment variables from .env file
def load_dotenv():
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value

# Load .env file when this module is imported
load_dotenv()

class Config:
    """Base configuration class"""
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
    
    # OpenAI settings
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YOUR_OPENAI_API_KEY')
    CHAT_MODEL = os.getenv('CHAT_MODEL', 'gpt-4o-mini')
    
    # HuggingFace settings for diarization
    HF_TOKEN = os.getenv('HF_TOKEN', 'YOUR_HF_TOKEN')
    
    # File upload settings
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    ALLOWED_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm'}
    
    # Upload folder
    BASE_DIR = Path(__file__).parent
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    
    # Diarization settings
    ENABLE_DIARIZATION_DEFAULT = False
    
    @staticmethod
    def init_app(app):
        """Initialize app with config"""
        # Ensure upload directory exists
        Config.UPLOAD_FOLDER.mkdir(exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
