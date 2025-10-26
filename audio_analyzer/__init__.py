"""
VoiceVibe Audio Analyzer Package
"""
import os
from flask import Flask
from config import config

def create_app(config_name='default'):
    """Application factory pattern"""
    # Get the project root directory (one level up from this package)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create Flask app with correct template and static folders
    app = Flask(__name__, 
                template_folder=os.path.join(project_root, 'templates'),
                static_folder=os.path.join(project_root, 'static'))
    
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    
    # Register blueprints
    from .routes import main
    app.register_blueprint(main)
    
    return app
