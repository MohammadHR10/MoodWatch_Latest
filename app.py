#!/usr/bin/env python3
"""
VoiceVibe - AI Audio Mood Analyzer
Main application entry point

A professional audio analysis tool that transcribes speech, identifies speakers,
and performs comprehensive emotional analysis - like "MoodWatch but for sound".
"""

import os
from audio_analyzer import create_app

def main():
    """Main application entry point"""
    
    # Get configuration from environment
    config_name = os.getenv('FLASK_CONFIG', 'development')
    
    # Create Flask app using factory pattern
    app = create_app(config_name)
    
    # Get host and port from environment
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 5000))
    debug = config_name == 'development'
    
    print("🎵 VoiceVibe Audio Analyzer")
    print("=" * 50)
    print(f"🚀 Starting server...")
    print(f"📡 Running on: http://{host}:{port}")
    print(f"🔧 Environment: {config_name}")
    print(f"🐛 Debug mode: {debug}")
    print("=" * 50)
    print("\n💡 Features available:")
    print("   • 🎤 Audio transcription (OpenAI Whisper)")
    print("   • 🎭 Emotional analysis (GPT-powered)")
    print("   • 🗣️  Speaker diarization (optional)")
    print("   • 📱 Modern responsive UI")
    print("\n⚙️  Setup requirements:")
    print("   • OPENAI_API_KEY environment variable")
    print("   • HF_TOKEN environment variable (optional, for speaker diarization)")
    print("\n🎯 Ready to analyze your audio files!")
    print("-" * 50)
    
    # Run the application
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True
    )

if __name__ == '__main__':
    main()
