# 🎵 VoiceVibe - AI Audio Mood Analyzer

VoiceVibe is a professional AI-powered audio analysis tool that transcribes speech, identifies different speakers, and performs comprehensive emotional analysis. Think of it as "MoodWatch but for sound" - it analyzes the emotional tone, energy level, and overall vibe of audio recordings.

## 🏗️ Project Structure

```
voicevibe/
├── app.py                 # Main application entry point
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── env.example           # Environment variables template
├── README.md             # This file
├── app_old.py            # Original monolithic version (backup)
├── audio_analyzer/       # Core application package
│   ├── __init__.py       # Application factory
│   ├── routes.py         # Flask routes and request handling
│   ├── models.py         # Audio processing and AI models
│   └── utils.py          # Utility functions
├── templates/            # HTML templates
│   └── index.html        # Main UI template
├── static/               # Static assets
│   ├── css/
│   │   └── style.css     # Application styles
│   ├── js/
│   │   └── app.js        # Frontend JavaScript
│   └── images/           # Images and icons
└── uploads/              # Temporary upload directory
```

## ✨ Features

- **🎤 Audio Transcription**: Convert speech to text using OpenAI Whisper
- **🗣️ Speaker Diarization**: Identify and separate different speakers (optional)
- **🎭 Emotional Analysis**: Comprehensive mood and emotion detection including:
  - Primary and secondary emotions
  - Energy levels (Low, Medium, High)
  - Speaking tone analysis
  - Stress indicators
  - Emotional intensity measurement
  - Key emotional phrases extraction
- **📱 Modern UI**: Beautiful, responsive web interface with drag-and-drop file upload
- **🌐 Multi-format Support**: MP3, WAV, M4A, FLAC, OGG audio files
- **🔧 Professional Architecture**: Modular Flask application with proper separation of concerns
- **🚀 API Endpoints**: RESTful API for programmatic access

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- (Optional) HuggingFace token for speaker diarization

### Installation

1. **Clone or download this project**

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:

   ```bash
   # Copy the example environment file
   cp env.example .env

   # Edit .env with your actual values
   nano .env
   ```

   Or set environment variables directly:

   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   export HF_TOKEN="your-huggingface-token-here"  # Optional, for speaker diarization
   ```

4. **Run the application (Flask + OpenFace)**:

   ```bash
   python flask_app.py
   ```

5. **Open your browser** and go to: `http://127.0.0.1:5002`

Note: The project now uses OpenFace for facial Action Units and emotion estimation in the video UI. MediaPipe paths are deprecated and kept only as stubs.

## 🎯 How to Use

1. **Upload Audio**: Click or drag your audio file into the upload area
2. **Optional**: Check "Enable Speaker Separation" if you want to identify different speakers
3. **Analyze**: Click "Analyze Audio" to process your file
4. **View Results**: Get comprehensive analysis including:
   - Full transcript
   - Detailed emotional analysis
   - Speaker timeline (if enabled)

## 📋 API Keys Setup

### OpenAI API Key (Required)

1. Go to [OpenAI API Keys](https://platform.openai.com/account/api-keys)
2. Create a new API key
3. Set it as environment variable: `OPENAI_API_KEY`

### HuggingFace Token (Optional - for Speaker Diarization)

1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token
3. Accept the terms for `pyannote/speaker-diarization` model
4. Set it as environment variable: `HF_TOKEN`

## 🔧 Configuration

You can modify these settings in `app.py`:

- `CHAT_MODEL`: OpenAI model for emotion analysis (default: "gpt-4o-mini")
- `ENABLE_DIARIZATION_DEFAULT`: Default state for speaker diarization checkbox

## 📊 Emotional Analysis Details

VoiceVibe provides comprehensive emotional analysis including:

- **Primary Emotion**: Main detected emotion (Happy, Sad, Angry, Excited, Calm, Anxious, etc.)
- **Secondary Emotions**: Additional emotions detected
- **Mood Category**: Overall mood classification (Positive, Negative, Neutral, Mixed)
- **Energy Level**: Speaking energy (Low, Medium, High)
- **Tone**: Communication style (Formal, Casual, Emotional, etc.)
- **Stress Indicators**: Signs of stress (Fast speech, Repetition, Filler words, Hesitation)
- **Emotional Intensity**: How intense the emotions are (0-100%)
- **Key Phrases**: Important emotional phrases from the transcript
- **Overall Vibe**: Casual description of the overall feeling

## 🛠️ Troubleshooting

### Common Issues

1. **"No speech detected"**: Ensure your audio file contains clear speech
2. **API errors**: Check your OpenAI API key and account limits
3. **Speaker diarization not working**: Install pyannote.audio and set HF_TOKEN
4. **File upload issues**: Ensure your audio file is in a supported format

### File Size Limits

- Maximum file size depends on your OpenAI plan
- For large files, consider splitting them into smaller chunks

## 🎵 Supported Audio Formats

- MP3
- WAV
- M4A
- FLAC
- OGG

## 🤝 Contributing

This project was built based on conversation requirements for creating an audio analysis tool that can "rip" audio files, separate speakers, and analyze emotional content like MoodWatch does for visual content.

## 📄 License

This project is open source. Feel free to modify and use it for your needs.

## 🔌 API Endpoints

VoiceVibe provides RESTful API endpoints for programmatic access:

### POST `/api/analyze`

Analyze audio file and return JSON results.

**Parameters:**

- `audio`: Audio file (multipart/form-data)
- `diarize`: Boolean, enable speaker diarization (optional)

**Response:**

```json
{
  "success": true,
  "result": {
    "transcript": "...",
    "primary_emotion": "Happy",
    "secondary_emotions": ["Excited"],
    "mood_category": "Positive",
    "energy_level": "High",
    "tone": "Casual",
    "confidence": 0.87,
    "stress_indicators": [],
    "emotional_intensity": 0.75,
    "key_phrases": ["great news", "excited to share"],
    "overall_vibe": "Enthusiastic and positive",
    "explanation": "...",
    "diarization": [...]
  }
}
```

### GET `/health`

Health check endpoint.

### GET `/config`

Get public configuration information.

## 🧪 Development

### Project Architecture

- **Flask Application Factory**: Modular app creation with configuration management
- **Blueprint Structure**: Organized route handling
- **Separation of Concerns**: Models, utils, routes, and templates are separate
- **Professional Error Handling**: User-friendly error messages and logging
- **Static Asset Management**: Organized CSS, JS, and image files

### Adding New Features

1. **Models**: Add audio processing functions to `audio_analyzer/models.py`
2. **Routes**: Add new endpoints to `audio_analyzer/routes.py`
3. **Frontend**: Update templates and static files
4. **Configuration**: Update `config.py` for new settings

## 🆘 Support

If you encounter issues:

1. Check that all environment variables are set correctly
2. Ensure you have a stable internet connection
3. Verify your audio file format is supported
4. Check your OpenAI API account status and limits
5. Review the logs for detailed error information

### Common Issues

- **Import errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
- **API key errors**: Double-check your OpenAI API key is valid and has sufficient credits
- **Diarization issues**: Verify HuggingFace token and model access permissions

---

Built with ❤️ using Flask, OpenAI Whisper, and modern web technologies.
