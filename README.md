# MoodWatch - AI Audio & Video Mood Analyzer

MoodWatch is an AI-powered multimodal analysis tool that transcribes speech, identifies different speakers, and performs comprehensive emotional analysis on both audio and video content.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **Audio Transcription** - Convert speech to text using Groq's fast Whisper API
- **Speaker Diarization** - Identify and separate different speakers (optional)
- **Emotional Analysis** - Comprehensive mood and emotion detection:
  - Primary and secondary emotions
  - Energy levels (Low, Medium, High)
  - Speaking tone analysis
  - Stress indicators
  - Emotional intensity measurement
  - Key emotional phrases extraction
- **Video Analysis** - Facial expression and Action Unit (AU) detection via Py-Feat
- **Real-time Processing** - Stream analysis with webcam support
- **Modern Web UI** - Responsive interface with drag-and-drop file upload
- **REST API** - Programmatic access for integration

## Project Structure

```
moodwatch/
├── flask_app.py          # Main Flask application
├── config.py             # Configuration settings
├── pyfeat_bridge.py      # Py-Feat facial analysis bridge
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variables template
├── templates/
│   └── index.html        # Main web interface
├── static/
│   ├── app.js            # Frontend JavaScript
│   ├── style.css         # Application styles
│   ├── css/              # Additional stylesheets
│   ├── js/               # Additional scripts
│   └── images/           # Static images
├── ml-worker/            # Lightweight ML worker service
│   ├── app.py            # Worker Flask app
│   ├── lightweight_detector.py
│   └── requirements.txt
├── uploads/              # Temporary upload directory
└── sessions/             # Session data storage
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Groq API key (free at [console.groq.com](https://console.groq.com/keys))
- (Optional) HuggingFace token for speaker diarization
- (Optional) OpenAI API key for legacy support

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/MohammadHR10/MoodWatch_Beyond-.01.git
   cd MoodWatch_Beyond-.01
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your API keys:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   # Optional:
   OPENAI_API_KEY=your_openai_api_key_here
   HF_TOKEN=your_huggingface_token_here
   ```

5. **Run the application**

   ```bash
   python flask_app.py
   ```

6. **Open your browser** at [http://127.0.0.1:5000](http://127.0.0.1:5000)

## API Keys Setup

### Groq API Key (Required)

MoodWatch uses Groq for fast audio transcription and emotion analysis:

1. Go to [Groq Console](https://console.groq.com/keys)
2. Create a free account (no credit card required)
3. Generate an API key
4. Add to `.env`: `GROQ_API_KEY=your_key`

**Why Groq?**

- Up to 20x faster Whisper transcription than OpenAI
- Free tier with generous limits
- Llama 3.3 70B for accurate emotion analysis

### HuggingFace Token (Optional)

Required for speaker diarization (separating different speakers):

1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token
3. Accept terms for `pyannote/speaker-diarization` model
4. Add to `.env`: `HF_TOKEN=your_token`

## Usage

### Web Interface

1. Upload an audio or video file (drag & drop or click)
2. (Optional) Enable "Speaker Separation" for multi-speaker content
3. Click "Analyze" to process
4. View comprehensive results including transcript, emotions, and speaker timeline

### Supported Formats

**Audio:** MP3, WAV, M4A, FLAC, OGG, AAC, WebM  
**Video:** MP4, WebM, AVI, MOV

### API Endpoints

#### `POST /api/analyze`

Analyze audio file and return JSON results.

```bash
curl -X POST -F "audio=@recording.mp3" http://localhost:5000/api/analyze
```

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
    "overall_vibe": "Enthusiastic and positive"
  }
}
```

#### `GET /health`

Health check endpoint.

#### `GET /config`

Get public configuration information.

## Configuration

Environment variables (set in `.env`):

| Variable          | Required | Description                                                    |
| ----------------- | -------- | -------------------------------------------------------------- |
| `GROQ_API_KEY`    | Yes      | Groq API key for audio analysis                                |
| `OPENAI_API_KEY`  | No       | OpenAI API key (legacy support)                                |
| `HF_TOKEN`        | No       | HuggingFace token for speaker diarization                      |
| `EMOTION_BACKEND` | No       | Backend for facial analysis: `pyfeat`, `openface`, or `worker` |
| `SECRET_KEY`      | No       | Flask secret key                                               |
| `PORT`            | No       | Server port (default: 5000)                                    |

## Emotional Analysis Output

MoodWatch provides comprehensive emotional analysis:

| Field                 | Description                                       |
| --------------------- | ------------------------------------------------- |
| `primary_emotion`     | Main detected emotion (Happy, Sad, Angry, etc.)   |
| `secondary_emotions`  | Additional emotions detected                      |
| `mood_category`       | Overall mood (Positive, Negative, Neutral, Mixed) |
| `energy_level`        | Speaking energy (Low, Medium, High)               |
| `tone`                | Communication style (Formal, Casual, Emotional)   |
| `stress_indicators`   | Signs of stress (Fast speech, Hesitation, etc.)   |
| `emotional_intensity` | Emotion intensity score (0-100%)                  |
| `key_phrases`         | Important emotional phrases from transcript       |
| `overall_vibe`        | Casual description of the overall feeling         |

## Deployment

See [DEPLOY.md](DEPLOY.md) for deployment instructions including:

- Docker deployment
- Google Cloud Run
- Production configuration

## Troubleshooting

| Issue                           | Solution                                |
| ------------------------------- | --------------------------------------- |
| "No speech detected"            | Ensure audio contains clear speech      |
| API errors                      | Verify API keys and account limits      |
| Speaker diarization not working | Install pyannote.audio and set HF_TOKEN |
| Import errors                   | Run `pip install -r requirements.txt`   |
| Large file issues               | Split into smaller chunks               |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

Built with Flask, Groq, Py-Feat, and modern web technologies.
