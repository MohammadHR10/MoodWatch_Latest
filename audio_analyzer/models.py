"""
Audio processing models and functions
"""
import json
from openai import OpenAI
from flask import current_app

# Optional diarization
try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

class AudioAnalyzer:
    """Main class for audio analysis operations"""
    
    def __init__(self):
        self.client = OpenAI(api_key=current_app.config['OPENAI_API_KEY'])
    
    def transcribe_with_whisper(self, audio_path: str) -> str:
        """
        Uses OpenAI Whisper to transcribe audio files.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            str: Transcribed text
        """
        with open(audio_path, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return response.text
    
    def analyze_emotions(self, transcript: str) -> dict:
        """
        Analyzes emotions and mood from transcript using GPT.
        
        Args:
            transcript (str): Text to analyze
            
        Returns:
            dict: Comprehensive emotional analysis
        """
        system_prompt = self._get_emotion_analysis_prompt()
        user_message = f"Transcript:\n\n{transcript}\n\nProvide comprehensive emotional analysis."
        
        response = self.client.chat.completions.create(
            model=current_app.config['CHAT_MODEL'],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,
            max_tokens=500
        )
        
        text = response.choices[0].message.content
        return self._parse_emotion_response(text, transcript)
    
    def _get_emotion_analysis_prompt(self) -> str:
        """Returns the system prompt for emotion analysis"""
        return (
            "You are an advanced audio-emotion analysis assistant, like 'MoodWatch but for sound'. "
            "The user will provide a transcript of spoken audio. Your job is to analyze the emotional "
            "content, tone, energy, and overall vibe. Return valid JSON ONLY with these keys:\n"
            "  transcript (string) - the transcript provided,\n"
            "  primary_emotion (string) - main emotion: Happy, Sad, Angry, Excited, Calm, Anxious, Frustrated, Content, Surprised, Fearful,\n"
            "  secondary_emotions (array) - up to 2 additional emotions detected,\n"
            "  mood_category (string) - overall mood: Positive, Negative, Neutral, Mixed,\n"
            "  energy_level (string) - Low, Medium, High,\n"
            "  tone (string) - Formal, Casual, Emotional, Professional, Intimate, Aggressive, Gentle,\n"
            "  confidence (float) - between 0 and 1,\n"
            "  stress_indicators (array) - signs of stress: ['Fast speech', 'Repetition', 'Filler words', 'Hesitation'],\n"
            "  emotional_intensity (float) - 0 to 1 how intense the emotions are,\n"
            "  key_phrases (array) - 2-3 phrases that reveal emotional state,\n"
            "  overall_vibe (string) - casual description of the overall feeling,\n"
            "  explanation (string) - detailed analysis (2-3 sentences).\n"
            "Respond with JSON only."
        )
    
    def _parse_emotion_response(self, text: str, original_transcript: str) -> dict:
        """Parse GPT response and return structured emotion data"""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON block
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end+1])
                except json.JSONDecodeError:
                    pass
            
            # Fallback to basic structure
            return {
                "transcript": original_transcript,
                "primary_emotion": "Unknown",
                "secondary_emotions": [],
                "mood_category": "Unknown",
                "energy_level": "Unknown",
                "tone": "Unknown",
                "confidence": 0.0,
                "stress_indicators": [],
                "emotional_intensity": 0.0,
                "key_phrases": [],
                "overall_vibe": "Could not analyze",
                "explanation": "Failed to parse emotion analysis response"
            }

class SpeakerDiarizer:
    """Class for speaker diarization functionality"""
    
    def __init__(self):
        self.available = PYANNOTE_AVAILABLE and self._has_valid_token()
    
    def _has_valid_token(self) -> bool:
        """Check if HuggingFace token is available and valid"""
        hf_token = current_app.config.get('HF_TOKEN')
        return hf_token and hf_token != 'YOUR_HF_TOKEN'
    
    def diarize_speakers(self, audio_path: str):
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            list or str: List of speaker segments or 'not_available'
        """
        if not self.available:
            return "not_available"
        
        try:
            # Use newer, more compatible model version
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", 
                use_auth_token=current_app.config['HF_TOKEN']
            )
            
            # Add audio preprocessing to handle format issues
            import torchaudio
            try:
                # Load and resample audio if needed
                waveform, sample_rate = torchaudio.load(audio_path)
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                    # Save resampled audio temporarily
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                        torchaudio.save(tmp_file.name, waveform, 16000)
                        diarization = pipeline(tmp_file.name)
                        import os
                        os.unlink(tmp_file.name)  # Clean up
                else:
                    diarization = pipeline(audio_path)
            except Exception:
                # Fallback to original method
                diarization = pipeline(audio_path)
            
            results = []
            for turn, track, label in diarization.itertracks(yield_label=True):
                results.append({
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": label
                })
            
            return results
        except Exception as e:
            current_app.logger.error(f"Diarization failed: {e}")
            return "not_available"

def process_audio_file(audio_path: str, enable_diarization: bool = False) -> dict:
    """
    Main function to process audio file completely.
    
    Args:
        audio_path (str): Path to audio file
        enable_diarization (bool): Whether to perform speaker diarization
        
    Returns:
        dict: Complete analysis results
    """
    analyzer = AudioAnalyzer()
    diarizer = SpeakerDiarizer()
    
    # Step 1: Transcribe
    transcript = analyzer.transcribe_with_whisper(audio_path)
    if not transcript or len(transcript.strip()) == 0:
        raise ValueError("No speech detected in the audio file")
    
    # Step 2: Analyze emotions
    emotion_data = analyzer.analyze_emotions(transcript)
    
    # Step 3: Diarization (optional)
    diarization_result = None
    if enable_diarization:
        diarization_result = diarizer.diarize_speakers(audio_path)
    
    # Normalize results
    result = {
        "transcript": emotion_data.get("transcript", transcript),
        "primary_emotion": emotion_data.get("primary_emotion", "Unknown"),
        "secondary_emotions": emotion_data.get("secondary_emotions", []),
        "mood_category": emotion_data.get("mood_category", "Unknown"),
        "energy_level": emotion_data.get("energy_level", "Unknown"),
        "tone": emotion_data.get("tone", "Unknown"),
        "confidence": float(emotion_data.get("confidence", 0.0)),
        "stress_indicators": emotion_data.get("stress_indicators", []),
        "emotional_intensity": float(emotion_data.get("emotional_intensity", 0.0)),
        "key_phrases": emotion_data.get("key_phrases", []),
        "overall_vibe": emotion_data.get("overall_vibe", ""),
        "explanation": emotion_data.get("explanation", ""),
        "diarization": diarization_result
    }
    
    return result
