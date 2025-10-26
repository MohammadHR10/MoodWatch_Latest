"""
Audio Analysis Interface for Streamlit
Extracted from the original Flask-based VoiceVibe application
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
import io
from typing import Optional, Dict, Any
from datetime import datetime

# Audio processing imports
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    st.error("OpenAI not available. Please install: pip install openai")

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

def render_audio_interface():
    """Main audio analysis interface"""
    
    st.subheader("🎵 Upload Audio File")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['mp3', 'wav', 'm4a', 'flac', 'ogg'],
        help="Supported formats: MP3, WAV, M4A, FLAC, OGG (Max size: 100MB)"
    )
    
    # Options
    col1, col2 = st.columns(2)
    
    with col1:
        enable_diarization = st.checkbox(
            "🗣️ Enable Speaker Separation", 
            value=False,
            help="Identify and separate different speakers (requires HuggingFace token)"
        )
    
    with col2:
        analysis_model = st.selectbox(
            "🧠 Analysis Model",
            ["gpt-4o-mini", "gpt-3.5-turbo"],
            help="Choose the AI model for emotion analysis"
        )
    
    # Process button
    if uploaded_file is not None:
        if st.button("🎯 Analyze Audio", type="primary"):
            process_audio_file_streamlit(uploaded_file, enable_diarization, analysis_model)
    
    # Display recent results if available
    if "audio_results" in st.session_state:
        display_audio_results(st.session_state.audio_results)

def process_audio_file_streamlit(uploaded_file, enable_diarization: bool, model: str):
    """Process uploaded audio file"""
    
    if not OPENAI_AVAILABLE:
        st.error("OpenAI is not properly configured. Please check your installation and API key.")
        return
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        return
    
    client = OpenAI(api_key=api_key)
    
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name
    
    try:
        with st.spinner("🎤 Transcribing audio..."):
            # Transcribe using OpenAI Whisper
            with open(temp_path, "rb") as audio_file:
                transcript_response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word"] if enable_diarization else ["segment"]
                )
            
            transcript = transcript_response.text
            segments = getattr(transcript_response, 'segments', [])
        
        # Speaker diarization (if enabled and available)
        speakers_timeline = []
        if enable_diarization and PYANNOTE_AVAILABLE:
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                try:
                    with st.spinner("🗣️ Identifying speakers..."):
                        speakers_timeline = perform_speaker_diarization(temp_path, hf_token, segments)
                except Exception as e:
                    st.warning(f"Speaker diarization failed: {str(e)}")
            else:
                st.warning("HF_TOKEN not found. Speaker diarization disabled.")
        
        # Emotion analysis
        with st.spinner("🎭 Analyzing emotions..."):
            emotion_analysis = analyze_emotions(client, transcript, model)
        
        # Store results in session state
        results = {
            "transcript": transcript,
            "segments": segments,
            "speakers_timeline": speakers_timeline,
            "emotion_analysis": emotion_analysis,
            "file_name": uploaded_file.name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        st.session_state.audio_results = results
        st.success("✅ Audio analysis completed!")
        
    except Exception as e:
        st.error(f"❌ Error processing audio: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def perform_speaker_diarization(audio_path: str, hf_token: str, segments: list) -> list:
    """Perform speaker diarization using pyannote.audio"""
    
    try:
        # Initialize the pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        # Apply the pipeline
        diarization = pipeline(audio_path)
        
        # Convert to timeline format
        timeline = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            timeline.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
                "duration": turn.end - turn.start
            })
        
        return timeline
        
    except Exception as e:
        st.error(f"Speaker diarization error: {str(e)}")
        return []

def analyze_emotions(client, transcript: str, model: str) -> Dict[str, Any]:
    """Analyze emotions in the transcript using OpenAI"""
    
    prompt = f"""
    Analyze the emotional content of this transcript and provide a comprehensive mood analysis.

    Transcript: "{transcript}"

    Please provide:
    1. Primary emotion (dominant emotional state)
    2. Secondary emotions (other notable emotions present)
    3. Energy level (Low/Medium/High)
    4. Speaking tone analysis
    5. Stress indicators (if any)
    6. Emotional intensity (1-10 scale)
    7. Key emotional phrases or moments
    8. Overall mood summary

    Format your response as a detailed analysis that helps understand the speaker's emotional state.
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in emotional analysis and psychology. Provide detailed, professional emotional analysis of audio transcripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return {
            "analysis": response.choices[0].message.content,
            "model_used": model,
            "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else 0
        }
        
    except Exception as e:
        return {
            "analysis": f"Error in emotion analysis: {str(e)}",
            "model_used": model,
            "tokens_used": 0
        }

def display_audio_results(results: Dict[str, Any]):
    """Display the audio analysis results"""
    
    st.subheader("📊 Analysis Results")
    
    # File info
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📁 File", results["file_name"])
    with col2:
        st.metric("⏰ Analyzed", results["timestamp"])
    
    # Transcript
    st.subheader("📝 Transcript")
    st.text_area(
        "Full Transcript",
        value=results["transcript"],
        height=200,
        help="Complete transcription of the audio"
    )
    
    # Emotion Analysis
    st.subheader("🎭 Emotional Analysis")
    st.write(results["emotion_analysis"]["analysis"])
    
    # Model info
    st.caption(f"Analysis by {results['emotion_analysis']['model_used']} | Tokens used: {results['emotion_analysis']['tokens_used']}")
    
    # Speaker Timeline (if available)
    if results["speakers_timeline"]:
        st.subheader("🗣️ Speaker Timeline")
        
        speaker_df = []
        for speaker_turn in results["speakers_timeline"]:
            speaker_df.append({
                "Speaker": speaker_turn["speaker"],
                "Start (s)": f"{speaker_turn['start']:.1f}",
                "End (s)": f"{speaker_turn['end']:.1f}",
                "Duration (s)": f"{speaker_turn['duration']:.1f}"
            })
        
        if speaker_df:
            import pandas as pd
            df = pd.DataFrame(speaker_df)
            st.dataframe(df, width="stretch")
    
    # Download results
    st.subheader("💾 Export Results")
    
    # Create downloadable report
    report = generate_report(results)
    
    st.download_button(
        label="📄 Download Full Report",
        data=report,
        file_name=f"audio_analysis_{results['file_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

def generate_report(results: Dict[str, Any]) -> str:
    """Generate a downloadable text report"""
    
    report = f"""
AUDIO ANALYSIS REPORT
====================

File: {results['file_name']}
Analysis Date: {results['timestamp']}
Model: {results['emotion_analysis']['model_used']}

TRANSCRIPT:
-----------
{results['transcript']}

EMOTIONAL ANALYSIS:
-------------------
{results['emotion_analysis']['analysis']}

"""
    
    if results["speakers_timeline"]:
        report += "\nSPEAKER TIMELINE:\n-----------------\n"
        for speaker_turn in results["speakers_timeline"]:
            report += f"Speaker {speaker_turn['speaker']}: {speaker_turn['start']:.1f}s - {speaker_turn['end']:.1f}s ({speaker_turn['duration']:.1f}s)\n"
    
    return report