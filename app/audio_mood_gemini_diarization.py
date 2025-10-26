import io
import os
import json
import streamlit as st
import google.generativeai as genai
from pyannote.audio import Pipeline

# ---- CONFIG ----
API_KEY = st.secrets.get("GOOGLE_API_KEY", None) or os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")

HF_TOKEN = st.secrets.get("HF_TOKEN", None) or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

@st.cache_resource
def get_diarization_pipeline():
    if not HF_TOKEN:
        return None
    return Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)

# ---- UI ----
st.title("🎙️ Audio Mood Analyzer (Gemini + Speaker Diarization)")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "ogg", "webm"])

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    st.audio(io.BytesIO(audio_bytes), format=uploaded_file.type or "audio/wav")

    # ---- Gemini transcription + vibe ----
    with st.spinner("Analyzing with Gemini..."):
        response = model.generate_content([
            {
                "mime_type": uploaded_file.type or "audio/wav",
                "data": audio_bytes,
            },
            (
                "You are an audio analysis assistant.\n"
                "Task:\n"
                "  1. Transcribe the audio.\n"
                "  2. Identify the overall mood/vibe of the speaker(s).\n"
                "Return JSON only with keys: \n"
                "  transcript (string),\n"
                "  mood_label (Positive, Neutral, Negative, Mixed),\n"
                "  confidence (0-1 float),\n"
                "  explanation (short text)."
            ),
        ])

    try:
        output = json.loads(response.text)
        st.subheader("Transcript")
        st.write(output.get("transcript", ""))

        st.subheader("Mood / Vibe")
        st.write(f"Label: {output.get('mood_label', '')}")
        st.write(f"Confidence: {float(output.get('confidence', 0)):.2f}")
        st.write(f"Explanation: {output.get('explanation', '')}")
    except Exception:
        st.error("Could not parse Gemini response as JSON:")
        st.write(getattr(response, "text", str(response)))

    # ---- Speaker diarization (pyannote) ----
    diarization_pipeline = get_diarization_pipeline()
    if diarization_pipeline is None:
        st.warning("HF_TOKEN is not configured. Skipping speaker diarization.")
    else:
        with st.spinner("Running speaker diarization..."):
            # Save to a temp file path because pyannote expects a file path
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            diarization = diarization_pipeline(tmp_path)

        st.subheader("Speakers")
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            st.write(f"{turn.start:.1f}s - {turn.end:.1f}s: {speaker}") 