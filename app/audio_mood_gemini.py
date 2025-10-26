import io
import json
import streamlit as st
import google.generativeai as genai
import os

# ---- CONFIG ----
# Use env var GOOGLE_API_KEY
API_KEY = st.secrets.get("GOOGLE_API_KEY", None) or os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")

# ---- UI ----
st.title("🎙️ Audio Mood Analyzer (Gemini)")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "ogg", "webm"])

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    st.audio(io.BytesIO(audio_bytes), format=uploaded_file.type or "audio/wav")

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
                "  mood_label (one of: Positive, Neutral, Negative, Mixed),\n"
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
        st.error("Could not parse Gemini response as JSON. Raw response:")
        st.write(getattr(response, "text", str(response))) 