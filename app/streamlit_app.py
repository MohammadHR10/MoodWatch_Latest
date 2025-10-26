# streamlit_app.py
from __future__ import annotations

import os, sys, tempfile, time, subprocess, signal, json, io, queue, threading
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root to PYTHONPATH
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html
from dotenv import load_dotenv
import altair as alt

# ---- realtime audio deps ----
import av, soundfile as sf, webrtcvad
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

# Optional clustering libs (graceful fallback if missing)
try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    HAVE_SK = True
except Exception:
    HAVE_SK = False

# Robust imports (package vs local)
try:
    from app.mood_logic import infer_mood
    from app.baseline import seed_from_session, zscore, add_sample, load as load_baseline_state
except Exception:
    from mood_logic import infer_mood
    from baseline import seed_from_session, zscore, add_sample, load as load_baseline_state

# ====================== Config & paths ======================
load_dotenv()

BASE_DIR        = Path(__file__).resolve().parents[1]
PROCESSED_DIR   = BASE_DIR / "processed"
SESSION_SUMMARY = BASE_DIR / "session_summary.csv"
LIVE_PREVIEW_IMG= BASE_DIR / "live_preview.jpg"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OPENFACE_BIN = os.getenv("OPENFACE_BIN", str(Path.home()/ "OpenFace"/ "build"/ "bin"/ "FeatureExtraction"))

# Stable schema
AU_FEATURES = [
    "AU01_r","AU02_r","AU04_r","AU06_r","AU07_r","AU09_r","AU10_r",
    "AU12_r","AU14_r","AU15_r","AU17_r","AU20_r","AU23_r","AU25_r",
    "AU26_r","AU45_c"
]
SESSION_FIELDS = ["timestamp", *AU_FEATURES, "label","positivity","energy","tension","hesitation","confidence","csv"]

# ----- Gemini setup (for audio) -----
import google.generativeai as genai
GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
gemini = genai.GenerativeModel("gemini-1.5-flash") if GEMINI_KEY else None

# ----- Audio/VAD settings -----
SAMPLE_RATE = 16000
FRAME_MS    = 30          # 10/20/30 for webrtcvad
VAD_LEVEL   = 2           # 0–3
HANG_MS     = 450
MAX_SEG_SEC = 12.0
MIN_SEG_SEC = 0.6

# ====================== Streamlit page ======================
st.set_page_config(page_title="MoodWatch 1.4", layout="wide")
st.title("🎥 MoodWatch 1.4 — Local, Streamlined")
st.caption("Record manual pulses, schedule automated recording, analyze existing data, or stream live audio. All processing is local with baseline-aware mood inference.")

# seed baseline from past session (safe if file missing/empty)
seed_from_session(SESSION_SUMMARY, user="default")

# ====================== State init ======================
def _init_audio_state():
    ss = st.session_state
    if "audio_events" not in ss:
        ss.audio_events = queue.Queue()
    ss.setdefault("live_transcript", "")
    ss.setdefault("live_mood", {"label":"", "confidence":0.0, "explanation":""})
    ss.setdefault("seg_count", 0)
    ss.setdefault("webrtc_live", False)
_init_audio_state()

# ====================== Helpers ======================
def read_csv_safely(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception as e:
        st.error(f"Failed to read `{p.name}`: {e}")
        return pd.DataFrame()

def col_mean(df: pd.DataFrame, name: str) -> float:
    for k in (name, f" {name}"):
        if k in df.columns:
            s = pd.to_numeric(df[k], errors="coerce")
            if s.notna().any():
                return float(s.mean(skipna=True))
            return 0.0
    return 0.0

def summarize_df(df: pd.DataFrame) -> tuple[dict, int, float, float, float]:
    summary = {c: col_mean(df, c) for c in AU_FEATURES}
    frames = int(col_mean(df, "frame") or len(df))
    pose_rx, pose_ry, pose_rz = col_mean(df, "pose_Rx"), col_mean(df, "pose_Ry"), col_mean(df, "pose_Rz")
    return summary, frames, pose_rx, pose_ry, pose_rz

def run_openface_on_video(video_path: Path) -> Optional[Path]:
    cmd = [OPENFACE_BIN, "-f", str(video_path), "-aus", "-pose", "-gaze", "-2Dfp", "-3Dfp", "-out_dir", str(PROCESSED_DIR), "-no_vis"]
    with st.spinner("Running OpenFace FeatureExtraction..."):
        res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        st.error("OpenFace failed. See logs below.")
        with st.expander("OpenFace logs"):
            st.code((res.stdout or "") + "\n" + (res.stderr or ""))
        return None
    csvs = sorted(PROCESSED_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    return csvs[-1] if csvs else None

def record_pulse_to_temp(duration_s=6.0, w=640, h=480, fps=15) -> Optional[Path]:
    try:
        import cv2
    except Exception as e:
        st.error("OpenCV (cv2) is not installed. Run: `pip install opencv-python`")
        st.caption(str(e))
        return None
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open camera. On macOS: System Settings → Privacy & Security → Camera.")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h); cap.set(cv2.CAP_PROP_FPS, fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_path = Path(tempfile.gettempdir()) / f"pulse_{datetime.now().strftime('%Y%m%dT%H%M%S')}.mp4"
    out = cv2.VideoWriter(str(tmp_path), fourcc, fps, (w, h))
    if not out.isOpened():
        st.error("Failed to initialize video writer.")
        cap.release(); return None
    frames_target = int(duration_s * fps)
    progress = st.progress(0, text="Recording pulse…")
    for i in range(frames_target):
        ok, frame = cap.read()
        if not ok: break
        out.write(frame)
        progress.progress((i+1)/frames_target, text=f"Recording pulse… {i+1}/{frames_target} frames")
    cap.release(); out.release(); progress.empty()
    return tmp_path if Path(tmp_path).exists() else None

def _calibration_ring_svg(pct: float, label: str) -> str:
    pct = max(0.0, min(1.0, pct))
    size, r, cx, cy, stroke_w = 64, 28, 32, 32, 6
    circ = 2 * 3.14159 * r
    dash, gap = circ * pct, circ - (circ * pct)
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 64 64">
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#eee" stroke-width="{stroke_w}"/>
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#4a90e2"
              stroke-width="{stroke_w}" stroke-linecap="round"
              stroke-dasharray="{dash} {gap}" transform="rotate(-90 {cx} {cy})"/>
      <text x="32" y="36" text-anchor="middle" font-size="14" font-family="sans-serif">{label}</text>
    </svg>
    """

def _render_calibration_sidebar():
    st.sidebar.subheader("Calibration")
    state = load_baseline_state("default")
    n = len(state.get("samples", [])) if state else 0
    need = 5
    pct = min(n / need, 1.0)
    st_html(_calibration_ring_svg(pct, f"{min(n, need)}/{need}"), height=80)
    st.sidebar.markdown(f"Collect a few short pulses to personalize results.\n\n**{n}** sample{'s' if n != 1 else ''} so far.")
    if n < need:
        st.sidebar.caption("Tip: aim for 3–5 pulses in neutral and typical states.")

def _repair_session_csv(path: Path) -> tuple[bool, str]:
    want = ["timestamp", *AU_FEATURES, "label","positivity","energy","tension","hesitation","confidence","csv"]
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False, "Nothing to repair."
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        if "valence_proxy" in df.columns and "positivity" not in df.columns:
            df["positivity"] = pd.to_numeric(df["valence_proxy"], errors="coerce")
        if "arousal_proxy" in df.columns and "energy" not in df.columns:
            df["energy"] = pd.to_numeric(df["arousal_proxy"], errors="coerce")
        if "expr_score" in df.columns and "confidence" not in df.columns:
            df["confidence"] = pd.to_numeric(df["expr_score"], errors="coerce")
        for col in want:
            if col not in df.columns:
                df[col] = ""
        df = df.reindex(columns=want, fill_value="")
        df.to_csv(path, index=False)
        return True, "Session file repaired."
    except Exception as e:
        return False, f"Repair failed: {e}"

def _append_session_row(summary: dict, mood: dict, csv_name: str = "") -> None:
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **{k: float(summary.get(k, 0.0)) for k in AU_FEATURES},
        "label": mood.get("label", ""),
        "positivity": mood.get("positivity", ""),
        "energy": mood.get("energy", ""),
        "tension": mood.get("tension", ""),
        "hesitation": mood.get("hesitation", ""),
        "confidence": mood.get("confidence", ""),
        "csv": csv_name,
    }
    import csv
    write_header = not SESSION_SUMMARY.exists() or SESSION_SUMMARY.stat().st_size == 0
    with open(SESSION_SUMMARY, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SESSION_FIELDS)
        if write_header: w.writeheader()
        w.writerow(row)

def _ema(series: pd.Series, alpha: float = 0.3) -> pd.Series:
    try:
        return series.astype(float).ewm(alpha=alpha, adjust=False).mean()
    except Exception:
        return series

def _render_session_file_tools():
    st.sidebar.divider()
    st.sidebar.subheader("Session file")
    c1, c2 = st.sidebar.columns(2)
    if c1.button("Repair CSV", key="btn_repair_csv"):
        ok, msg = _repair_session_csv(SESSION_SUMMARY)
        (st.sidebar.success if ok else st.sidebar.error)(msg)
        st.rerun()
    if c2.button("Reset CSV", key="btn_reset_csv"):
        try:
            if SESSION_SUMMARY.exists():
                SESSION_SUMMARY.unlink()
            st.sidebar.success("Deleted. It will be recreated on the next pulse.")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Reset failed: {e}")

_render_calibration_sidebar()
_render_session_file_tools()

# ====================== Visualization helpers ======================
def _canon_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    out = []
    for c in cols:
        if c in df.columns: out.append(c)
        elif f" {c}" in df.columns: out.append(f" {c}")
    return out

def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    for c in cols: d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

def render_histograms(df: pd.DataFrame, key_prefix: str):
    st.markdown("### 📊 AU Histograms (this pulse)")
    cols = _canon_cols(df, AU_FEATURES)
    if not cols:
        st.info("No AU columns found."); return
    default_sel = [c for c in cols if c.strip() in ["AU12_r","AU04_r","AU07_r","AU26_r"]][:4] or cols[:6]
    sel = st.multiselect("Features", cols, default=default_sel, key=f"{key_prefix}_hist_sel")
    bins = st.slider("Bins", 10, 80, 30, key=f"{key_prefix}_bins")
    g = None
    for c in sel:
        hist = (alt.Chart(df)
            .transform_bin("bin", field=c, bin=alt.Bin(maxbins=bins))
            .transform_aggregate(count='count()', groupby=['bin'])
            .mark_bar().encode(
                x=alt.X('bin:Q', title=c),
                y=alt.Y('count:Q', title='count'),
                tooltip=["bin:Q","count:Q"])
            .properties(height=180))
        g = hist if g is None else g | hist
    if g is not None: st.altair_chart(g, use_container_width=True)

def render_cluster(df: pd.DataFrame, key_prefix: str):
    st.markdown("### 🧭 Cluster / PCA view (this pulse)")
    cols = _canon_cols(df, AU_FEATURES)
    if len(cols) < 2: st.info("Need at least two AU columns for PCA/cluster."); return
    df2 = _ensure_numeric(df[cols], cols).dropna()
    if df2.empty: st.info("No numeric rows to plot."); return
    if HAVE_SK:
        k = st.slider("k (KMeans)", 2, 6, 3, key=f"{key_prefix}_k")
        sample_n = st.slider("Sample rows", 100, 4000, min(1000, len(df2)), key=f"{key_prefix}_samp")
        if len(df2) > sample_n: df2 = df2.sample(sample_n, random_state=42)
        pca = PCA(n_components=2).fit_transform(df2.values)
        km = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(df2.values)
        plot_df = pd.DataFrame({"pc1": pca[:,0], "pc2": pca[:,1], "cluster": km.labels_.astype(str)})
        ch = alt.Chart(plot_df).mark_circle().encode(x="pc1:Q", y="pc2:Q", color="cluster:N", tooltip=["pc1","pc2","cluster"]).properties(height=380)
        st.altair_chart(ch, use_container_width=True)
    else:
        X = df2.to_numpy(); X = X - X.mean(axis=0, keepdims=True)
        U, _, _ = np.linalg.svd(X, full_matrices=False)
        plot_df = pd.DataFrame({"pc1": U[:,0], "pc2": U[:,1]})
        ch = alt.Chart(plot_df).mark_circle().encode(x="pc1:Q", y="pc2:Q").properties(height=380)
        st.altair_chart(ch, use_container_width=True)

def render_time_series(session_csv: Path, key_prefix: str):
    st.markdown("### ⏱️ Session time-series (positivity / energy / tension / hesitation)")
    if not session_csv.exists(): st.info("No session history yet."); return
    try:
        hist = pd.read_csv(session_csv, engine="python", on_bad_lines="skip")
    except Exception as e:
        st.warning(f"Could not read session history: {e}"); return
    if hist.empty: st.info("No rows yet. Run a few pulses to populate the timeline."); return
    if "timestamp" in hist.columns:
        t = pd.to_datetime(hist["timestamp"], errors="coerce"); x_encoding = alt.X("t:T", title="time")
    else:
        t = pd.Series(range(len(hist))); x_encoding = alt.X("t:Q", title="time")
    for col, title in [("positivity","Positivity"), ("energy","Energy"), ("tension","Tension"), ("hesitation","Hesitation")]:
        if col not in hist.columns: continue
        vals = pd.to_numeric(hist[col], errors="coerce")
        df_raw = pd.DataFrame({"t": t, "v": vals}).dropna()
        if df_raw.empty: continue
        df_smooth = df_raw.copy(); df_smooth["v"] = _ema(df_raw["v"], alpha=0.35)
        line_raw = alt.Chart(df_raw).mark_line(point=True).encode(x=x_encoding, y=alt.Y("v:Q", title=title), tooltip=["t","v"])
        line_smooth = alt.Chart(df_smooth).mark_line(strokeDash=[4,3]).encode(x=x_encoding, y="v:Q")
        st.altair_chart((line_raw + line_smooth).properties(height=190), use_container_width=True)

# ====================== Real-time audio ======================
class MicProcessor(AudioProcessorBase):
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_LEVEL)
        self._buf = b""; self._silence_ms = 0; self._segment = bytearray()
        self._frame_bytes = int(SAMPLE_RATE * (FRAME_MS/1000.0)) * 2  # 16-bit mono

    def recv_queued(self, frames):
        for f in frames: self._handle_frame(f)
        return frames[-1] if frames else None

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        self._handle_frame(frame); return frame

    def _handle_frame(self, frame: av.AudioFrame):
        pcm = frame.to_ndarray()
        if pcm.ndim > 1: pcm = pcm.mean(axis=0)
        pcm = pcm.astype("int16")
        step = max(1, int(round(frame.sample_rate / SAMPLE_RATE)))
        if step > 1: pcm = pcm[::step]
        self._buf += pcm.tobytes()

        while len(self._buf) >= self._frame_bytes:
            chunk = self._buf[:self._frame_bytes]; self._buf = self._buf[self._frame_bytes:]
            try: voiced = self.vad.is_speech(chunk, SAMPLE_RATE)
            except Exception: voiced = False
            self._segment.extend(chunk)
            self._silence_ms = 0 if voiced else (self._silence_ms + FRAME_MS)
            seg_len_sec = len(self._segment) / (2*SAMPLE_RATE)
            should_flush = ((self._silence_ms >= HANG_MS) and (seg_len_sec >= MIN_SEG_SEC)) or (seg_len_sec >= MAX_SEG_SEC)
            if should_flush:
                try: st.session_state.audio_events.put(bytes(self._segment), block=False)
                except Exception: pass
                self._segment.clear(); self._silence_ms = 0

def _audio_worker():
    while True:
        if not st.session_state.get("webrtc_live"): break
        try:
            seg = st.session_state.audio_events.get(timeout=0.5)
        except queue.Empty:
            continue
        if not gemini:
            st.toast("Gemini key not configured.", icon="⚠️") if hasattr(st, "toast") else None
            continue
        try:
            wav = io.BytesIO()
            sf.write(wav, data=np.frombuffer(seg, dtype="int16"), samplerate=SAMPLE_RATE, format="WAV")
            wav.seek(0)
            resp = gemini.generate_content([
                {"mime_type":"audio/wav","data": wav.read()},
                ("Return STRICT JSON only: {transcript:string, mood_label:Positive|Neutral|Negative|Mixed, "
                 "confidence:number, explanation:string}")
            ])
            try: data = json.loads(getattr(resp, "text", "") or "{}")
            except Exception: data = {}
            tr = (data.get("transcript") or "").strip()
            lb = (data.get("mood_label") or "—").strip()
            cf = float(data.get("confidence") or 0.0)
            ex = (data.get("explanation") or "").strip()
            if tr:
                st.session_state.live_transcript = (st.session_state.live_transcript + " " + tr).strip()
            st.session_state.live_mood = {"label": lb, "confidence": cf, "explanation": ex}
            st.session_state.seg_count += 1
        except Exception as e:
            st.toast(f"Gemini chunk error: {e}", icon="⚠️") if hasattr(st, "toast") else None
            continue

# ====================== UI ======================
tab_manual, tab_scheduled, tab_audio, tab_existing = st.tabs(
    ["🎬 Manual Pulse", "⏰ Scheduled Recording", "🎤 Audio Analysis", "📁 Analyze Existing CSV"]
)

# --------- Manual Pulse ---------
with tab_manual:
    st.subheader("🎬 Manual Pulse Recording")
    with st.expander("Settings", expanded=False):
        duration = st.slider("Duration (seconds)", 3, 10, 6, 1)
        width = st.select_slider("Width", options=[640, 800, 1280], value=640)
        height = st.select_slider("Height", options=[480, 600, 720], value=480)
        fps = st.select_slider("FPS", options=[10, 15, 24, 30], value=15)

    missing = []
    if not Path(OPENFACE_BIN).exists():
        missing.append(f"OPENFACE_BIN not found at: `{OPENFACE_BIN}`")
    if missing:
        st.error("Cannot run pulse:")
        for m in missing: st.write("• " + m)
        st.caption("Set `OPENFACE_BIN` in `.env` (e.g., `OPENFACE_BIN=$HOME/OpenFace/build/bin/FeatureExtraction`).")

    if st.button("🎥 Run Pulse Now", disabled=bool(missing)):
        st.info("Starting capture…")
        vid = record_pulse_to_temp(duration_s=duration, w=width, h=height, fps=fps)
        if vid is not None:
            st.success(f"Captured video: {vid.name}")
            csv_path = run_openface_on_video(vid)
            if csv_path is not None:
                st.success(f"OpenFace complete → `{csv_path.name}`")
                df = read_csv_safely(csv_path)
                if not df.empty:
                    st.dataframe(df.head(25), use_container_width=True)
                    summary, frames, rx, ry, rz = summarize_df(df)
                    zs = zscore("default", summary)
                    mood = infer_mood(summary, frames=frames, pose_rx=rx, pose_ry=ry, pose_rz=rz, z=zs)
                    add_sample("default", summary)
                    try: _append_session_row(summary, mood, csv_name=csv_path.name or "")
                    except Exception as e: st.warning(f"Could not append to session history: {e}")

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Positivity", f"{mood['positivity']:+.2f}")
                    m2.metric("Energy", f"{mood['energy']:.2f}")
                    m3.metric("Tension", f"{mood['tension']:.2f}")
                    m4.metric("Hesitation", f"{mood['hesitation']:.2f}")
                    m5.metric("Confidence", f"{mood['confidence']:.2f}")

                    st.subheader(f"Inferred Mood (This Pulse): **{mood['label']}**")
                    if mood.get("notes"): st.caption(mood["notes"])
                    with st.expander("Signals used (for transparency)"):
                        st.json({"features": mood.get("features", {}), "raw_AUs_used": summary})

                    render_histograms(df, key_prefix="manual")
                    render_cluster(df, key_prefix="manual")
                    render_time_series(SESSION_SUMMARY, key_prefix="manual")
                else:
                    st.error("CSV read returned empty. Check OpenFace output logs above.")
        else:
            st.error("No frames captured. Is the camera accessible?")

# --------- Scheduled Recording ---------
with tab_scheduled:
    st.subheader("⏰ Automated Scheduled Recording")
    st.markdown("Capture pulses at intervals in a background process until you stop it.")

    col1, col2 = st.columns(2)
    with col1:
        schedule_duration = st.number_input("Pulse Duration (seconds)", 3, 30, 6)
        schedule_interval = st.number_input("Interval Between Pulses (seconds)", 10, 3600, 100)
    with col2:
        schedule_fps = st.number_input("FPS", 5, 60, 15)
        schedule_width = st.number_input("Width", 320, 1920, 640)
        schedule_height = st.number_input("Height", 240, 1080, 480)

    ss = st.session_state
    ss.setdefault("scheduler_running", False)
    ss.setdefault("scheduler_process", None)
    ss.setdefault("scheduler_start_time", None)
    ss.setdefault("last_analyzed_csv", None)
    ss.setdefault("last_csv_mtime", 0.0)
    ss.setdefault("last_step", "idle")

    col_start, col_stop, col_status = st.columns([1, 1, 2])

    with col_start:
        if st.button("🚀 Start Scheduled Recording", disabled=ss.scheduler_running):
            try:
                env = os.environ.copy()
                env.update({
                    "PULSE_DURATION": str(schedule_duration),
                    "PULSE_INTERVAL": str(schedule_interval),
                    "FPS": str(schedule_fps),
                    "FRAME_W": str(schedule_width),
                    "FRAME_H": str(schedule_height),
                    "PREVIEW_IMG": str(LIVE_PREVIEW_IMG),
                })
                proc = subprocess.Popen(
                    [sys.executable, "-m", "app.camera_schedule"],
                    cwd=BASE_DIR, env=env, text=True, start_new_session=True
                )
                ss.scheduler_process = proc; ss.scheduler_running = True
                ss.scheduler_start_time = time.time()
                ss.last_analyzed_csv = None; ss.last_csv_mtime = 0.0; ss.last_step = "start"
                st.success("Scheduled recording started!"); st.rerun()
            except Exception as e:
                st.error(f"Failed to start scheduled recording: {e}")

    with col_stop:
        if st.button("⏹️ Stop Scheduled Recording", disabled=not ss.scheduler_running):
            proc = ss.get("scheduler_process")
            try:
                if proc is not None:
                    try: os.killpg(proc.pid, signal.SIGTERM)
                    except Exception: proc.terminate()
                    try: proc.wait(timeout=4)
                    except subprocess.TimeoutExpired:
                        try: os.killpg(proc.pid, signal.SIGKILL)
                        except Exception: proc.kill()
            except Exception as e:
                st.warning(f"Error stopping process: {e}")
            ss.scheduler_process = None; ss.scheduler_running = False
            ss.scheduler_start_time = None; ss.last_analyzed_csv = None
            ss.last_csv_mtime = 0.0; ss.last_step = "idle"
            try:
                if LIVE_PREVIEW_IMG.exists(): LIVE_PREVIEW_IMG.unlink()
            except Exception: pass
            st.success("Scheduled recording stopped!"); st.rerun()

    with col_status:
        if ss.scheduler_running:
            if ss.scheduler_start_time:
                elapsed = time.time() - ss.scheduler_start_time
                h, rem = divmod(int(elapsed), 3600); m, s = divmod(rem, 60)
                st.success(f"🟢 Running for {h:02d}:{m:02d}:{s:02d}")
            else:
                st.success("🟢 Running")
            if ss.scheduler_process and ss.scheduler_process.poll() is not None:
                ss.scheduler_running = False; ss.scheduler_process = None
                ss.scheduler_start_time = None; ss.last_step = "idle"
                st.error("❌ Process stopped unexpectedly"); st.rerun()
        else:
            st.info("⚪ Stopped")

    st.divider()
    st.subheader("🎬 Pulse Status (Live)")
    if not ss.scheduler_running:
        st.info("Start the scheduler to see live capture status.")
    else:
        preview_fresh = LIVE_PREVIEW_IMG.exists() and (time.time() - LIVE_PREVIEW_IMG.stat().st_mtime) < 1.0
        csvs_all = sorted(PROCESSED_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime)
        latest_csv = csvs_all[-1] if csvs_all else None
        latest_mtime = latest_csv.stat().st_mtime if latest_csv else 0.0
        started_at = ss.scheduler_start_time or 0.0

        if ss.last_step in ("idle","start"):
            st.info("Starting capture…"); ss.last_step = "capturing"

        if preview_fresh:
            st.success("Capturing video… (live preview below)")
            st.image(str(LIVE_PREVIEW_IMG), caption="Live camera (during pulses)", use_container_width=True)
        else:
            st.caption("Camera idle between pulses.")

        if latest_csv and (latest_mtime > started_at) and (ss.last_analyzed_csv != str(latest_csv) or latest_mtime > ss.last_csv_mtime):
            if not preview_fresh and ss.last_step == "capturing":
                st.info("Running OpenFace FeatureExtraction…"); ss.last_step = "openface"

            df_latest = read_csv_safely(latest_csv)
            if not df_latest.empty:
                st.success(f"OpenFace complete → `{latest_csv.name}`")
                st.dataframe(df_latest.head(25), use_container_width=True)

                summary, frames, rx, ry, rz = summarize_df(df_latest)
                zs = zscore("default", summary)
                mood = infer_mood(summary, frames=frames, pose_rx=rx, pose_ry=ry, pose_rz=rz, z=zs)
                add_sample("default", summary)

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Positivity", f"{mood['positivity']:+.2f}")
                m2.metric("Energy", f"{mood['energy']:.2f}")
                m3.metric("Tension", f"{mood['tension']:.2f}")
                m4.metric("Hesitation", f"{mood['hesitation']:.2f}")
                m5.metric("Confidence", f"{mood['confidence']:.2f}")

                st.subheader(f"Inferred Mood (Latest Pulse): **{mood['label']}**")
                if mood.get("notes"): st.caption(mood["notes"])
                with st.expander("Signals used (for transparency)"):
                    st.json({"features": mood.get("features", {}), "raw_AUs_used": summary})

                render_histograms(df_latest, key_prefix="sched")
                render_cluster(df_latest, key_prefix="sched")
                render_time_series(SESSION_SUMMARY, key_prefix="sched")

                ss.last_analyzed_csv = str(latest_csv)
                ss.last_csv_mtime = latest_mtime
                ss.last_step = "done"

        time.sleep(1)
        st.rerun()

# --------- Live Audio (near real-time, chunked) ---------
with tab_audio:
    st.header("🎤 Live Audio (near real-time, chunked)")
    if not GEMINI_KEY:
        st.error("Set GEMINI_API_KEY / GOOGLE_API_KEY in your environment for live audio.")
    else:
        ctx = webrtc_streamer(
            key="mic",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            media_stream_constraints={"audio": True, "video": False},
            audio_processor_factory=MicProcessor,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Transcript (rolling)")
            st.write(st.session_state.live_transcript or "…")
        with c2:
            st.subheader("Mood (last segment)")
            m = st.session_state.live_mood
            st.metric("Label", m.get("label","—"))
            st.metric("Confidence", f"{m.get('confidence',0):.2f}")
            st.caption(m.get("explanation",""))

        st.caption(f"debug → playing: {ctx.state.playing} | queued: {st.session_state.audio_events.qsize()} | segments: {st.session_state.seg_count}")

        # Start/stop worker once per session
        try:
            from streamlit.runtime.scriptrunner import add_script_run_ctx
        except Exception:
            def add_script_run_ctx(t): return

        if ctx.state.playing and not st.session_state.get("webrtc_live"):
            st.session_state.webrtc_live = True
            t = threading.Thread(target=_audio_worker, daemon=True)
            add_script_run_ctx(t); t.start()
        if not ctx.state.playing and st.session_state.get("webrtc_live"):
            st.session_state.webrtc_live = False

        # Gentle auto-refresh while streaming (ONLY main thread uses st.rerun)
        if ctx.state.playing:
            time.sleep(0.8)
            st.rerun()

# --------- Analyze Existing CSV ---------
with tab_existing:
    st.subheader("📁 Analyze Existing CSV Data")
    with st.sidebar:
        st.header("Controls")
        auto_refresh = st.checkbox("Auto-refresh every 3s", value=False, key="auto_refresh_existing")
        st.caption("💡 Generate data via Manual Pulse or start Scheduled Recording.")

    csvs = sorted(PROCESSED_DIR.glob("*.csv"))
    if not csvs:
        st.warning("📂 No processed CSVs found yet.")
        st.info("Try a Manual Pulse or start Scheduled Recording.")
    else:
        names = [p.name for p in csvs]
        default_idx = len(csvs) - 1
        choice = st.selectbox("Choose a capture CSV", names, index=default_idx, key="existing_picker")
        current = PROCESSED_DIR / choice

        st.write(f"Selected: `{current.name}`")
        df = read_csv_safely(current)
        if not df.empty:
            st.dataframe(df.head(25), use_container_width=True)
            summary, frames, rx, ry, rz = summarize_df(df)
            zs = zscore("default", summary)
            mood = infer_mood(summary, frames=frames, pose_rx=rx, pose_ry=ry, pose_rz=rz, z=zs)
            add_sample("default", summary)

            if st.button("➕ Add this capture to session history"):
                try:
                    _append_session_row(summary, mood, csv_name=current.name)
                    st.success("Added to session history.")
                except Exception as e:
                    st.error(f"Failed to add: {e}")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Positivity", f"{mood['positivity']:+.2f}")
            m2.metric("Energy", f"{mood['energy']:.2f}")
            m3.metric("Tension", f"{mood['tension']:.2f}")
            m4.metric("Hesitation", f"{mood['hesitation']:.2f}")
            m5.metric("Confidence", f"{mood['confidence']:.2f}")

            st.subheader(f"Inferred Mood (Selected CSV): **{mood['label']}**")
            if mood.get("notes"): st.caption(mood["notes"])
            with st.expander("Signals used (for transparency)"):
                st.json({"features": mood.get("features", {}), "raw_AUs_used": summary})

            render_histograms(df, key_prefix="exist")
            render_cluster(df, key_prefix="exist")
            render_time_series(SESSION_SUMMARY, key_prefix="exist")
        else:
            st.info("This CSV could not be read; choose another one or generate a new pulse.")

    st.divider()
    st.subheader("Session History (raw)")
    if SESSION_SUMMARY.exists():
        try:
            hist = pd.read_csv(SESSION_SUMMARY, engine="python", on_bad_lines="skip")
            st.dataframe(hist.tail(200), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read `session_summary.csv`: {e}")
    else:
        st.info("`session_summary.csv` will appear after your first completed pulse.")

# Optional soft auto-refresh for the Existing tab only
if st.session_state.get("auto_refresh_existing"):
    time.sleep(3)
    st.rerun()
