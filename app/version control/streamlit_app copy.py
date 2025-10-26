from __future__ import annotations

# --- Prefer package import; fall back to adding project root to sys.path ---
try:
    from app.mood_logic import infer_mood  # type: ignore
except ModuleNotFoundError:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from app.mood_logic import infer_mood  # type: ignore

import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "processed"
SESSION_SUMMARY = BASE_DIR / "session_summary.csv"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="MoodWatch 1.2", layout="wide")
st.title("🎥 MoodWatch 1.2 — Local, Streamlined")
st.write("This app reads OpenFace CSVs from `processed/` and infers mood locally via AU heuristics.")

with st.sidebar:
    st.header("Controls")
    auto_refresh = st.checkbox("Auto-refresh every 3s", value=False)
    st.caption("Tip: In a separate terminal, run `python -m app.camera_schedule` to generate new pulses.")

# --- Safe helper functions ---------------------------------------------------
def read_csv_safely(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception as e:
        st.error(f"Failed to read `{p.name}`: {e}")
        return pd.DataFrame()

def col_mean(df: pd.DataFrame, name: str) -> float:
    # Some OpenFace dumps include a leading space in column names occasionally.
    for k in (name, f" {name}"):
        if k in df.columns:
            s = pd.to_numeric(df[k], errors="coerce")
            if s.notna().any():
                return float(s.mean(skipna=True))
            return 0.0
    return 0.0

# --- Discover CSVs -----------------------------------------------------------
csvs = sorted(PROCESSED_DIR.glob("*.csv"))
if not csvs:
    st.warning("No processed CSVs yet. Run a pulse via `python -m app.camera_schedule`.")
else:
    names = [p.name for p in csvs]
    default_idx = len(csvs) - 1
    choice = st.selectbox("Choose a capture CSV", names, index=default_idx)
    current = PROCESSED_DIR / choice

    st.subheader(f"Selected Capture: `{current.name}`")
    df = read_csv_safely(current)
    if not df.empty:
        st.dataframe(df.head(25), use_container_width=True)

        # Build AU summary
        summary = {
            "AU12_r": col_mean(df, "AU12_r"),  # smile
            "AU04_r": col_mean(df, "AU04_r"),  # brow lower (furrow)
            "AU26_r": col_mean(df, "AU26_r"),  # mouth open
            "AU07_r": col_mean(df, "AU07_r"),  # lid tightener (arousal proxy)
        }
        summary["valence_proxy"] = summary["AU12_r"] - summary["AU04_r"]
        summary["arousal_proxy"] = summary["AU07_r"] + summary["AU26_r"]

        mood = infer_mood(summary)

        # Quick metrics row
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("😊 AU12 (smile)", f"{summary['AU12_r']:.3f}")
        m2.metric("🤨 AU04 (furrow)", f"{summary['AU04_r']:.3f}")
        m3.metric("👄 AU26 (mouth open)", f"{summary['AU26_r']:.3f}")
        m4.metric("👁️ AU07 (lid tighten)", f"{summary['AU07_r']:.3f}")
        m5.metric("⚖️ Valence ~ AU12−AU04", f"{summary['valence_proxy']:.3f}")

        st.subheader("Inferred Mood (Selected)")
        st.json(mood)
    else:
        st.info("This CSV could not be read; choose another one or generate a new pulse.")

# --- Session rollup ----------------------------------------------------------
st.divider()
st.subheader("Session History")
if SESSION_SUMMARY.exists():
    try:
        hist = pd.read_csv(SESSION_SUMMARY)
        st.dataframe(hist.tail(200), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read `session_summary.csv`: {e}")
else:
    st.info("`session_summary.csv` will appear after your first completed pulse.")

# --- Simple auto-refresh (sleep + rerun) ------------------------------------
if auto_refresh:
    st.caption("🔄 Auto-refreshing every 3 seconds…")
    time.sleep(3)
    st.experimental_rerun()