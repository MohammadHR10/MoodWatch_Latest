"""
Scheduler v2: pulse capture + OpenFace + optional live preview.

- Camera opened only during a pulse; released immediately after.
- Cooperative stop via SIGINT/SIGTERM or parent termination.
- Optional live preview written to PREVIEW_IMG (if env set).
- Writes session_summary.csv with a STABLE schema.
"""

from __future__ import annotations

import os
import time
import uuid
import csv
import tempfile
import signal
from pathlib import Path
from typing import Dict, Tuple

import cv2
from dotenv import load_dotenv

# --- robust imports for both -m and direct runs ---
try:
    from .openface_pulse import PulseRun
    from .mood_logic import infer_mood
except Exception:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from app.openface_pulse import PulseRun
    from app.mood_logic import infer_mood

load_dotenv()

PULSE_DURATION = float(os.getenv("PULSE_DURATION", "6"))
PULSE_INTERVAL = float(os.getenv("PULSE_INTERVAL", "100"))
FPS            = int(os.getenv("FPS", "15"))
FRAME_W        = int(os.getenv("FRAME_W", "640"))
FRAME_H        = int(os.getenv("FRAME_H", "480"))
PREVIEW_IMG    = os.getenv("PREVIEW_IMG", "")  # if provided, write JPEG preview during capture

BASE_DIR        = Path(__file__).resolve().parents[1]
PROCESSED_DIR   = BASE_DIR / "processed"
SESSION_SUMMARY = BASE_DIR / "session_summary.csv"
LOCK_FILE       = BASE_DIR / ".camera_schedule.lock"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- Stable CSV schema --------------------
AU_FEATURES = [
    "AU01_r","AU02_r","AU04_r","AU06_r","AU07_r","AU09_r","AU10_r",
    "AU12_r","AU14_r","AU15_r","AU17_r","AU20_r","AU23_r","AU25_r",
    "AU26_r","AU45_c"
]

SESSION_FIELDS = [
    "timestamp",
    *AU_FEATURES,
    "label","positivity","energy","tension","hesitation","confidence",
    "csv"
]

# -------------------- Stop/Signal Handling --------------------
_STOP = False

def _request_stop(signum=None, frame=None):
    global _STOP
    _STOP = True

signal.signal(signal.SIGINT,  _request_stop)
try:
    signal.signal(signal.SIGTERM, _request_stop)
except Exception:
    pass

# -------------------- Helpers --------------------
def _video_tempfile() -> str:
    return os.path.join(tempfile.gettempdir(),
                        f"pulse_{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex}.mp4")

def _append_session(row: Dict):
    """Append one row to session_summary.csv using a constant schema."""
    write_header = not SESSION_SUMMARY.exists() or os.path.getsize(SESSION_SUMMARY) == 0
    safe = {k: row.get(k, "") for k in SESSION_FIELDS}
    with open(SESSION_SUMMARY, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SESSION_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(safe)

def _safe_sleep(seconds: float):
    end = time.time() + float(seconds)
    while time.time() < end and not _STOP:
        time.sleep(min(0.2, end - time.time()))

def _open_camera() -> Tuple[cv2.VideoCapture, bool]:
    cap = cv2.VideoCapture(0)
    ok = cap.isOpened()
    if ok:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_FPS, FPS)
    return cap, ok

def _record_pulse(seconds: float) -> str:
    cap, ok = _open_camera()
    if not ok:
        raise RuntimeError("Could not open camera. Grant permissions and ensure no other app is using it.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    path   = _video_tempfile()
    out    = cv2.VideoWriter(path, fourcc, FPS, (FRAME_W, FRAME_H))
    if not out.isOpened():
        cap.release()
        raise RuntimeError("Failed to initialize video writer.")

    target_frames = max(1, int(seconds * FPS))
    grabbed = 0
    last_preview = 0.0

    try:
        while grabbed < target_frames and not _STOP:
            ok, frame = cap.read()
            if not ok:
                break
            out.write(frame)
            grabbed += 1

            # optional live preview written to disk (Streamlit displays it)
            if PREVIEW_IMG:
                now = time.time()
                if now - last_preview >= 0.2:  # ~5 fps
                    try:
                        small_w = min(640, FRAME_W)
                        small_h = int(small_w * FRAME_H / FRAME_W)
                        small = cv2.resize(frame, (small_w, small_h))
                        cv2.imwrite(PREVIEW_IMG, small, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    except Exception:
                        pass
                    last_preview = now
    finally:
        try: out.release()
        except: pass
        try: cap.release()
        except: pass

    if _STOP:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
        raise KeyboardInterrupt("Stopped during capture")

    if grabbed < max(5, int(0.4 * target_frames)):
        try:
            os.remove(path)
        except OSError:
            pass
        raise RuntimeError("Camera captured too few frames; check permissions/lighting and retry.")
    return path

class _LockFile:
    def __init__(self, path: Path): self.path, self._held = path, False
    def __enter__(self):
        if self.path.exists():
            raise RuntimeError("camera_schedule is already running (lock present).")
        with open(self.path, "w") as f: f.write(str(os.getpid()))
        self._held = True; return self
    def __exit__(self, exc_type, exc, tb):
        if self._held:
            try: self.path.unlink(missing_ok=True)
            except Exception: pass

def main():
    print(f"[INFO] Scheduler starting: pulse={PULSE_DURATION:.0f}s interval={PULSE_INTERVAL:.0f}s fps={FPS} res={FRAME_W}x{FRAME_H}")
    _safe_sleep(2.0)
    if _STOP: return

    while not _STOP:
        try:
            print(f"[{time.strftime('%H:%M:%S')}] Camera ON for {PULSE_DURATION:.0f}s")
            video = _record_pulse(PULSE_DURATION)

            pr = PulseRun(video_path=video)
            summary, csv_path = pr.run_openface()

            mood = infer_mood(summary)
            row = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                **{k: round(float(summary.get(k, 0.0)), 6) for k in AU_FEATURES},
                "label": mood.get("label", ""),
                "positivity": mood.get("positivity", ""),
                "energy": mood.get("energy", ""),
                "tension": mood.get("tension", ""),
                "hesitation": mood.get("hesitation", ""),
                "confidence": mood.get("confidence", ""),
                "csv": os.path.basename(csv_path) if csv_path else "",
            }
            _append_session(row)
            print(f"[OF] {os.path.basename(csv_path) if csv_path else 'N/A'} → {row['label']} pos={row['positivity']}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            if _STOP: break
            print(f"[ERR] {e}")

        if _STOP: break
        print(f"[{time.strftime('%H:%M:%S')}] Camera OFF")
        _safe_sleep(PULSE_INTERVAL)

    print("[INFO] Scheduler stopped cleanly.")

if __name__ == "__main__":
    try:
        with _LockFile(LOCK_FILE):
            main()
    finally:
        pass
