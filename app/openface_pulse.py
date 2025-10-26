"""
Handles OpenFace feature extraction on recorded video pulses
and produces a compact summary.
"""
from __future__ import annotations

import os
import glob
import uuid
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Tuple, Dict

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

OPENFACE_BIN = os.getenv("OPENFACE_BIN", os.path.expanduser("~/OpenFace/build/bin/FeatureExtraction"))
OUTPUT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "processed"))

def _latest_csv(out_dir: str) -> str | None:
    csvs = sorted(glob.glob(os.path.join(out_dir, "*.csv")))
    return csvs[-1] if csvs else None

@dataclass
class PulseRun:
    video_path: str
    out_csv: str | None = None
    summary: Dict = None

    def run_openface(self) -> Tuple[Dict, str]:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cmd = [
            OPENFACE_BIN,
            "-f", self.video_path,
            "-aus", "-pose", "-gaze", "-2Dfp", "-3Dfp",
            "-out_dir", OUTPUT_DIR,
            "-no_vis"
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"OpenFace failed (code {proc.returncode}).\nSTDERR:\n{proc.stderr}")

        csv_path = _latest_csv(OUTPUT_DIR)
        if not csv_path or not os.path.exists(csv_path):
            raise FileNotFoundError("OpenFace did not produce a CSV in the output directory.")

        df = pd.read_csv(csv_path)
        # OpenFace sometimes prefixes a space in column names; normalize access
        def mean(col):
            for k in (col, f" {col}"):
                if k in df.columns:
                    return float(df[k].mean())
            return 0.0

        au12 = mean("AU12_r")
        au04 = mean("AU04_r")
        au26 = mean("AU26_r")
        au07 = mean("AU07_r")

        self.out_csv = csv_path
        self.summary = {
            "frames": int(len(df)),
            "dur_s": round(len(df)/30.0, 3),
            "AU12_r": au12,
            "AU04_r": au04,
            "AU26_r": au26,
            "AU07_r": au07,
            "valence_proxy": au12 - au04,
            "arousal_proxy": au07 + au26,
        }
        return self.summary, csv_path
