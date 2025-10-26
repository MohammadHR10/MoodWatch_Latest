# app/mood_logic.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import math

@dataclass
class MoodOutput:
    label: str
    positivity: float      # [-1, +1]
    energy: float          # [0, 1]
    tension: float         # [0, 1]
    hesitation: float      # [0, 1]
    notes: str
    confidence: float      # [0, 1]
    features: Dict[str, float]

def _get(d: Dict[str, float], k: str, default=0.0) -> float:
    # Some CSVs have a leading space before AU names; tolerate both.
    return float(d.get(k, d.get(f" {k}", default)))

def _clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))

def _norm01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return _clamp((x - lo) / (hi - lo), 0.0, 1.0)

def _z(x: float, mu: Optional[float], sigma: Optional[float]) -> float:
    if mu is None or sigma is None or sigma <= 1e-6:
        return x
    return (x - mu) / sigma

def infer_mood(
    au_means: Dict[str, float],
    *,
    frames: Optional[int] = None,
    pose_rx: Optional[float] = None,
    pose_ry: Optional[float] = None,
    pose_rz: Optional[float] = None,
    # Option A: per-user baseline stats (mean/std dict, e.g. from seed_from_session)
    baseline: Optional[Dict[str, Dict[str, float]]] = None,
    # Option B: precomputed z-scores from app.baseline.zscore(user, summary)
    z: Optional[Dict[str, float]] = None,
    # Optional exponential smoothing of signals (insert smoothed values here if you have them)
    ema: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Inputs:
      - au_means: per-capture mean AU values (keys like AU12_r, AU04_r, etc.)
      - frames/pose_*: optional metadata for confidence estimation
      - baseline: dict with {'positivity': {'mean','std'}, 'energy': {...}, ...}
      - z: dict with z-scores for keys ['valence_proxy'|'positivity', 'energy_proxy'|'energy', 'tension', 'hesitation'].
           If provided, takes precedence over 'baseline'.
      - ema: optionally provide pre-smoothed values for ['positivity','energy','tension','hesitation'].

    Returns:
      A plain dict suitable for Streamlit display.
    """

    # --- Pull raw AU signals (default 0 if missing) ---
    AU01 = _get(au_means, "AU01_r")   # inner brow raise
    AU02 = _get(au_means, "AU02_r")   # outer brow raise
    AU04 = _get(au_means, "AU04_r")   # brow lower (furrow)
    AU06 = _get(au_means, "AU06_r")   # cheek raise
    AU07 = _get(au_means, "AU07_r")   # lid tighten
    AU09 = _get(au_means, "AU09_r")   # nose wrinkle
    AU10 = _get(au_means, "AU10_r")   # upper lip raise
    AU12 = _get(au_means, "AU12_r")   # smile
    AU14 = _get(au_means, "AU14_r")   # dimpler (can counter AU12)
    AU15 = _get(au_means, "AU15_r")   # lip corner depressor
    AU17 = _get(au_means, "AU17_r")   # chin raise
    AU20 = _get(au_means, "AU20_r")   # lip stretch
    AU23 = _get(au_means, "AU23_r")   # lip tighten
    AU25 = _get(au_means, "AU25_r")   # lips part
    AU26 = _get(au_means, "AU26_r")   # jaw drop
    AU45 = _get(au_means, "AU45_c")   # blink-ish

    # --- Construct interpretable composites (raw, unbounded) ---
    # Positivity proxy: smile vs. brow tension & nose wrinkle
    smile_eff   = AU12 - 0.5 * max(AU14, 0.0) + 0.2 * AU06
    brow_tense  = AU04 + 0.3 * AU01 - 0.2 * AU02
    aversive    = 0.6 * AU09 + 0.2 * AU10
    positivity_raw  = 1.1 * smile_eff - 0.9 * brow_tense - 0.4 * aversive  # can be negative

    # Energy proxy (“activation”): eyes & mouth movement
    mouth_open  = 0.6 * AU26 + 0.4 * AU25
    eye_tense   = AU07 + 0.3 * AU45
    energy_raw  = 0.9 * eye_tense + 0.5 * mouth_open                       # >= 0

    # Tension proxy: furrow + lid tighten + lip tighten
    tension_raw = _clamp(0.8 * AU04 + 0.7 * AU07 + 0.4 * AU23, 0.0, 3.0)

    # Hesitation proxy: lip corner depress (AU15) + lip stretch (AU20) with low smile
    hesitation_raw = _clamp(0.8 * AU15 + 0.5 * AU20 - 0.4 * max(AU12, 0.0), 0.0, 3.0)

    # --- Choose normalization path: precomputed z  > baseline  > raw ---
    if z is not None:
        # Accept a few aliases for convenience
        pos_z = float(z.get("positivity", z.get("valence_proxy", 0.0)))
        eng_z = float(z.get("energy", z.get("energy_proxy", z.get("arousal_proxy", 0.0))))
        ten_z = float(z.get("tension", 0.0))
        hes_z = float(z.get("hesitation", 0.0))
    elif baseline:
        def mu_sd(key):
            s = baseline.get(key, {})
            return s.get("mean"), s.get("std")
        pos_z = _z(positivity_raw, *mu_sd("positivity"))
        eng_z = _z(energy_raw,     *mu_sd("energy"))
        ten_z = _z(tension_raw,    *mu_sd("tension"))
        hes_z = _z(hesitation_raw, *mu_sd("hesitation"))
    else:
        pos_z, eng_z, ten_z, hes_z = positivity_raw, energy_raw, tension_raw, hesitation_raw

    # --- Optional smoothing override (EMA) if caller supplies it ---
    if ema:
        pos_z = ema.get("positivity", pos_z)
        eng_z = ema.get("energy",     eng_z)
        ten_z = ema.get("tension",    ten_z)
        hes_z = ema.get("hesitation", hes_z)

    # --- Map to friendly, bounded scales ---
    positivity_scaled = _clamp(pos_z / 2.0, -1.0, 1.0)          # ~[-1,1]
    energy_scaled     = _clamp(_norm01(eng_z, 0.0, 2.0), 0.0, 1.0)
    tension_scaled    = _clamp(_norm01(ten_z, 0.0, 2.0), 0.0, 1.0)
    hesitation_scaled = _clamp(_norm01(hes_z, 0.0, 2.0), 0.0, 1.0)

    # --- Label logic (quadrant + modifiers), with plain-English wording ---
    if positivity_scaled >= 0.25 and energy_scaled >= 0.5:
        label = "engaged / upbeat"
    elif positivity_scaled >= 0.25 and energy_scaled < 0.5:
        label = "calm / content"
    elif positivity_scaled < -0.25 and energy_scaled >= 0.5:
        label = "tense / frustrated"
    elif positivity_scaled < -0.25 and energy_scaled < 0.5:
        label = "down / withdrawn"
    else:
        label = "neutral / mixed"

    # Modifiers for the notes line
    mods = []
    if tension_scaled >= 0.6:
        mods.append("noticeable tension")
    if hesitation_scaled >= 0.6:
        mods.append("some hesitation")
    notes = "; ".join(mods) if mods else "signals look typical for you"

    # --- Confidence heuristic ---
    pose_mag = 0.0
    for v in (pose_rx, pose_ry, pose_rz):
        if v is not None:
            pose_mag += float(v) ** 2
    pose_mag = math.sqrt(pose_mag)
    frames_ok = _norm01((frames or 0), 60, 180)                 # ~90 frames for 6s@15fps
    pose_ok   = 1.0 - _norm01(pose_mag, 0.25, 0.6)              # prefer small head motion
    energy_ok = 1.0 - abs(energy_scaled - 0.5) * 2 / 2          # bell-ish around mid
    confidence = _clamp(0.5 * frames_ok + 0.3 * pose_ok + 0.2 * energy_ok, 0.0, 1.0)

    out = MoodOutput(
        label=label,
        positivity=positivity_scaled,
        energy=energy_scaled,
        tension=tension_scaled,
        hesitation=hesitation_scaled,
        notes=notes,
        confidence=confidence,
        features={
            # Expose internals for transparency/debug
            "smile_eff": smile_eff,
            "brow_tense": brow_tense,
            "mouth_open": mouth_open,
            "eye_tense": eye_tense,
            "positivity_raw": positivity_raw,
            "energy_raw": energy_raw,
            "tension_raw": tension_raw,
            "hesitation_raw": hesitation_raw,
        },
    )

    return {
        "label": out.label,
        "positivity": round(out.positivity, 3),
        "energy": round(out.energy, 3),
        "tension": round(out.tension, 3),
        "hesitation": round(out.hesitation, 3),
        "confidence": round(out.confidence, 3),
        "notes": out.notes,
        "features": {k: round(v, 3) for k, v in out.features.items()},
    }