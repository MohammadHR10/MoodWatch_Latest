# app/baseline.py
from __future__ import annotations
from pathlib import Path
import json
from statistics import mean, pstdev
import pandas as pd

BASELINE_DIR = Path("baselines")
BASELINE_DIR.mkdir(exist_ok=True)

# Core AUs we rely on + our simple proxies.
DEFAULT_FEATURES = [
    "AU12_r",          # smile
    "AU04_r",          # brow-furrow
    "AU07_r",          # lid-tighten
    "AU26_r",          # jaw-drop
    "AU15_r",          # lip corner depress (for hesitation)
    "AU20_r",          # lip stretch (for hesitation)
    "valence_proxy",   # AU12 - AU04
    "energy_proxy",    # AU07 + AU26
]

def _path(user: str) -> Path:
    safe = "".join(c for c in user if c.isalnum() or c in "-_").strip() or "default"
    return BASELINE_DIR / f"{safe}.json"

def load(user: str) -> dict:
    p = _path(user)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"samples": [], "mean": {}, "std": {}}

def save(user: str, state: dict) -> None:
    _path(user).write_text(json.dumps(state, indent=2))

def _get_num(row: pd.Series, name: str) -> float | None:
    """
    Try to pull a numeric value for `name` from the row.
    - Tolerates a leading space in the column name (e.g., " AU12_r").
    - Returns None if value is non-numeric or missing.
    """
    for k in (name, f" {name}"):
        if k in row.index:
            val = pd.to_numeric(row[k], errors="coerce")
            if pd.notna(val):
                try:
                    return float(val)
                except Exception:
                    return None
    return None

def _ensure_proxies(feats: dict) -> dict:
    """Ensure valence_proxy and energy_proxy exist; tolerate missing keys."""
    out = dict(feats)
    au12 = float(out.get("AU12_r", 0.0))
    au04 = float(out.get("AU04_r", 0.0))
    au07 = float(out.get("AU07_r", 0.0))
    au26 = float(out.get("AU26_r", 0.0))
    out["valence_proxy"] = out.get("valence_proxy", au12 - au04)
    out["energy_proxy"]  = out.get("energy_proxy",  au07 + au26)
    return out

def add_sample(user: str, feats: dict, keep_last: int = 50) -> dict:
    """Append one sample for a user, then recompute mean/std for DEFAULT_FEATURES."""
    st = load(user)
    feats = _ensure_proxies(feats)

    row = {k: float(feats.get(k, 0.0)) for k in DEFAULT_FEATURES}
    st["samples"].append(row)
    st["samples"] = st["samples"][-keep_last:]

    # recompute baseline mean/std (population std; guard against 0)
    for k in DEFAULT_FEATURES:
        xs = [float(s.get(k, 0.0)) for s in st["samples"]]
        if xs:
            st["mean"][k] = mean(xs)
            st["std"][k]  = max(pstdev(xs), 1e-6)
        else:
            st["mean"][k] = 0.0
            st["std"][k]  = 1.0

    save(user, st)
    return st

def zscore(user: str, feats: dict) -> dict:
    """Return z-scores per DEFAULT_FEATURES. Also annotate with _has_baseline and _n."""
    st = load(user)
    feats = _ensure_proxies(feats)
    out = {}
    for k in DEFAULT_FEATURES:
        mu = float(st["mean"].get(k, 0.0))
        sd = float(st["std"].get(k, 1.0)) or 1.0
        out[k] = (float(feats.get(k, 0.0)) - mu) / sd
    n = len(st.get("samples", []))
    out["_has_baseline"] = n >= 5
    out["_n"] = n
    return out

def seed_from_session(session_csv: Path, user: str = "default", keep_last: int = 50) -> dict:
    """
    Optional: prime the baseline from an existing session_summary.csv.
    Reads the last `keep_last` rows and treats each as one sample.
    Robust to non-numeric columns (e.g., 'expr' == 'happy') and leading-space headers.
    """
    st = {"samples": [], "mean": {}, "std": {}}
    if not session_csv or not Path(session_csv).exists():
        save(user, st); return st

    try:
        df = pd.read_csv(session_csv)
    except Exception:
        save(user, st); return st

    if df.empty:
        save(user, st); return st

    rows = []
    # Pull DEFAULT_FEATURES and AUs we care about; compute proxies per-row.
    au_needed = ["AU12_r", "AU04_r", "AU07_r", "AU26_r", "AU15_r", "AU20_r"]
    for _, r in df.tail(keep_last).iterrows():
        feats = {}
        for k in set(DEFAULT_FEATURES + au_needed):
            val = _get_num(r, k)
            if val is not None:
                feats[k] = float(val)
        feats = _ensure_proxies(feats)
        rows.append({k: float(feats.get(k, 0.0)) for k in DEFAULT_FEATURES})

    st["samples"] = rows[-keep_last:]

    # recompute mean/std
    for k in DEFAULT_FEATURES:
        xs = [float(s.get(k, 0.0)) for s in st["samples"]]
        if xs:
            st["mean"][k] = mean(xs)
            st["std"][k]  = max(pstdev(xs), 1e-6)
        else:
            st["mean"][k] = 0.0
            st["std"][k]  = 1.0

    save(user, st)
    return st