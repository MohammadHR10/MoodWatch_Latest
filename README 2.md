Here is the updated version of the **MoodWatch 1.4** README with your changes. You can directly copy and paste it into your `README.md`:

---

# **MoodWatch 1.4 (Streamlined, Local + Baseline-Aware)**

## **What’s New in 1.4**

- ✅ **All-local**: No Gemini/API keys.
- 🎬 **Record inside Streamlit** (no separate script needed).
- 🧮 Plain-language metrics: **Positivity**, **Energy**, **Tension**, **Hesitation**, **Confidence**.
- 🧠 **Per-user baseline** that adapts after a few pulses.
- 🔎 **Transparent signals** panel showing AUs + derived features.
- 📁 Drop-in with your existing `processed/` CSVs and `session_summary.csv`.

---

## **Prerequisites**

- **Python 3.10–3.12**
- OpenCV (camera access required)
- **OpenFace** built with `FeatureExtraction` binary (with models downloaded)

> **macOS Tip**: Grant camera permissions under **System Settings → Privacy & Security → Camera** for Terminal/IDE/VS Code.

---

## **Install**

```bash
# 1) Unzip and enter the folder
cd MoodWatch-1.4

# 2) Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Set OPENFACE_BIN in .env
cp .env.example .env
# Edit .env and set:
#   OPENFACE_BIN=/absolute/path/to/OpenFace/build/bin/FeatureExtraction

# Verify OpenFace path
$ $OPENFACE_BIN -h
# Should display FeatureExtraction help
```

---

## **Run**

### Streamlit UI

```bash
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

- Use the **🎬 Record Pulse** tab to capture a 3–10s clip and analyze it.
- Switch to the **📁 Analyze Existing CSV** tab to pick files from `processed/`.

### Optional: Headless Periodic Capture

```bash
# Terminal 1 — periodic capture
source .venv/bin/activate
python -m app.camera_schedule

# Terminal 2 — view results in UI
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

---

## **Dataset Setup (DISFA+)**

### **1. Upload Dataset to OneDrive**

1. Upload **SN Dataset** folders (SN001, SN001 2, SN001 3) to OneDrive/SharePoint.
2. Set folders to **"Always keep on this device"** for offline sync.
3. Wait for the dataset to sync completely to your local machine.

### **2. Locate Synced Dataset**

```bash
# Example OneDrive paths:
# OneDrive: /Users/[username]/Library/CloudStorage/OneDrive-[org]/SN/SN Dataset/
# Or: /Users/[username]/OneDrive/SN Dataset/

# Create a symlink for easy access
ln -sf "/path/to/synced/SN Dataset" ~/sn_dataset
```

### **3. Verify Dataset Structure**

```bash
# Check if the dataset is synced correctly
ls -la ~/sn_dataset/
# Should show: SN001, SN001 2, SN001 3

# Verify trial data exists
ls -la ~/sn_dataset/SN001/ | head -10
# Should show: A1_AU1_TrailNo_1, B2_AU6_TrailNo_2, etc.
```

### **4. Run Dataset Demo**

```bash
# Run DISFA+ dataset analysis
python -m app.demo_disfa_dataset \
  --images ~/sn_dataset/SN001 \
  --landmarks ~/sn_dataset/SN001 \
  --labels ~/sn_dataset/SN001

# For other variants:
python -m app.demo_disfa_dataset \
  --images ~/sn_dataset/"SN001 2" \
  --landmarks ~/sn_dataset/"SN001 2" \
  --labels ~/sn_dataset/"SN001 2"
```

---

## **Role of `demo_disfa_dataset.py`**

The `demo_disfa_dataset.py` script is used for testing and analyzing the **DISFA+** dataset. It allows you to pass image, landmark, and label directories and run a demo analysis of facial action units (AUs). The results will be shown in the transparent signals panel within the app.

---

## **Folder Layout**

```
MoodWatch-1.4/
  app/
    __init__.py
    streamlit_app.py      # Streamlit UI (record & analyze)
    camera_schedule.py    # Headless periodic capture (optional)
    mood_logic.py         # Local, explainable heuristics
    baseline.py           # Per-user baseline helpers
  baselines/              # Per-user baseline JSONs (e.g., default.json)
  processed/              # OpenFace CSV outputs live here
  session_summary.csv     # Appends rollup data (after first pulse)
  requirements.txt
  .env.example  -> copy to .env and set OPENFACE_BIN
  README.md
```

---

## **Personalization (Baseline)**

- The app maintains a lightweight **per-user baseline** in `baselines/<user>.json`.
- On startup, it **seeds from `session_summary.csv`** if present (last \~50 rows).
- After **≥5 pulses**, scores are normalized to the user (z-scores), reducing false positives.
- Each new analysis **updates the baseline** automatically.
- To reset, delete `baselines/default.json`.

> From a **cold start**, results may appear generic until you record \~5–10 pulses.

---

## **How the Heuristics Work (Plain Language)**

We keep things simple and transparent:

- **Positivity** ≈ AU12_r (smile) − AU04_r (brow-furrow) _(personalized after baseline is learned)_
- **Energy** ≈ AU07_r (lid-tighten) + AU26_r (mouth-open)
- **Tension**: weighted AU04/AU07/AU23 (also considers head steadiness)
- **Hesitation**: AU15/AU20 vs. smile/activation
- **Confidence**: higher with more frames, steadier head pose, and consistent signals

The app also displays an overall label (e.g., engaged, calm, tense).
You can adjust thresholds or weights inside `app/mood_logic.py`.

---

## **Troubleshooting**

- **ModuleNotFoundError: No module named 'app'**
  Ensure `app/__init__.py` exists and run from the project root:

  ```bash
  touch app/__init__.py
  streamlit run app/streamlit_app.py
  ```

- **Camera won’t open (macOS AVFoundation)**
  Trigger the prompt once, then grant permissions in **System Settings → Privacy & Security → Camera** for your terminal/IDE.

- **OpenFace not found or not executable**
  Set `OPENFACE_BIN` correctly in `.env` and make it executable:

  ```bash
  chmod +x /path/to/FeatureExtraction
  ```

- **No CSVs appear**
  Ensure a pulse completes (either via the Record tab or `python -m app.camera_schedule`). Outputs will be saved to `processed/`.

---

## **Accuracy Notes & Next Steps**

This build is intentionally simple and explainable. For a data-science upgrade (optional):

- Train a lightweight model on public AU-labeled datasets (e.g., DISFA, BP4D, CK+, Aff-Wild2) → export a `.pkl`.
- Replace/augment the rules in `app/mood_logic.py` with your classifier/regressor (scikit-learn / lightgbm).
- Keep the transparency panel so users can see which signals influenced the result.

---

## **License**

This template is provided as-is for personal/research use. Follow OpenFace’s license for their models/tools.

---

You can now simply copy and paste this into your `README.md`.
