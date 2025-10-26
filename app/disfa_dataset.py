# disfa_dataset.py
"""
SN/DISFA-like dataset loader that joins:
  - Images in per-trial folders (…/TrialNo_X/###.jpg)
  - Landmarks in .mat files (…/<TrialName>_FaceCropped.mat)
  - AU label text files (…/<TrialName>/AUxx.txt with '000.jpg 0/1' rows)

Works from:
  - Google Drive (Colab mount):   /content/drive/MyDrive/<...>
  - Local mirror (fastest):       /content/data/<...>  or any local path

Returns each sample as:
  {
    "image": PIL.Image or transformed tensor,
    "landmarks": np.ndarray [N,2] or None,
    "labels": dict {"AU01":0/1,...} or {},
    "meta": {"trial": str, "frame": int, "img_path": str}
  }

Minimal deps: pillow, numpy, pandas, scipy (for .mat),
optionally torch/torchvision transforms if you use transform.
"""

from __future__ import annotations
import os
import re
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from PIL import Image
from scipy.io import loadmat

try:
    import torch
    from torch.utils.data import Dataset
except Exception:
    # Let it still work without torch; only if you want DataLoader you need torch installed.
    class Dataset:  # type: ignore
        pass
    torch = None  # type: ignore


_NUM_RE = re.compile(r"(\d+)\.jpg$", re.IGNORECASE)


def _frame_from_path(p: Path) -> Optional[int]:
    m = _NUM_RE.search(p.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _trial_name_from_image_dir(dir_path: Path) -> str:
    # Use the last directory name as the trial key, e.g. "Z_SupriseText_TrailNo_2"
    return dir_path.name


def _safe_load_image(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def _pick_landmark_array(md: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Heuristics to find the landmark array inside a .mat dict.
    Looks for one of:
      - (T, N, 2)
      - (T, 2, N) -> transpose to (T, N, 2)
      - (T, 2*N)  -> reshape to (T, N, 2) if N integer
    Chooses the 'most plausible' candidate if several exist.
    """
    candidates: List[Tuple[str, np.ndarray]] = []
    for k, v in md.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray):
            arr = np.array(v)
            if arr.ndim in (2, 3):
                candidates.append((k, arr))

    # Prefer 3D arrays with last dim==2 or second dim==2
    def score(arr: np.ndarray) -> int:
        if arr.ndim == 3 and (arr.shape[-1] == 2 or arr.shape[1] == 2):
            return 3
        if arr.ndim == 2 and arr.shape[1] % 2 == 0:
            return 2
        return 1

    candidates.sort(key=lambda kv: score(kv[1]), reverse=True)
    for name, arr in candidates:
        a = np.array(arr)
        if a.ndim == 3:
            # (T, N, 2) OK
            if a.shape[-1] == 2:
                return a.astype(np.float32)
            # (T, 2, N) -> (T, N, 2)
            if a.shape[1] == 2:
                return np.transpose(a, (0, 2, 1)).astype(np.float32)
        elif a.ndim == 2:
            # (T, 2*N) -> (T, N, 2)
            T, D = a.shape
            if D % 2 == 0:
                N = D // 2
                return a.reshape(T, N, 2).astype(np.float32)
    return None


class SNDataset(Dataset):
    """
    images_root:    path to folder that contains per-trial subfolders full of .jpg
                    e.g. ".../SN Dataset/SN001"
    landmarks_root: path to folder that contains "<TrialName>_FaceCropped.mat" files
                    e.g. ".../SN Dataset/SN001 2"
    labels_root:    path to folder that contains per-trial subfolders with AU*.txt,
                    OR the AU*.txt can be colocated in images_root/<TrialName>
    """

    def __init__(
        self,
        images_root: str | Path,
        landmarks_root: Optional[str | Path] = None,
        labels_root: Optional[str | Path] = None,
        transform=None,
        return_landmarks: bool = True,
        return_labels: bool = True,
        trials_include: Optional[List[str]] = None,
        trials_exclude: Optional[List[str]] = None,
        image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ):
        self.images_root = Path(images_root)
        self.landmarks_root = Path(landmarks_root) if landmarks_root else None
        self.labels_root = Path(labels_root) if labels_root else None
        self.transform = transform
        self.return_landmarks = return_landmarks
        self.return_labels = return_labels
        self.image_exts = tuple([e.lower() for e in image_exts])

        # Build index from images
        self.samples: List[Dict[str, Any]] = []
        self.trials: List[str] = []
        self._build_index(trials_include, trials_exclude)

        # Caches keyed by trial name
        self._lm_cache: Dict[str, np.ndarray] = {}
        self._labels_cache: Dict[str, Dict[int, Dict[str, int]]] = {}

    # ---------- index ----------
    def _build_index(self, include: Optional[List[str]], exclude: Optional[List[str]]):
        if not self.images_root.exists():
            raise FileNotFoundError(f"images_root not found: {self.images_root}")

        for trial_dir in sorted([p for p in self.images_root.iterdir() if p.is_dir()]):
            trial = _trial_name_from_image_dir(trial_dir)
            if include and trial not in include:
                continue
            if exclude and trial in exclude:
                continue

            img_paths = []
            for ext in self.image_exts:
                img_paths += sorted(trial_dir.glob(f"*{ext}"))
            if not img_paths:
                # No images, skip trial
                continue

            added_any = False
            for ip in img_paths:
                frame = _frame_from_path(ip)
                if frame is None:
                    continue
                self.samples.append({
                    "img_path": str(ip),
                    "trial": trial,
                    "frame": frame
                })
                added_any = True

            if added_any:
                self.trials.append(trial)

        if not self.samples:
            raise RuntimeError(f"No images found under {self.images_root} with exts={self.image_exts}")

    # ---------- landmarks ----------
    def _landmark_mat_for_trial(self, trial: str) -> Optional[Path]:
        if not self.landmarks_root:
            return None
        # Expected pattern: "<TrialName>_FaceCropped.mat"
        cand = self.landmarks_root / f"{trial}_FaceCropped.mat"
        if cand.exists():
            return cand
        # fallback: any .mat containing the trial name
        hits = list(self.landmarks_root.glob(f"*{trial}*.mat"))
        return hits[0] if hits else None

    def _load_landmarks_for_trial(self, trial: str) -> Optional[np.ndarray]:
        if trial in self._lm_cache:
            return self._lm_cache[trial]
        mp = self._landmark_mat_for_trial(trial)
        if not mp or not mp.exists():
            self._lm_cache[trial] = None  # type: ignore
            return None
        try:
            md = loadmat(str(mp))
            arr = _pick_landmark_array(md)
            self._lm_cache[trial] = arr  # may be None
            return arr
        except Exception:
            self._lm_cache[trial] = None  # type: ignore
            return None

    # ---------- labels ----------
    def _labels_dir_for_trial(self, trial: str) -> Optional[Path]:
        # Try labels_root/<TrialName> first if labels_root provided
        if self.labels_root:
            d = self.labels_root / trial
            if d.exists():
                return d
        # Else, try colocated under images_root/<TrialName>
        d2 = self.images_root / trial
        if d2.exists():
            # if AU*.txt are present here, use it
            if any(d2.glob("AU*.txt")):
                return d2
        return None

    @staticmethod
    def _read_au_txt(path: Path) -> Dict[int, int]:
        """
        Parse AU text file format like:
           000.jpg 0
           001.jpg 1
        Returns {frame:int -> value:int}
        """
        out: Dict[int, int] = {}
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                fname, val = parts[0], parts[1]
                m = _NUM_RE.search(fname)
                if not m:
                    continue
                try:
                    frame = int(m.group(1))
                    out[frame] = int(float(val))
                except Exception:
                    continue
        return out

    def _load_labels_for_trial(self, trial: str) -> Dict[int, Dict[str, int]]:
        """
        Returns mapping: frame -> { 'AU01':0/1, 'AU02':..., ... }
        Cached per trial.
        """
        if trial in self._labels_cache:
            return self._labels_cache[trial]

        out: Dict[int, Dict[str, int]] = {}
        d = self._labels_dir_for_trial(trial)
        if not d:
            self._labels_cache[trial] = out
            return out

        au_files = sorted(d.glob("AU*.txt"))
        for af in au_files:
            au_name = af.stem.upper()  # e.g., "AU05"
            # Normalize to "AU05" formatting
            m = re.match(r"(AU)(\d+)", au_name)
            if not m:
                continue
            au_key = f"AU{int(m.group(2)):02d}"
            series = self._read_au_txt(af)
            for frame, val in series.items():
                out.setdefault(frame, {})[au_key] = int(val)

        self._labels_cache[trial] = out
        return out

    # ---------- dataset API ----------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        img_path = Path(s["img_path"])
        trial = s["trial"]
        frame = s["frame"]

        img = _safe_load_image(img_path)

        sample: Dict[str, Any] = {
            "image": img,
            "meta": {"trial": trial, "frame": int(frame), "img_path": str(img_path)}
        }

        if self.return_landmarks:
            lm_all = self._load_landmarks_for_trial(trial)
            lm = None
            if lm_all is not None:
                # Assume 1-based frame index commonly used; clamp to range
                fidx = max(1, int(frame))
                if fidx <= lm_all.shape[0]:
                    lm = lm_all[fidx - 1]  # [N,2]
            sample["landmarks"] = lm  # np.ndarray or None

        if self.return_labels:
            labels_map = self._load_labels_for_trial(trial)
            sample["labels"] = labels_map.get(int(frame), {})
        else:
            sample["labels"] = {}

        if self.transform is not None:
            sample["image"] = self.transform(sample["image"])

        return sample


# ---------------- Convenience: simple test ----------------
if __name__ == "__main__":
    # Edit these to your environment. Examples for Colab (Drive mounted):
    EXAMPLES = {
        # Fast local mirror (recommended for training):
        # "images_root": "/content/data/images/SN001",
        # "landmarks_root": "/content/data/landmarks/SN001 2",
        # "labels_root": "/content/data/labels",

        # Direct from Drive (slower but fine for smoke tests):
        "images_root": "/content/drive/MyDrive/SN Dataset/SN001",
        "landmarks_root": "/content/drive/MyDrive/SN Dataset/SN001 2",
        "labels_root": "/content/drive/MyDrive/SN Dataset/Labels",
    }

    ds = SNDataset(
        images_root=EXAMPLES["images_root"],
        landmarks_root=EXAMPLES["landmarks_root"],
        labels_root=EXAMPLES["labels_root"],
        transform=None,
        return_landmarks=True,
        return_labels=True,
    )
    print("Samples:", len(ds))
    x = ds[0]
    print("First sample meta:", x["meta"])
    print("Image size:", x["image"].size)
    lm = x.get("landmarks")
    if lm is not None:
        print("Landmarks shape:", lm.shape)
    print("Labels:", x.get("labels", {}))
