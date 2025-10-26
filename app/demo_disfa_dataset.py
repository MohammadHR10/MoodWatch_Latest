# app/demo_disfa_dataset.py
from __future__ import annotations
import argparse, random, os, sys
from pathlib import Path

# Works both with `python -m app.demo_disfa_dataset` and `python app/demo_disfa_dataset.py`
try:
    from .disfa_dataset import SNDataset
except Exception:
    sys.path.append(os.path.dirname(__file__))
    from disfa_dataset import SNDataset  # type: ignore

import numpy as np
import matplotlib.pyplot as plt

def draw(ax, sample):
    im = sample["image"]
    ax.imshow(im)
    ax.axis("off")
    labels = [k for k, v in (sample.get("labels") or {}).items() if v]
    meta = sample.get("meta", {})
    title = f'{meta.get("trial","?")} #{meta.get("frame","?")}  ' + (",".join(labels) or "no AU=1")
    ax.set_title(title, fontsize=8)
    lm = sample.get("landmarks")
    if isinstance(lm, np.ndarray) and lm.size:
        ax.scatter(lm[:, 0], lm[:, 1], s=5)

def main():
    ap = argparse.ArgumentParser(description="Show a few SN/Ext-DISFA samples")
    ap.add_argument("--images", required=True, help="Root of frames (e.g., SN001)")
    ap.add_argument("--landmarks", help="Root of FaceCropped .mat files (e.g., SN001 2)")
    ap.add_argument("--labels", help="Root with AU*.txt per trial (e.g., SN001 3)")
    ap.add_argument("-n", type=int, default=5, help="How many samples to draw")
    args = ap.parse_args()

    ds = SNDataset(images_root=args.images,
                   landmarks_root=args.landmarks,
                   labels_root=args.labels,
                   transform=None,
                   return_landmarks=bool(args.landmarks),
                   return_labels=bool(args.labels))
    print("Dataset size:", len(ds))

    k = min(max(1, args.n), len(ds))
    idxs = random.sample(range(len(ds)), k=k)
    cols = min(5, k); rows = (k + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(3.6*cols, 3.6*rows), dpi=120)
    axs = np.atleast_1d(axs).ravel()

    for i, idx in enumerate(idxs):
        draw(axs[i], ds[idx])
    for j in range(i+1, len(axs)):
        axs[j].axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
