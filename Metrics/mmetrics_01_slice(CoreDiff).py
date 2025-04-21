# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 18:31:34 2025

@author: cjj18
"""

# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
ORIG_DIR = Path("D:/College/Study/25 Spring/BME 589/BME-589-Project/axial_npy")
DENO_DIR = Path("D:/College/Study/25 Spring/BME 589/CoreDiff/results")
PATIENTS = [f"CT{i}" for i in range(10)]
HU_WINDOW = (-1000, 400)
AX_RE = re.compile(r"ax_(\d{3})\.npy$")   # three‑digit index

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def hu_to_u8(arr: np.ndarray, lo: int, hi: int) -> np.ndarray:
    arr = np.clip(arr, lo, hi)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def auto_u8(arr: np.ndarray) -> np.ndarray:
    if arr.max() - arr.min() > 1.5:  # treat as HU
        return hu_to_u8(arr, *HU_WINDOW)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def sobel_mag(u8: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(u8, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(u8, cv2.CV_32F, 0, 1, 3)
    return cv2.magnitude(gx, gy)


def metrics(x8: np.ndarray, y8: np.ndarray) -> Dict[str, float]:
    res = x8.astype(np.int16) - y8.astype(np.int16)
    sx, sy, sr = x8.std(ddof=0), y8.std(ddof=0), res.std(ddof=0)
    nrr = 1.0 - sy / sx if sx else 0.0
    gx, gy = sobel_mag(x8), sobel_mag(y8)
    aag = gy.mean() / gx.mean() if gx.mean() else 0.0
    l2px = np.linalg.norm(res) / res.size
    dot = float((gx.ravel() @ gy.ravel()))
    denom = math.sqrt(float((gx**2).sum() * (gy**2).sum()))
    gcos = dot / denom if denom else 0.0
    return {"sigma_res": sr, "NRR": nrr, "AAG_ratio": aag, "L2_per_px": l2px, "grad_cos": gcos}

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def process_patient(pid: str, rows: List[Dict[str, float]], acc: Dict[str, Dict[str, list]]):
    p_orig, p_deno = ORIG_DIR / pid, DENO_DIR / pid
    if not p_orig.is_dir() or not p_deno.is_dir():
        print(f"[WARN] {pid}: missing folder(s)")
        return

    for f_orig in sorted(p_orig.glob("ax_*.npy")):
        m = AX_RE.match(f_orig.name)
        if not m:
            continue
        idx3 = m.group(1)               # e.g. '007'
        idx4 = idx3.zfill(4)            # '0007'
        f_deno = p_deno / f"ct_{idx4}.npy"
        if not f_deno.exists():
            print(f"[WARN] {pid}: missing denoised slice {f_deno.name}")
            continue

        x_hu = np.load(f_orig)
        y_raw = np.load(f_deno)
        if x_hu.ndim != 2 or y_raw.ndim != 2:
            continue

        x8 = hu_to_u8(x_hu, *HU_WINDOW)
        y8 = auto_u8(y_raw)
        mvals = metrics(x8, y8)
        mvals.update({"patient": pid, "slice_id": idx3})
        rows.append(mvals)

        ac = acc.setdefault(pid, {k: [] for k in mvals if k not in ("patient", "slice_id")})
        for k in ac:
            ac[k].append(mvals[k])


def main():
    slice_rows: List[Dict[str, float]] = []
    accum: Dict[str, Dict[str, list]] = {}
    for pid in PATIENTS:
        process_patient(pid, slice_rows, accum)

    pd.DataFrame(slice_rows).to_excel("ldct_pair_slice_scores.xlsx", index=False)
    print("Per‑slice scores → ldct_pair_slice_scores.xlsx")

    summary = []
    for pid, vals in accum.items():
        row = {"patient": pid}
        for k, lst in vals.items():
            row[f"{k}_mean"] = float(np.mean(lst))
            row[f"{k}_median"] = float(np.median(lst))
        summary.append(row)
    pd.DataFrame(summary).to_excel("ldct_pair_patient_summary.xlsx", index=False)
    print("Per‑patient summary → ldct_pair_patient_summary.xlsx")


if __name__ == "__main__":
    main()
