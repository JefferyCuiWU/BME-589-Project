# -*- coding: utf-8 -*-

import math
import re
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ORIG_DIR = Path("D:/College/Study/25 Spring/BME 589/BME-589-Project/axial_npy")   # per‑slice HU .npy
DENO_DIR = Path("D:/College/Study/25 Spring/BME 589/CLIPDenoising/results_denoised")   # one volume .npy per patient
PATIENTS = [f"CT{i}" for i in range(10)]
HU_WINDOW = (-1000, 400)
SLICE_PATTERN = re.compile(r"ax_(\d{3})\.npy$")

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def window_uint8(arr: np.ndarray, lo: int, hi: int) -> np.ndarray:
    arr = np.clip(arr, lo, hi)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def sobel_grad(u8: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(u8, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(u8, cv2.CV_32F, 0, 1, 3)
    return cv2.magnitude(gx, gy)


def metrics_pair(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    res = x.astype(np.int16) - y.astype(np.int16)
    sigma_x, sigma_y, sigma_res = x.std(ddof=0), y.std(ddof=0), res.std(ddof=0)
    nrr = 1.0 - sigma_y / sigma_x if sigma_x else 0.0
    gx, gy = sobel_grad(x), sobel_grad(y)
    aag_ratio = gy.mean() / gx.mean() if gx.mean() else 0.0
    l2px = np.linalg.norm(res) / res.size
    dot = float((gx.ravel() @ gy.ravel()))
    denom = math.sqrt(float((gx**2).sum() * (gy**2).sum()))
    grad_cos = dot / denom if denom else 0.0
    return {
        "sigma_res": sigma_res,
        "NRR": nrr,
        "AAG_ratio": aag_ratio,
        "L2_per_px": l2px,
        "grad_cos": grad_cos,
    }

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def process_patient(patient: str, slice_records: List[Dict[str, float]], patient_accum: Dict[str, Dict[str, list]]):
    p_orig = ORIG_DIR / patient
    p_deno = DENO_DIR / patient

    if not p_orig.is_dir():
        print(f"[WARN] {patient}: missing original folder")
        return

    # Load denoised volume ----------------------------------------------------
    deno_files = list(p_deno.glob("*.npy"))
    if len(deno_files) != 1:
        print(f"[WARN] {patient}: expected 1 volume in {p_deno}, found {len(deno_files)}")
        return
    vol_hu = np.load(deno_files[0])
    if vol_hu.ndim != 3:
        print(f"[WARN] {patient}: volume is not 3‑D")
        return

    # Iterate original slices -------------------------------------------------
    for f_orig in sorted(p_orig.glob("ax_*.npy")):
        m = SLICE_PATTERN.search(f_orig.name)
        if not m:
            continue
        z_idx = int(m.group(1))
        if z_idx >= vol_hu.shape[0]:
            print(f"[WARN] {patient}: slice {z_idx} outside deno volume")
            continue

        x_hu = np.load(f_orig)
        y_hu = vol_hu[z_idx]
        if x_hu.ndim != 2 or y_hu.ndim != 2:
            print(f"[WARN] {f_orig.name}: slice not 2‑D; skipped")
            continue

        x_u8 = window_uint8(x_hu, *HU_WINDOW)
        y_u8 = window_uint8(y_hu, *HU_WINDOW)
        mvals = metrics_pair(x_u8, y_u8)
        mvals.update({"patient": patient, "slice_id": f_orig.stem})
        slice_records.append(mvals)

        acc = patient_accum.setdefault(patient, {k: [] for k in mvals if k not in ("patient", "slice_id")})
        for k in acc:
            acc[k].append(mvals[k])


def main() -> None:
    slice_records: List[Dict[str, float]] = []
    patient_accum: Dict[str, Dict[str, list]] = {}

    for patient in PATIENTS:
        process_patient(patient, slice_records, patient_accum)

    # Save -------------------------------------------------------------------
    pd.DataFrame(slice_records).to_excel("ldct_pair_slice_scores.xlsx", index=False)
    print("Per‑slice scores → ldct_pair_slice_scores.xlsx")

    summary = []
    for pat, vals in patient_accum.items():
        row = {"patient": pat}
        for k, lst in vals.items():
            row[f"{k}_mean"] = float(np.mean(lst))
            row[f"{k}_median"] = float(np.median(lst))
        summary.append(row)
    pd.DataFrame(summary).to_excel("ldct_pair_patient_summary_CLIP.xlsx", index=False)
    print("Per‑patient summary → ldct_pair_patient_summary_CLIP.xlsx")


if __name__ == "__main__":
    main()
