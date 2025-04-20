import os
from pathlib import Path

import numpy as np
import pydicom

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

INPUT_BASE = Path("CT_data")       # folder holding CT0 … CT9 with DICOMs
OUTPUT_BASE = Path("axial_npy")     # where .npy slices will be written
PATIENTS = [f"CT{i}" for i in range(10)]

# -----------------------------------------------------------------------------
# DICOM → volume helper
# -----------------------------------------------------------------------------

def load_ct_series_with_rescale(folder_path: Path) -> np.ndarray:
    """Return 3‑D volume (Z,Y,X) in Hounsfield Units from a DICOM series."""
    dicom_files = sorted(folder_path.glob("*.dcm"))
    if not dicom_files:
        raise FileNotFoundError(f"no DICOM files in {folder_path}")

    slices = []
    for dcm_path in dicom_files:
        ds = pydicom.dcmread(dcm_path)
        if hasattr(ds, "InstanceNumber"):
            slices.append((ds.InstanceNumber, ds))
        else:
            print(f"[WARN] {dcm_path.name} missing InstanceNumber – skipped")

    if not slices:
        raise ValueError(f"no valid slices in {folder_path}")

    slices.sort(key=lambda t: t[0])
    datasets = [t[1] for t in slices]

    volume = np.stack([ds.pixel_array.astype(np.int16) for ds in datasets], axis=0)
    slope = float(datasets[0].RescaleSlope)
    intercept = float(datasets[0].RescaleIntercept)

    return slope * volume + intercept  # HU

# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main():
    OUTPUT_BASE.mkdir(exist_ok=True)

    for patient in PATIENTS:
        in_dir = INPUT_BASE / patient
        if not in_dir.is_dir():
            print(f"[WARN] missing {in_dir}; skipping")
            continue

        try:
            vol = load_ct_series_with_rescale(in_dir)
        except Exception as e:
            print(f"[ERROR] {patient}: {e}")
            continue

        z_dim, y_dim, x_dim = vol.shape
        out_dir = OUTPUT_BASE / patient
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"{patient}: saving {z_dim} axial slices → {out_dir}")
        for z in range(z_dim):
            axial = vol[z, :, :]             # shape (Y, X)
            np.save(out_dir / f"ax_{z:03d}.npy", axial)

    print("Done.")


if __name__ == "__main__":
    main()
