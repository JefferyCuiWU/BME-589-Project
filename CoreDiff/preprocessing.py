import os
from pathlib import Path
import pydicom
import numpy as np

# Set your HU normalization range (based on CoreDiff defaults or paper)
HU_MIN, HU_MAX = -1000, 2000

def load_ct_series(folder: Path) -> np.ndarray:
    """Load a sorted DICOM series and return a (Z, H, W) volume in Hounsfield Units (HU)."""
    dicoms = sorted(
        (d for d in folder.glob("*.dcm")),
        key=lambda d: getattr(pydicom.dcmread(d, stop_before_pixels=True), "InstanceNumber", 0)
    )
    if not dicoms:
        raise RuntimeError(f"No DICOMs in {folder}")

    slices = []
    for f in dicoms:
        ds = pydicom.dcmread(f)
        try:
            arr = ds.pixel_array.astype(np.int16)
        except Exception as e:
            raise RuntimeError(f"Cannot read {f}: {e}")

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        slices.append(slope * arr + intercept)

    volume = np.stack(slices)  # shape: (Z, H, W)
    return volume

def save_slice_triplets(volume: np.ndarray, out_dir: Path, prefix: str):
    """
    Save overlapping 3-slice triplets from a 3D volume as .npy files.
    Format: {prefix}_{slice:03d}_triplet.npy
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    num_slices = volume.shape[0]

    saved = 0
    for i in range(1, num_slices - 1):
        triplet = np.stack([volume[i - 1], volume[i], volume[i + 1]], axis=0)  # shape: (3, H, W)
        np.save(out_dir / f"{prefix}_{i:03d}_triplet.npy", triplet.astype(np.float32))
        saved += 1

    print(f"  → saved {saved} triplets to {out_dir}")

def process_all_cts(input_root: str, output_root: str):
    """Process all DICOM folders and create CoreDiff-compatible 3-slice .npy files."""
    in_root, out_root = Path(input_root), Path(output_root)
    for ct_dir in sorted(p for p in in_root.iterdir() if p.is_dir()):
        print(f"Processing {ct_dir.name} …")
        try:
            vol = load_ct_series(ct_dir)
        except RuntimeError as e:
            print(f"  !! {e}")
            continue

        out_path = out_root / ct_dir.name
        save_slice_triplets(vol, out_path, prefix=ct_dir.name)

if __name__ == "__main__":
    # Example usage
    process_all_cts("D:/College/Study/25 Spring/BME 589/CLIPDenoising/CT_data", "corediff_inputs")
