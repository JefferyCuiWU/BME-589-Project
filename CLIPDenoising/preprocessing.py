import os
from pathlib import Path
import pydicom
import numpy as np
from PIL import Image

HU_MIN, HU_MAX = -1000, 2000  # window for normalization

def load_ct_series(folder: Path) -> np.ndarray:
    """Return a (Z,H,W) volume in HU or raise RuntimeError."""
    dicoms = sorted(
        (d for d in folder.glob("*.dcm")),
        key=lambda d: getattr(pydicom.dcmread(d, stop_before_pixels=True), "InstanceNumber", 0)
    )
    if not dicoms:
        raise RuntimeError(f"No DICOMs in {folder}")

    arrays = []
    for f in dicoms:
        ds = pydicom.dcmread(f)
        try:
            arr = ds.pixel_array.astype(np.int16)
        except Exception as e:
            raise RuntimeError(f"Cannot read {f}: {e}")

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arrays.append(slope * arr + intercept)

    return np.stack(arrays)  # shape: (Z, H, W)

def normalise(img: np.ndarray) -> np.ndarray:
    """Normalize HU slice to [0, 255] uint8 for PNG saving."""
    img = np.clip(img, HU_MIN, HU_MAX)
    return ((img - HU_MIN) / (HU_MAX - HU_MIN) * 255).astype(np.uint8)

def save_png_and_npy_slices(volume: np.ndarray, out_dir: Path, prefix: str):
    """Save each slice in PNG and NPY format."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, sl in enumerate(volume):
        img_name = f"{prefix}_{i:03d}"
        # Save PNG
        Image.fromarray(normalise(sl)).save(out_dir / f"{img_name}.png")
        # Save raw HU slice as .npy (float32)
        np.save(out_dir / f"{img_name}.npy", sl.astype(np.float32))
    print(f"  → saved {i+1} PNGs and NPYs to {out_dir}")

def process_all_cts(input_root: str, output_root: str):
    in_root, out_root = Path(input_root), Path(output_root)
    for ct_dir in sorted(p for p in in_root.iterdir() if p.is_dir()):
        print(f"Processing {ct_dir.name} …")
        try:
            vol = load_ct_series(ct_dir)
        except RuntimeError as e:
            print(f"  !! {e}")
            continue
        save_png_and_npy_slices(vol, out_root / ct_dir.name, ct_dir.name)

if __name__ == "__main__":
    process_all_cts("CT_data", r"inputs/LDCT")
