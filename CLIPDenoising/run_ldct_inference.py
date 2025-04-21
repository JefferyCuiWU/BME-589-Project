# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 10:48:25 2025

@author: cjj18
"""
import sys
from pathlib import Path
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'basicsr/models/archs'))
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

from CLIPDenoising_arch import CLIPDenoising

# ========== Config ==========
INPUT_ROOT = Path("inputs/LDCT")
OUTPUT_ROOT = Path("results_denoised")
OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)

MODEL_PATH = Path("experiments/pretrained_models/net_g_latest.pth")
CLIP_PATH = Path("model_data/clip/RN50.pt")  # Adjust as needed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Load Model ==========
print("Loading model...")

model = CLIPDenoising(
    inp_channels=1,
    out_channels=1,
    depth=5,
    wf=64,
    num_blocks=[3, 4, 6, 3],
    bias=False,
    model_path=str(CLIP_PATH),
    aug_level=0.025,
)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['params'])
model = model.to(DEVICE)
model.eval()

# ========== Inference ==========
def denoise_patient(patient_dir: Path, save_dir: Path):
    """Process one CT folder of .npy slices and save the denoised 3D volume"""
    slice_paths = sorted(patient_dir.glob("*.npy"))
    denoised_slices = []

    for sl_path in tqdm(slice_paths, desc=f"Inferencing {patient_dir.name}"):
        hu_slice = np.load(sl_path)[..., np.newaxis]  # shape: (H, W, 1)
        img_tensor = torch.from_numpy(hu_slice).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)  # [1, 1, H, W]

        h, w = img_tensor.shape[2:]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        with torch.no_grad():
            output = model(img_tensor)

        output = output[:, :, :h, :w]  # remove padding
        output_np = output.squeeze().cpu().numpy()
        denoised_slices.append(output_np)

    # Save full 3D volume
    full_volume = np.stack(denoised_slices, axis=0)
    save_dir.mkdir(parents=True, exist_ok=True)
    volume_path = save_dir / f"{patient_dir.name}_denoised_volume.npy"
    np.save(volume_path, full_volume)
    print(f"  → Saved full volume to {volume_path}")


# ========== Run All ==========
def main():
    for ct_folder in sorted(INPUT_ROOT.iterdir()):
        if ct_folder.is_dir():
            save_folder = OUTPUT_ROOT / ct_folder.name
            denoise_patient(ct_folder, save_folder)

if __name__ == "__main__":
    main()
