# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 18:21:58 2025

@author: cjj18
"""

import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt

def load_ct_series_with_rescale(folder_path):
    """
    Load a series of CT DICOM images from a folder, sort them, convert to a 3D volume,
    and apply RescaleSlope and RescaleIntercept to convert to Hounsfield Units (HU).

    Parameters:
        folder_path (str): Path to the folder containing CT DICOM files.

    Returns:
        volume_hu (np.ndarray): A 3D NumPy array (slices, height, width) in Hounsfield Units.
    """
    dicom_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".dcm")
    ]

    # Read and sort slices
    slices = []
    for file in dicom_files:
        ds = pydicom.dcmread(file)
        if hasattr(ds, 'InstanceNumber'):
            slices.append((ds.InstanceNumber, ds))
        else:
            print(f"Warning: Skipping {file} (no InstanceNumber)")

    # Sort slices by InstanceNumber
    slices.sort(key=lambda x: x[0])
    sorted_datasets = [s[1] for s in slices]

    # Stack the slices into a 3D array
    pixel_arrays = [ds.pixel_array.astype(np.int16) for ds in sorted_datasets]
    volume = np.stack(pixel_arrays, axis=0)

    # Apply rescaling
    rescale_slope = float(sorted_datasets[0].RescaleSlope)
    rescale_intercept = float(sorted_datasets[0].RescaleIntercept)

    volume_hu = rescale_slope * volume + rescale_intercept

    return volume_hu

# Load volume (already rescaled to Hounsfield Units)
ct_folder = "CT_data/CT9"
ct_volume = load_ct_series_with_rescale(ct_folder)

# ct_volume shape: (num_slices, height, width) → (Z, Y, X)
print("Volume shape (Z, Y, X):", ct_volume.shape)

# Get the central sagittal slice (middle along X-axis)
sagittal_index = ct_volume.shape[2] // 2  # Middle of width (X-axis)
sagittal_slice = ct_volume[:, :, sagittal_index]  # Shape: (Z, Y)


# Plot the sagittal slice
plt.figure(figsize=(6, 8))
plt.imshow(sagittal_slice, cmap='gray')
plt.title(f"Central Sagittal Slice (index={sagittal_index})")
plt.xlabel("Slice (Z-axis)")
plt.ylabel("Vertical (Y-axis)")
plt.colorbar(label="Hounsfield Units (HU)")
plt.axis('on')
plt.tight_layout()
plt.show()

# Extract the central Y-index
coronal_index = ct_volume.shape[1] // 2  # Middle along Y-axis (rows)
coronal_slice = ct_volume[:, coronal_index, :]  # Shape: (Z, X)

# Plot it
plt.figure(figsize=(6, 8))
plt.imshow(coronal_slice, cmap='gray')
plt.title(f"Central Coronal Slice (Y-index={coronal_index})")
plt.xlabel("Slice (Z-axis)")
plt.ylabel("Width (X-axis)")
plt.colorbar(label="Hounsfield Units (HU)")
plt.axis("on")
plt.tight_layout()
plt.show()

# Get the middle slice along the Z-axis
axial_index = ct_volume.shape[0] // 2  # Middle of slice stack
axial_slice = ct_volume[axial_index, :, :]

# Plot it
plt.figure(figsize=(6, 6))
plt.imshow(axial_slice, cmap='gray')
plt.title(f"Central Axial Slice (Z-index={axial_index})")
plt.xlabel("X (width)")
plt.ylabel("Y (height)")
plt.colorbar(label="Hounsfield Units (HU)")
plt.axis("on")
plt.tight_layout()
plt.show()