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
    dicom_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".dcm")
    ]

    slices = []
    for file in dicom_files:
        ds = pydicom.dcmread(file)
        if hasattr(ds, 'InstanceNumber'):
            slices.append((ds.InstanceNumber, ds))
        else:
            print(f"Warning: Skipping {file} (no InstanceNumber)")

    if not slices:
        raise ValueError(f"No valid DICOM files found in {folder_path}")

    slices.sort(key=lambda x: x[0])
    sorted_datasets = [s[1] for s in slices]

    pixel_arrays = [ds.pixel_array.astype(np.int16) for ds in sorted_datasets]
    volume = np.stack(pixel_arrays, axis=0)

    rescale_slope = float(sorted_datasets[0].RescaleSlope)
    rescale_intercept = float(sorted_datasets[0].RescaleIntercept)

    volume_hu = rescale_slope * volume + rescale_intercept

    return volume_hu


# 路径配置
root_folder = "CT_data"
save_folder = r"Metrics\data\LDCT"
os.makedirs(save_folder, exist_ok=True)

# 遍历 CT0 到 CT9
for i in range(10):
    patient_id = f"CT{i}"
    ct_folder = os.path.join(root_folder, patient_id)

    try:
        ct_volume = load_ct_series_with_rescale(ct_folder)

        # Sagittal slice
        sagittal_index = ct_volume.shape[2] // 2
        sagittal_slice = ct_volume[:, :, sagittal_index]
        plt.figure(figsize=(6, 8))
        plt.imshow(sagittal_slice, cmap='gray')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(save_folder, f"{patient_id}_sagittal.png"), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Coronal slice
        coronal_index = ct_volume.shape[1] // 2
        coronal_slice = ct_volume[:, coronal_index, :]
        plt.figure(figsize=(6, 8))
        plt.imshow(coronal_slice, cmap='gray')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(save_folder, f"{patient_id}_coronal.png"), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Axial slice
        axial_index = ct_volume.shape[0] // 2
        axial_slice = ct_volume[axial_index, :, :]
        plt.figure(figsize=(6, 6))
        plt.imshow(axial_slice, cmap='gray')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(save_folder, f"{patient_id}_axial.png"), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"✅ 已保存 {patient_id} 的三张图像")

    except Exception as e:
        print(f"❌ 处理 {patient_id} 时出错: {e}")
