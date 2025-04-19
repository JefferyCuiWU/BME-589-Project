import os
import cv2
from joblib import load
import pandas as pd
from brisque import brisque
from niqe import niqe
from piqe import piqe
import numpy as np
# Image directory structure
base_dir = "data/LDCT"
views = ["axial", "coronal", "sagittal"]
patients = [f"CT{i}" for i in range(10)]

# Store evaluation results
all_scores = {}

# Load BRISQUE model
brisque_model = load("svr_brisque.joblib")

for patient in patients:
    all_scores[patient] = {}
    for view in views:
        image_name = f"{patient}_{view}.png"
        image_path = os.path.join(base_dir, patient, image_name)

        im = cv2.imread(image_path)
        if im is None:
            print(f"Unable to read image: {image_path}")
            continue

        #  PIQE
        piqe_score, _, _, _ = piqe(im)

        #  NIQE
        niqe_score = niqe(im)

        #  BRISQUE
        brisque_feature = brisque(im).reshape(1, -1)
        brisque_score = brisque_model.predict(brisque_feature)[0]


        all_scores[patient][view] = {
            "PIQE": piqe_score,
            "NIQE": niqe_score,
            "BRISQUE": brisque_score
        }

# output
for patient in patients:
    print(f"\nImage quality evaluation for {patient}:")
    for view in views:
        print(f"  {view.capitalize()} View:")
        print(f"    PIQE: {all_scores[patient][view]['PIQE']}")
        print(f"    NIQE: {all_scores[patient][view]['NIQE']}")
        print(f"    BRISQUE: {all_scores[patient][view]['BRISQUE']}")

df_scores = pd.DataFrame(all_scores)

# Define the output path for the Excel file
output_path = os.path.join(base_dir, "LDCT_image_quality_scores.xlsx")

# Save the results to an Excel file
df_scores.to_excel(output_path, index=False)

print(f"Image quality scores saved to {output_path}")
summary_scores = {}
for patient in patients:
    scores = {"PIQE": [], "NIQE": [], "BRISQUE": []}
    for view in views:
        if view in all_scores[patient]:
            scores["PIQE"].append(all_scores[patient][view]["PIQE"])
            scores["NIQE"].append(all_scores[patient][view]["NIQE"])
            scores["BRISQUE"].append(all_scores[patient][view]["BRISQUE"])

    summary_scores[patient] = {
        "PIQE_mean": np.mean(scores["PIQE"]),
        "PIQE_median": np.median(scores["PIQE"]),
        "NIQE_mean": np.mean(scores["NIQE"]),
        "NIQE_median": np.median(scores["NIQE"]),
        "BRISQUE_mean": np.mean(scores["BRISQUE"]),
        "BRISQUE_median": np.median(scores["BRISQUE"]),
    }

#
for patient in patients:
    print(f"\nSummary for {patient}:")
    print(
        f"  PIQE - Mean: {summary_scores[patient]['PIQE_mean']:.2f}, Median: {summary_scores[patient]['PIQE_median']:.2f}")
    print(
        f"  NIQE - Mean: {summary_scores[patient]['NIQE_mean']:.2f}, Median: {summary_scores[patient]['NIQE_median']:.2f}")
    print(
        f"  BRISQUE - Mean: {summary_scores[patient]['BRISQUE_mean']:.2f}, Median: {summary_scores[patient]['BRISQUE_median']:.2f}")

#
df_summary = pd.DataFrame.from_dict(summary_scores, orient="index")
df_summary.index.name = "Patient"
summary_output_path = os.path.join(base_dir, "LDCT_image_quality_summary.xlsx")
df_summary.to_excel(summary_output_path)

print(f"\nImage quality summary saved to {summary_output_path}")