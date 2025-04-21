# LDCT Denoising DL Model Benchmark

This directory contains our code and results for benchmarking 3 deep learning models for LDCT denoising.

## Dataset

The dataset we used to test the pre-trained models comes from [NLST | National Lung Screening Trial](https://www.cancerimagingarchive.net/collection/nlst/).

We randomly selected 10 patients from this dataset and randomly selected one CT sequence from each of the patients. We have 1440 slices of LDCT images in the chest window.

## Models for Benchmark

### CLIPDenoising

Access their GitHub Repo at [CLIPDenoising](https://github.com/alwaysuu/CLIPDenoising), and their paper at [Transfer CLIP for Generalizable Image Denoising](https://arxiv.org/html/2403.15132v1).

### CoreDiff

Access their GitHub Repo at [CoreDiff](https://github.com/qgao21/CoreDiff), and their paper at [CoreDiff: Contextual Error-Modulated Generalized Diffusion Model for Low-Dose CT Denoising and Generalization](https://arxiv.org/abs/2304.01814).

### CNN10

## Evaluation Metrics

### Residual $\sigma$
Residual σ measures how much variance the denoiser removes by computing the standard deviation of the residual image $x - y$, where $x$ is the original slice and $y$ is the denoised slice.  In practice we calculate  
$$\displaystyle σ_{\text{res}} = \sqrt{\frac{1}{N}\sum_{i=1}^N\bigl((x_i - y_i) - \overline{x-y}\bigr)^2}$$.  
On an 8‑bit CT window (0–255) values from about 0–10 HU mean almost no noise was removed, while 20–60 HU indicates substantial noise suppression.  Higher σ₍res₎ is better—provided edge preservation metrics remain within acceptable bounds.

### NRR  
The Noise‑Reduction Ratio (NRR) tells you the fraction of original variance eliminated by computing  
$$\displaystyle 1 - \frac{\sigma_y}{\sigma_x}$$,  
with $\sigma_x$ and $\sigma_y$ being the global standard deviations of the original and denoised slices.  An NRR of 0 means no change in noise, 1 means complete theoretical noise removal, and negative values mean the filter actually increased variance.  In most CT denoising workflows, values between 0.3 and 0.7 represent a good balance between noise suppression and feature retention.

### AAG ratio  
The AAG (Average Absolute Gradient) ratio compares edge strength before and after denoising by taking the mean Sobel gradient magnitude of the denoised image divided by that of the original.  Values close to 1.0 mean edges are preserved; values below 0.9 indicate blurring, and values above 1.1 suggest over‑sharpening or ringing artifacts.  Keeping the AAG ratio within 0.9–1.1 ensures that resolution is maintained without artificial edge exaggeration.

### L2 per pixel  
L2 per pixel is the root‑mean‑square difference between the original and denoised images, normalized by the number of pixels:  
$$\displaystyle \frac{\|x - y\|_2}{N}$$.  
This metric captures all changes—noise removal, blur, and artifacts.  Lower values are generally better, but if it approaches 0 it means almost no change occurred.  Comparing L2 per pixel to the original noise level helps determine whether the denoiser is making meaningful improvements.

### Gradient Cosine  
Gradient cosine measures structural fidelity by computing the cosine similarity of the Sobel gradient fields of $x$ and $y$:  
$$\displaystyle \frac{\langle\nabla x,\nabla y\rangle}{\|\nabla x\|\,\|\nabla y\|}$$.  
A value of 1 indicates perfect preservation of edge directions; values ≥ 0.95 are typically acceptable, 0.90–0.95 indicates minor edge shifts, and values below 0.90 suggest noticeable deformations.  High gradient‐cosine scores confirm that anatomical structures remain aligned after denoising.
