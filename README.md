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

### NRR

### AAG ratio

### L2 per pixel

### Gradient Cosine
