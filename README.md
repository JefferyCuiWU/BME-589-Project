# LDCT Denoising DL Model Benchmark

This directory contains our code and results for benchmarking 3 deep learning models for LDCT denoising.

## Dataset

The dataset we used to test the pre-trained models comes from [NLST | National Lung Screening Trial](https://www.cancerimagingarchive.net/collection/nlst/).

We randomly selected 10 patients from this dataset and randomly selected one CT sequence from each of the patients. We have 1440 slices of LDCT images in the chest window.

## Models for Benchmark

### CLIPDenoising

Access their GitHub Repo at [CLIPDenoising](https://github.com/alwaysuu/CLIPDenoising), and their paper at [Transfer CLIP for Generalizable Image Denoising](https://arxiv.org/html/2403.15132v1).

First, make sure that you get their pre-trained model and put it under experiments/pretrained_models. Then, download [ResNet50](https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt), and place it under model_data/clip.

Then, put [preprocessing.py](CLIPDenoising/preprocessing.py) and [run_ldct_inference.py](CLIPDenoising/run_ldct_inference.py) in the root directory of CLIPDenoising. Place [CLIPDenoising_LDCT_Inference.yml](CLIPDenoising/CLIPDenoising_LDCT_Inference.yml) under Denoising/Options. 

To run the model, first run [preprocessing.py](CLIPDenoising/preprocessing.py), then run [run_ldct_inference.py](CLIPDenoising/run_ldct_inference.py)

### CoreDiff

Access their GitHub Repo at [CoreDiff](https://github.com/qgao21/CoreDiff), and their paper at [CoreDiff: Contextual Error-Modulated Generalized Diffusion Model for Low-Dose CT Denoising and Generalization](https://arxiv.org/abs/2304.01814).

First, make sure that you get their pre-trained model and put it under output\corediff_test\save_models. 

Then, put [preprocessing.py](CoreDiff/preprocessing.py) in the root directory of CoreDiff. Replace models/corediff/corediff.py with [corediff.py](CoreDiff/corediff.py), and replace utils/dataset.py with [dataset.py](CoreDiff/dataset.py).

To run the model, first run [preprocessing.py](CoreDiff/preprocessing.py), then at the root directory, enter the following command

`python main.py --model_name corediff --run_name test --batch_size 4 --max_iter 150000 --test_dataset custom --test_batch_size 1 --dose 5 --mode test --test_iter 150000`

### CNN10
CNN10 is a convolutional neural network algorithm for denoising low-dose CT scan images. The pretrained version of this model provided by [ldct-benchmark](https://github.com/eeulig/ldct-benchmark) and its [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5330597/)
To run this algorithm, [run pre_trained_model.ipynb] in [CNN10] folder. Before runing the code, make sure that the dataset and [preprocesssing.py] located in your google drive.
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

### Usage Information

In the [Metrics](Metrics) folder, we provide two scripts for metrics evaluation. They only differ from the input type. If you are evaluating results of models in HU unit and stored as 3D volumes, please use [metrics_HU_3D.py](Metrics/metrics_HU_3D.py). If you are evaluating results in (0,1) and stored as slices, please use [metrics_01_slice.py](Metrics/metrics_01_slice.py). The calculations for the metrics are the same.
We also provide a script for plotting. We calculate the mean scores of each patient from CT0 to CT10 on each metric and further calculate the mean to compare the performance of each model on different metrics.

We have saved the results of metrics evaluation in [Results](BME-589-Project/Metrics/Results/).
