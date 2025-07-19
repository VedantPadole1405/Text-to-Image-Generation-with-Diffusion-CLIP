////// Note //////////
GradeScope 8.2: Our evaluation is based on prompts given by the users so there are no real images corresponding to the prompts.

# Vedant Padole, Madhumita Katam, Kavish Patel
# Dataset Setup for DiffusionCLIP

This repository provides a dataset preparation utility for training and evaluating models such as DiffusionCLIP. It supports downloading, extracting, and reformatting **ImageNet-128** and **Tiny-ImageNet** datasets for machine learning workflows. It also includes a Jupyter Notebook for DiffusionCLIP-based image manipulation and translation tasks.

## 📁 Project Structure

- `dataset_setup.py` – CLI tool to download and preprocess datasets.
- `DiffusionCLIP.ipynb` – Jupyter notebook to perform CLIP-guided diffusion image generation.
- `data/` – Output directory for preprocessed datasets.

---

## ⚙️ Dataset Preparation Utility

### Supported Datasets
- **ImageNet-128** (≈13 GB): Downloaded via HuggingFace.
- **Tiny-ImageNet** (≈250 MB): Fetched and extracted from Stanford CS231n.

### Usage

#### 1. Download ImageNet-128
```bash
python dataset_setup.py --dataset imagenet128 --out ./imagenet128
```

#### 2. Download Tiny-ImageNet
```bash
python dataset_setup.py --dataset tiny-imagenet --out ./tiny-imagenet
```

#### 3. Re-split Existing Dataset
Reorganize any ImageFolder-format dataset into train/val splits:
```bash
python dataset_setup.py --split_only --src ./tiny-imagenet/train --out ./tiny-imagenet-split --val_pct 10
```

---

## 📓 DiffusionCLIP Notebook

The `DiffusionCLIP.ipynb` notebook demonstrates:
- CLIP-guided image generation using diffusion models.
- Style and attribute transfer between unseen image domains.
- Evaluation of manipulation quality using CLIP scores.

Ensure you have preprocessed data from `dataset_setup.py` before running the notebook.

---

## 🔧 Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Required packages include:
- `datasets`
- `tqdm`
- `Pillow`
- `transformers`
- `torch`
- `diffusers`
- `CLIP` model libraries

For `ImageNet-128`, a HuggingFace token may be required for streaming.

---

## 📎 Notes

- The dataset preparation script is modular and can be extended for other image classification datasets.
- Ensure sufficient disk space (13+ GB for ImageNet-128).
- Internet connection is required for streaming or downloading datasets.

---


/////////////////// File Name :- CLIPGuidedDiffusion.ipynb ///////////////////////////////

Demonstrates: Integration of a basic Diffusion Model with CLIP for text-driven image generation
How to run:
Download the 'CLIPGuidedDiffusion.ipynb' and open it in Google colab. 
Set the GPU as T4 and connect to the RAM
Run all the cells to get the required outputs. The explanation of the results are explicitly mentioned in the pdf submitted in gradescope.(
(Due to computational constraints, we decided to use a pretrained model. The model that we took reference from do not account for CLIP loss
We have added the clip loss into the training function and ran on the promts that are specified by user)


/////////////////// File Name :- VQGAN+CLIP (with Pooling). ipynb /////////////////

This notebook implements a modified VQGAN+CLIP pipeline that incorporates average and max pooling into the MakeCutouts class, enhancing text-image alignment for better generation quality.

How to run:
Option 1: Google Colab (Recommended)
Easiest and fastest way to get started.
Automatically handles most dependencies and provides free GPU.

Option 2: Local Setup
Clone the required repositories:
taming-transformers
CLIP
Ensure the following files (included in the .zip) are in the same folder as the notebook:
vqgan_imagenet_f16_16384.ckpt
vqgan_imagenet_f16_16384.yaml

Common Errors & Fixes:
1) ModuleNotFoundError: No module named 'torch._six'
Fix: edit the file: /content/taming-transformers/taming/data/utils.py find the line: from torch._six import string_classes and replace it with string_classes = (str, bytes). 
2) If you encounter CUDA out-of-memory errors, reduce the number of cutouts or image resolution.

The notebook modifies the default MakeCutouts behavior to use average and max pooling to potentially improve semantic coherence between text and image.


/////////////////// File Name :- DCGAN+CLIP. ipynb ///////////////////////////////

Demonstrates: Integration of a basic DCGAN architecture with CLIP for text-driven image generation
Objective: Investigate how traditional GANs (like DCGAN) can be steered using CLIP embeddings to produce semantically relevant visuals.
How it works:
A latent vector is passed through a DCGAN generator to produce images.
The image is then evaluated using CLIP to compare its embedding against the embedding of a given text prompt.
The latent vector is optimized to reduce the distance between these embeddings, effectively aligning the generated image with the text.

/////////////////// File Name :- Untitled.ipynb ///////////////////////////////////
Goal: Functions as a sandbox/testing notebook for DCGAN+CLIP integration
What it includes:
Code to visualize DCGAN outputs.
CLIP-based latent optimization logic similar to DCGAN+CLIP.ipynb.
Tests for different prompt embeddings and image transformations.
Use: Helpful for understanding individual components before running the complete DCGAN+CLIP pipeline.

We use Spectral Normalization in the generator network (ConditionalGenerator) to stabilize GAN training by constraining the Lipschitz constant of each layer.
✅ Helps prevent exploding gradients.
✅ Encourages smoother updates during optimization.
✅ Especially beneficial when combined with CLIP-based feedback, which can be unstable.
its implemented like:
from torch.nn.utils import spectral_norm
self.model = nn.Sequential(
    spectral_norm(nn.ConvTranspose2d(...)),
    ...
)
