# UnmixDiff: Unmixing-based Diffusion Model for Hyperspectral Image Synthesis



## Introduction

This is the source code for our paper: [UnmixDiff: Unmixing-based Diffusion Model for Hyperspectral Image Synthesis]
## Usage

### 1: Train the Unmixing network and infer the abundance maps by the trained Unmixing network.
Input: Oringinal HSIs.

Output: Inferred abundance maps.

For training the unmixing net, change the file path and run the following code.

python Unmixing.py train --dataset_name 'Chikusei' --n_blocks 3 --epochs 50 --batch_size 8 --gpus "1"


After training, run the following code to get the abundance maps. 

python Unmixing.py infer --dataset_name 'Chikusei' --n_blocks 3 --gpus "0"

After that, we can obtain the inferred abundance of RGB dataset in `./datasets/inferred_abu/`.

### 2: Train the Diffusion model and synthesize abundace maps by the trained Diffusion model.
Input: Inferred abundance maps by Step 1.

Output: Synthetic abundance maps.

For training the Abundance-based Diffusion, run the following code:

`python Diffusion.py -p train -c config/RS_256_abu_DDPM.json`

After training, modify the 'resume_state' in the `./config/*.json` file, and run:

`python Diffusion.py -p val -c config/RS_256_abu_DDPM.json`

After that, we can obtain the synthesized abundance in `./experiments/ddpm/\*/mat_results/`.

### 3: HSI synthesis.
Input: Synthetic abundance maps by Step 2.

Output: Synthetic HSIs.

Change the `train_path` (path of synthesized abundances) and the `model_name`(the trained model of the unmixing net)

Run the following code to obtain the synthetic HSIs:

`python Synthesis.py`

After that, we can obtain the synthesized HSIs in `./experiments/fusion/HSI/` and its corresponding false-color images in `./experiments/fusion/RGB/`.







