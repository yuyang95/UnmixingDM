## Introduction

This is the implementation of article: [UnmixDiff: Unmixing-based Diffusion Model for Hyperspectral Image Synthesis]

## Citation

Y. Yu, E. Pan, Y. Ma, X. Mei, Q. Chen and J. Ma, "UnmixDiff: Unmixing-based Diffusion Model for Hyperspectral Image Synthesis," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2024.3425517.

## Usage


### Data

The selected datasets are open-source, including Chikusei, Whu-OHS, and ARAD. Before training the network, please download the dataset (HSI) and crop the HSI into several MAT files. It is recommended that the spatial size should not be larger than 256 x 256. If you want to train the network using your own data, please modify the input according to the provided data.

### 1: Train the Unmixing network and infer the abundance maps by the trained Unmixing network.
Input: Oringinal HSIs.

Output: Inferred abundance maps.

For training the unmixing net, change the file path and run the following code.

python Unmixing.py train 


After training, run the following code to get the abundance maps. 

python Unmixing.py infer 

After that, we can obtain the inferred abundance in `./datasets/inferred_abu/`.

### 2: Train the Diffusion model and synthesize abundance maps by the trained Diffusion model.
Input: Inferred abundance maps by Step 1.

Output: Synthetic abundance maps.

For training the Diffusion model, please run:

`python Diffusion.py -p train -c config/Chikusei_256_DDPM.json`

For synthesizing abundance maps, please modify the 'resume_state' in the json file, and run:

`python Diffusion.py -p val -c config/Chikusei_256_DDPM.json`

After that, we can obtain the synthesized abundance in `./experiments/ddpm/\*/mat_results/`.

### 3: HSI synthesis.
Input: Synthetic abundance maps by Step 2.

Output: Synthetic HSIs.

Change the `train_path` (path of synthesized abundances) and the `model_name`(the trained model of the unmixing net)

Run the following code to obtain the synthetic HSIs:

`python Synthesis.py`

After that, we can obtain the synthesized HSIs in `./experiments/Synthesis/HSI/` and corresponding RGB images in `./experiments/Synthesis/RGB/`.







