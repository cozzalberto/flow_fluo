# Flow Fluo: Fluctuation-based Fluorescence Super-Resolution
This repository implement the code for the article: [**Flow Matching for Fluctuation-Based Fluorescence Super-Resolution**](https://arxiv.org/abs/2610.37802).

## Abstract

Fluctuation-based super-resolution techniques are an appealing alternative to single-molecule and structured-illumination microscopy, as they adapt to standard acquisition settings and exploit the independent stochastic fluctuations of fluorophores across a sequence of frames. However, most existing approaches lack an explicit model of the acquisition process and provide only relative intensity estimates, limiting quantitative analyses. In this work, we combine model-based and generative approaches to exploit the full stochastic information in the sequence, proposing a reconstruction framework built on flow matching and a double-Poisson forward model accounting for blinking, blur, under-sampling and noise.
The method alternates two steps: a flow matching model first learns the distribution of the acquired data; then, the sample and the background are jointly estimated by minimising a Kullback-Leibler divergence between observed and simulated distributions via regularised proximal gradient steps. On simulated and real Argolight data, the method resolves lines separated by 120 nm, outperforming SACD and eSRRF.

## Setup & Installation
 1. Clone the repository:
```bash
git clone https://github.com/cozzalberto/flow_fluo
```
 2. Install the requirements: `pip install -r requirements.txt`
 
## Reproduce Simulate Argolight Experiment


### 1. Prepare the Dataset
Download the corrupt.zip dataset from the [drive folder](https://drive.google.com/drive/folders/1IoZNxYp4JLzh4YlPC4hpRju-DMqQYLH-?usp=sharing)
* Update the `train_path` key inside `ddm4ip/configs/dataset/argolight_cropped.yaml` to point to the location of the downloaded dataset. 
* **Note:** You do not need to unzip the file.

> **Tip:** A rectangular version `corrupt_rect.zip` of the dataset is also available with image sizes 64 x 256 (instead of 256 x 256). The results in the paper were obtained with the 'square' dataset, but    using the rectangular one will provide similar results in less computational time.

### 2. Reproduce Step1 
```console
python ddm4ip/main.py exp=step1_argolight_cropped
```
*To change hyper-parameters, you can modify the config file `ddm4ip/configs/exp/step1_argolight_cropped.yaml` (or `step1_argolight_rect.yaml` if you are using the rectangular dataset).*
### 3. Reproduce Step 2:
```console
python ddm4ip/main.py exp=step2_argolight_cropped loss.lambda1=0.05 loss.lambda2=40 training.max_step=512000
```
*To change hyper-parameters, you can modify the config file `ddm4ip/configs/exp/step2_argolight_cropped.yaml` (or `step2_argolight_rect.yaml` if you are using the rectangular dataset).*

## Use your own dataset
You can use your own dataset with your own dataset. 
The code supports:
 1. A .zip folder containing N `.pt` tensors where N is the number of your samples.
 2. A .tif or .tiff file with shape (N,W,H)
 3. A .czi file.

**For detailed instructions on how to configure the YAML files for your custom data, please refer to our guide: [CUSTOM_DATASET.md](CUSTOM_DATASET.md).**
