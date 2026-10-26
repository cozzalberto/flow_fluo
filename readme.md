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

 1. Download the corrupt.zip dataset from the [drive folder](https://drive.google.com/drive/folders/1IoZNxYp4JLzh4YlPC4hpRju-DMqQYLH-?usp=sharing) and modify the key 'train_path' inside ddm4ip/configs/dataset/argolight_cropped.yaml according to the dataset path. You do not need to unzip the file. 
It's available also a rectangular version of the dataset with image sizes 64 x 256 instead of 256 x 256. The results on the paper are obtained with the 'square' dataset, but using this would provide similar results with less computational time.

 2. Reproduce Step1:
      ```console
     python ddm4ip/main.py exp=step1_argolight_cropped
     ```
     If you wanted to change hyper-parameters, you can modify the config file ddm4ip/configs/exp/step1_argolight_cropped.yaml (or step1_argolight_rect.yaml if you want to use the rectangular dataset)
 3. Reproduce Step 2:
     ```console
     python ddm4ip/main.py exp=step2_argolight_cropped loss.lambda1=0.05 loss.lambda2=40 training.max_step=512000
     ```
     If you wanted to change hyper-parameters, you can modify the config file ddm4ip/configs/exp/step2_argolight_cropped.yaml (or step2_argolight_rect.yaml if you want to use the rectangular dataset)

