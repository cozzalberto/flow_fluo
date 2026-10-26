## Abstract

Fluctuation-based super-resolution techniques are an appealing alternative to single-molecule and structured-illumination microscopy, as they adapt to standard acquisition settings and exploit the independent stochastic fluctuations of fluorophores across a sequence of frames. However, most existing approaches lack an explicit model of the acquisition process and provide only relative intensity estimates, limiting quantitative analyses. In this work, we combine model-based and generative approaches to exploit the full stochastic information in the sequence, proposing a reconstruction framework built on flow matching and a double-Poisson forward model accounting for blinking, blur, under-sampling and noise.
The method alternates two steps: a flow matching model first learns the distribution of the acquired data; then, the sample and the background are jointly estimated by minimising a Kullback-Leibler divergence between observed and simulated distributions via regularised proximal gradient steps. On simulated and real Argolight data, the method resolves lines separated by 120 nm, outperforming SACD and eSRRF.

## Setup & Installation
 1. Clone the repository:
```bash
git clone https://github.com/cozzalberto/flow_fluo
```
 2. Install the requirements: `pip install -r requirements.txt`

## Step 1
 ```console
     python ddm4ip/main.py exp=step1_argolight_cropped
 ```

## Step 2

   ```console
     python ddm4ip/main.py exp=step2_argolight_cropped exp_name="step2_lambda1_${l1}_lambda2_${l2}"   loss.lambda1=$l1 loss.lambda2=$l2
   ```

