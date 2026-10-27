# Using Custom Datasets in Flow Fluo

This guide explains how to configure `flow_fluo` to run on your own fluctuation-based fluorescence microscopy datasets.

## 1. Supported Data Formats

The code automatically handles different data structures based on the configuration. You can provide your dataset as:
- A `.zip` folder containing `N` individual `.pt` tensors (where `N` is the number of frames).
- A `.tif` or `.tiff` file containing a volume of shape `(N, W, H)`.
- A `.czi` file.

## 2. Creating your Dataset Configuration File (YAML)

To use your data, you need to create a `.yaml` configuration file inside `ddm4ip/configs/dataset/` (e.g., `ddm4ip/configs/dataset/my_custom_data.yaml`).

This file describes not only where the data is, but also the physical parameters of the acquisition (camera characteristics, PSF, under-sampling factor, etc.) that the forward double-Poisson model needs to work correctly.

### YAML Template

Here is a heavily commented template you can copy and adapt for your dataset:

```yaml
# Default settings to not add artificial noise/degradation on the fly
# (assumes your raw data is already noisy/blurred as acquired by the microscope)
defaults:
  - degradation: no_degradation
  - noise: no_noise       

# Data loader type:
# Use 'zip' if your data is a .zip of .pt files.
# Use 'volume' if your data is a .tif, .tiff, or .czi file.
name: volume 

# Paths to your data (can be the same file for both if you just want to reconstruct it)
train_path: datasets/my_custom_acquisition.tif
test_path: datasets/my_custom_acquisition.tif

# ---------------------------------------------------------
# Microscope & Physical Parameters
# ---------------------------------------------------------

# Number of input channels (usually 1 for grayscale microscopy)
in_channels: 1

# Super-resolution factor (e.g., 2 means the HR reconstruction will be 2x larger in H and W)
undersampling: 2

# Camera Conversion Factor (ADU per electron / gain)
M: 2963

# Camera Baseline/Offset (value to subtract from raw pixel intensities)
offset: 1726

# Point Spread Function (PSF) size in pixels
kwidth: 61

# PSF standard deviation (Gaussian blur sigma)
ksigma: 1.42

# ---------------------------------------------------------
# Generative Model / Loss parameters
# ---------------------------------------------------------

# Poisson rates parameters
alpha: 0.014
alpha2: 1
alpha3: 1

# Set to True if using the specific double-Poisson forward model variant
double_poisson: False

# Ensure non-negativity in intermediate steps
allow_negative_values: True
```

## 3. Running the Pipeline

Once you have created `my_custom_data.yaml`, you can create custom experiment files in `ddm4ip/configs/exp/` (e.g., `step1_my_custom_data.yaml` and `step2_my_custom_data.yaml`). 

Inside those experiment files, make sure to override the dataset:

```yaml
# Example snippet for step1_my_custom_data.yaml
defaults:
  - /models@models.flow: xxxs_flow_128
  - override /training: default
  - override /loss: flow_matching
  - override /dataset: my_custom_data # <--- Points to your new yaml!
  - override /generation: flow
```

Then run the steps normally:
```bash
python ddm4ip/main.py exp=step1_my_custom_data
python ddm4ip/main.py exp=step2_my_custom_data loss.lambda1=... loss.lambda2=...
```