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
undersampling: 2    #Required

M: 2963 #Required: the maximum pixel intensity value present in your dataset.

# Camera Baseline/Offset/Pedestal. Default: 0 (value to subtract from raw pixel intensities)
offset: 0 #Optional

#---------
#PSF parameters: you should provide at least ksigma or px & FWHM or px & NA & wavelength to model PSF
#---------

# Point Spread Function (PSF) size in pixels: Required
kwidth: 61

# PSF standard deviation (Gaussian blur sigma)
ksigma: 1.42

#FWHM in nanometers
FWHM: 198

#Pixel size in nanometers
px: 100

#NA Numerical Aperture of the microscope
NA: 1.4

#Wavelength
wavelength: 500

# Poisson rates parameters: the physical model contain two Poisson distribution and can be written as
# y = alpha2* 1/alpha P(alpha (UH* 1/beta* P(beta x) + b )) + eps where UH undersampling+ convolution. alpha3 can be described as but was set to 1 in all our experiments;
# the inner Poisson is used only if you set double_poisson as True
 
alpha: 0.014      #Required
beta: 1           #Optional
alpha2: 1         #Optional

# Set to True if using the specific double-Poisson forward model variant
double_poisson: False #Optional

# Ensure non-negativity in intermediate steps. Default to False. In case of offset we suggest to set it to True
allow_negative_values: False   #Optional
```

## 3. Running the Pipeline 

Once you have created `my_custom_data.yaml`, you can create custom experiment files in `ddm4ip/configs/exp/` (e.g., `step1_my_custom_data.yaml` and `step2_my_custom_data.yaml`). 

You can look at `ddm4ip/configs/exp/step1_custom.yaml` and `ddm4ip/configs/exp/step2_custom.yaml` for reference for step1 and step2 config files, respectively.

Then run the steps normally:
```bash
python ddm4ip/main.py exp=step1_my_custom_data
python ddm4ip/main.py exp=step2_my_custom_data
```
