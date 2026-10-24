from pathlib import Path
import deepinv
import scipy
import scipy.io
import torch
from deepinv.physics import BlurFFT

from ddm4ip.utils.torch_utils import pad_kernel


class GaussianNoise(deepinv.physics.GaussianNoise):
    """Wraps deepinv `GaussianNoise`"""
    def __getstate__(self):
        return fix_deepinv_state(self.__dict__)

    def __setstate__(self, d):
        self.__dict__ = d


def init_noise(noise_cfg):
    noise_kind = noise_cfg["kind"]
    if noise_kind == "gaussian":
        noise_model = GaussianNoise(sigma=noise_cfg["std"])
    if noise_kind == "poissongaussian":
        noise_model = GaussianNoise(sigma=noise_cfg["std"])
    elif noise_kind == "none":
        noise_model = None
    else:
        raise ValueError(f"noise '{noise_kind}' not recognized")
    return noise_model


def load_filter_from_file(path, kernel_size: int | None):
    path = Path(path)
    if path.suffix == ".mat":
        kernel = scipy.io.loadmat(path)["psf"]
        kernel = torch.from_numpy(kernel).float()
        if kernel.dim()!=4:
            kernel = kernel[None, None, ...]
    else:
        raise NotImplementedError("Only mat files are supported")
    if kernel_size is not None:
        kernel = pad_kernel(kernel, kernel_size)
    print('ottimale')
    return kernel

def instantiate_single_kernel_justpath(kernel_path):
        filter = load_filter_from_file(kernel_path)
        print('ottimo')
        return BlurFFT(
            img_size = (1, 1, 256, 256), #ADATTA!!!
            filter=filter,
        )


def instantiate_single_kernel(pert_cfg, noise_model):
    pert_kind = pert_cfg["kind"]
    if pert_kind == "gaussian_blur":
        from degradations.blur import Blur
        filter = deepinv.physics.blur.gaussian_blur(
            sigma=pert_cfg['kernel_std'],
            angle=pert_cfg.get('kernel_angle', 0),
        )
        return Blur(
            filter=filter,
            padding="replicate",
            noise_model=noise_model
        )
    elif pert_kind == 'gaussian_file':
        from degradations.blur import Blur
        filter = load_filter_from_file(pert_cfg["kernel_path"], kernel_size=pert_cfg.get("kernel_size", None))
        print('ottimo')
        return BlurFFT(
            img_size = (1, 1, 256, 256), #ADATTA!!!
            filter=filter,
        )

    elif pert_kind == "gaussian_downsampling":                  #forse può fare al caso tuo!!!! però devi aggiungere back ecc.... in più credo che lui prenda una foto e applichi rumore e stop. a te serve mult
                                                                 # che a partire da una foto si creino multipli esempi
        from degradations.downsampling import Downsampling
        filter = deepinv.physics.blur.gaussian_blur(
            sigma=pert_cfg['kernel_std'],
            angle=pert_cfg.get('kernel_angle', 0),
        )
        return Downsampling(
            img_size=None,  # only needed for restoration. TODO: We may want to allow specifying this.
            filter=filter,
            factor=pert_cfg["factor"],
            padding=pert_cfg["padding"],
            noise_model=noise_model
        )
    elif pert_kind == "file_downsampling":                     #o forse ancora meglio
        from degradations.downsampling import Downsampling
        filter = load_filter_from_file(pert_cfg["kernel_path"], kernel_size=pert_cfg.get("kernel_size", None))
        return Downsampling(
            img_size=None,
            filter=filter,
            factor=pert_cfg["factor"],
            padding=pert_cfg["padding"],
            noise_model=noise_model
        )
    elif pert_kind == "motion_blur":
        from degradations.blur import Blur, get_motion_blur_kernel
        filter = get_motion_blur_kernel(
            pert_cfg["kernel_size"],
            pert_cfg["intensity"],
            pert_cfg["rnd_seed"],
        )
        return Blur(
            filter=filter,
            padding="replicate",
            noise_model=noise_model
        )
    elif pert_kind == "per_pixel_blur":
        from degradations.varpsf import PerPixelInterpolatedBlur
        return PerPixelInterpolatedBlur(
            psf_path=pert_cfg["psf_path"],
            kernel_size=pert_cfg["kernel_size"],
            noise_model=noise_model,
        )
    elif pert_kind == "per_patch_blur":
        from degradations.varpsf import PerPatchInterpolatedBlur
        return PerPatchInterpolatedBlur(
            psf_path=pert_cfg["psf_path"],
            kernel_size=pert_cfg["kernel_size"],
            padding=pert_cfg["padding"],
            noise_model=noise_model,
        )
    elif pert_kind == "padding":
        from degradations.blur import Pad
        return Pad(
            size=pert_cfg["kernel_size"],
            padding=pert_cfg["padding"],
        )

    elif pert_kind == "none":
        return None
    else:
        raise ValueError(f"degradation '{pert_kind}' not recognized")


def init_perturbation(pert_cfg, noise_cfg):
    noise_model = init_noise(noise_cfg)
    degradation = instantiate_single_kernel(pert_cfg, noise_model)
    return degradation


def identity_fn(x, **kwargs):
    return x


def fix_deepinv_state(state):
    """Replace lambdas in the state-dict of deepinv objects with an identity function"""
    for k, v in state.items():
        if callable(v) and v.__name__ == "<lambda>":
            state[k] = identity_fn
            print(f"Replaced lambda at attribute {k} with identity function.")

    return state
