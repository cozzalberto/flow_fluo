import torch
import deepinv as dinv
import numpy as np
import torch.nn.functional as F
from torch import nn
from deepinv.optim.potential import Potential
import matplotlib.pyplot as plt
import scipy.io
from deepinv.physics import Denoising, GaussianNoise, PoissonNoise, PoissonGaussianNoise, Downsampling

class CustomReflectionPad2d(nn.Module):
    def __init__(self, pad_size):
        super().__init__()
        self.p = pad_size

    def forward(self, x):
        """
        Manually pads the tensor x by mirroring boundaries.
        x shape: (N, C, H, W)
        """
        p = self.p
        if p == 0:
            return x
            
        # Allocate the larger canvas
        N, C, H, W = x.shape
        x_padded = torch.zeros(N, C, H + 2*p, W + 2*p, dtype=x.dtype, device=x.device)
        
        # 1. Place the core original image in the center
        x_padded[..., p:-p, p:-p] = x
        
        # 2. Mirror Top and Bottom margins
        x_padded[..., :p, p:-p] = x[..., :p, :].flip(-2)
        x_padded[..., -p:, p:-p] = x[..., -p:, :].flip(-2)
        
        # 3. Mirror Left and Right margins (including padded top/bottom areas)
        x_padded[..., :, :p] = x_padded[..., :, p:2*p].flip(-1)
        x_padded[..., :, -p:] = x_padded[..., :, -2*p:-p].flip(-1)
        
        return x_padded

class Conv2dConstKernel(nn.Module):
    """
    A conv layer with constant gaussian kernel to modelize diffraction
    Default conv is computed with nn.functionnal.conv2d
    but may be replaced with othe
    """

    def __init__(self,
                 kwidth,
                 ksigma,
                 undersampling,
                 device,
                 conv=nn.functional.conv2d) :
        """
        Builder function for Conv2dConstKernel layer. It generates a
        constant kernel that will be used then for convolution. The
        kernel can be generated to compute undersampling at the same time
        with convolution. In this case the ouput is the convolved image
        on a coarser grid. Each pixel is the sum of several fine pixels
        corresponding to finer input grid.

        Parameters
        ----------
        kwidth : int (odd)
            The width of the convolution kernel (must be odd for padding
            reasons, the kernel is a square matrix).
        ksigma : float
            Standard error sigma for gaussian kernel
        undersampling : int
            Reduction ratio to apply for undersampling during convolution
            Remark : it is different from pytorch stride since it sums values
            of pixels in fine grid to get value for coarse grid. The stride
            parameter is different because the result is a selection of one
            pixel out of given number of pixel instead of the sum.
        dtype : type
            dtype of the kernel data (default: float)
        conv : function for (fft) conv with same signature like conv2d
            default nn.functionnal.conv2d but may be replaced with fft_conv
        """
        super().__init__()

        self.kwidth = kwidth
        self.undersampling = undersampling
        self.conv = conv

        #build gaussian kernel
        ax = torch.linspace(-kwidth//2, kwidth//2, kwidth, device = device)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(- (xx**2 + yy**2) / 2 / ksigma**2)
        kernel /= kernel.sum()

        #reshape it to a 4D tensor
        kernel = kernel.view(1,1,kwidth,kwidth)
        self.filter = kernel
        #convolve it with undersampling matrix of ones
        ones = torch.ones((1,1,undersampling, undersampling), device = device)
        kernel = nn.functional.conv2d(kernel, ones, padding=undersampling//2)

        #save it as a constant parameter
        self.kernel = nn.Parameter(kernel, requires_grad=False)

        #build a padding layer to keep dim
        #self.pad = nn.ReflectionPad2d(kwidth // 2)
        self.pad = CustomReflectionPad2d(kwidth // 2)

    def forward(self,x) :
        """
        Applies conv with constant kernel and stride to input x

        Parameters
        ----------
        x : tensor of shape (n_images, n_channels, height, width)

        Returns
        -------
        y : tensor of same shape like x
        """
        return self.conv(self.pad(x), self.kernel, \
                        stride=self.undersampling)

    def adjoint(self, y, target_shape=(32, 1, 256*6, 256*6)):
        """
        Adjoint operator matching forward(x) = conv2d(pad(x), kernel, stride=L)
        y: Low-res tensor input (N, C, H_out, W_out)
        target_shape: Original shape of x before padding (N, C, H_in, W_in)
        """
        # 1. Target shape dimensions of the padded space
        H_in, W_in = target_shape[-2], target_shape[-1]
        pad_size = self.kwidth // 2

        H_padded_target = H_in + 2 * pad_size
        W_padded_target = W_in + 2 * pad_size

        # 2. Get the current size of the modified kernel
        # (It's no longer kwidth due to the conv2d step with 'ones')
        kH_new, kW_new = self.kernel.shape[-2], self.kernel.shape[-1]

        # 3. Calculate dynamic output_padding for conv_transpose2d
        # This guarantees that the intermediate tensor size exactly matches H_padded_target
        out_pad_h = H_padded_target - ((y.shape[-2] - 1) * self.undersampling + kH_new)
        out_pad_w = W_padded_target - ((y.shape[-1] - 1) * self.undersampling + kW_new)

        # 4. Step A: Adjoint of strided convolution -> Back to padded space
        grad_padded = F.conv_transpose2d(
            y,
            self.kernel,
            stride=self.undersampling,
            padding=0,  # Keep 0 here to fully expand, we map to target size via out_pad
            output_padding=(out_pad_h, out_pad_w)
        )

        # 5. Step B: Adjoint of Reflection Padding -> Back to original space
        x_adjoint = self.reflection_pad_adjoint(grad_padded)

        return x_adjoint

    def reflection_pad_adjoint(self, grad_padded):
        """
        Adjoint of ReflectionPad2d(kwidth // 2).
        Folds the margins back onto the core tensor area.
        """
        p = self.kwidth // 2
        # Extract the core high-resolution region
        grad_x = grad_padded[..., p:-p, p:-p].clone()

        # Fold Top and Bottom margins back
        grad_x[..., :p, :] += grad_padded[..., :p, p:-p].flip(-2)
        grad_x[..., -p:, :] += grad_padded[..., -p:, p:-p].flip(-2)

        # Fold Left and Right margins back
        grad_x[..., :, :p] += grad_padded[..., p:-p, :p].flip(-1)
        grad_x[..., :, -p:] += grad_padded[..., p:-p, -p:].flip(-1)

        # Fold the 4 corners (double reflections)
        grad_x[..., :p, :p] += grad_padded[..., :p, :p].flip(-1).flip(-2)
        grad_x[..., :p, -p:] += grad_padded[..., :p, -p:].flip(-1).flip(-2)
        grad_x[..., -p:, :p] += grad_padded[..., -p:, :p].flip(-1).flip(-2)
        grad_x[..., -p:, -p:] += grad_padded[..., -p:, -p:].flip(-1).flip(-2)

        return grad_x

class Corruption_train(dinv.physics.Physics):
    def __init__(self, noise_level=5,std = 1.0, do_blur=True, device = "cpu"):
        super().__init__()
        self.device = device
        self.std = std
        self.conv = Conv2dConstKernel(61, 7.93, 6, self.device)
        self.noise_level = noise_level
        self.L = 6
        self.do_blur = do_blur
        self.kernel = self.conv.filter
        #self.M = 489.2048344026573
        #self.M = 80 
        #self.M = 5378.37841796875
        self.M =  42
        self.relu = nn.ReLU()


    def forward(self, x, b):
        if self.do_blur:
            out = self.conv(x)
        else:
            out = x
        out = out + b
        #out = PoissonGaussianNoise(gain = 1., sigma = self.noise_level)(out)
        out_safe = out.clamp(min=1e-6) 
        #out = out + torch.sqrt(out/self.std)*torch.randn_like(out) + self.noise_level/self.std *torch.randn_like(out)
        eps_noise = torch.randn_like(out)
        out_pre_relu = out + torch.sqrt(out_safe/self.M)*eps_noise + self.noise_level/self.M *torch.randn_like(out)
        out = self.relu(out_pre_relu)
        # out = out + torch.sqrt(out)*torch.randn_like(out) + self.noise_level *torch.randn_like(out)
        return out, eps_noise, out_pre_relu

    def A_t(self, y_obs):
        y_up = torch.nn.functional.interpolate(y_obs, scale_factor=self.L, mode="nearest")
        x  = torch.zeros((y_obs.shape[0], y_obs.shape[1], y_obs.shape[2]*self.L, y_obs.shape[3]*self.L), device=y_obs.device, dtype=y_obs.dtype)
        x[:,:, :: self.L, :: self.L] = y_obs  # upsample
        #print("finito self")
        if self.kernel is not None:
            x0 = F.conv2d(
                x, self.kernel, padding= "same" #se fau nell'altro modo x al posto di y_up
            )
        #np.save("x_wronginitial.npy",x0.cpu().numpy())
        return x0


from scipy.fft import next_fast_len


class Conv2dConstKernel_fft_fast(nn.Module):
    """
    Same as Conv2dConstKernel_fft, but with:
      1. Cached kernel FFT (computed once per shape/device, not every forward pass)
      2. FFT sizes rounded to 'fast lengths' for faster torch.fft performance
    """
    def __init__(self, kwidth, ksigma, undersampling, circular=False, device= "cpu"):
        super().__init__()
        self.kwidth = kwidth
        self.undersampling = undersampling
        self.circular = circular
        self.device = device

        ax = torch.arange(kwidth, device=device, dtype=torch.float32) - (kwidth / 2.0 - 0.5)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / 2 / ksigma**2)
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, kwidth, kwidth)
        
        self.filter = kernel
        ones = torch.ones((1, 1, undersampling, undersampling), device = device)
        kernel = nn.functional.conv2d(kernel, ones, padding=undersampling // 2)

        self.kernel = nn.Parameter(kernel, requires_grad=False)
        self.pad = nn.ReflectionPad2d(kwidth // 2)

        self._kernel_fft_cache = {}


        max_idx = torch.argmax(self.kernel.squeeze())
        self.start_h = int(max_idx // self.kernel.shape[-1])
        self.start_w = int(max_idx % self.kernel.shape[-1])

    def _get_kernel_fft(self, H_fft, W_fft):
        key = (H_fft, W_fft, self.kernel.device, self.kernel.dtype)
        if key not in self._kernel_fft_cache:
            self._kernel_fft_cache[key] = torch.fft.rfft2(
                self.kernel, s=(H_fft, W_fft)
            )
        return self._kernel_fft_cache[key]

    @staticmethod
    def _fast_shape(H, W, kH, kW):
        H_fft = next_fast_len(H + kH - 1)
        W_fft = next_fast_len(W + kW - 1)
        return H_fft, W_fft
    
    def fft_conv2d(self, x):
        H, W = x.shape[-2], x.shape[-1]
        kH, kW = self.kernel.shape[-2], self.kernel.shape[-1]

        # Calcolo del vero asse ottico: 30 + 3 = 33
        true_shift_h = (self.kwidth - 1) // 2 + self.undersampling // 2
        true_shift_w = (self.kwidth - 1) // 2 + self.undersampling // 2

        if self.circular:
            X = torch.fft.rfft2(x, s=(H, W))
            K = self._get_kernel_fft(H, W)
            y_full = torch.fft.irfft2(X * K, s=(H, W))

            # Applichiamo il vero shift
            shift_h = -true_shift_h
            shift_w = -true_shift_w
            y = torch.roll(y_full, shifts=(shift_h, shift_w), dims=(-2, -1))
        else:
            H_fft, W_fft = self._fast_shape(H, W, kH, kW)
            X = torch.fft.rfft2(x, s=(H_fft, W_fft))
            K = self._get_kernel_fft(H_fft, W_fft)
            y_full = torch.fft.irfft2(X * K, s=(H_fft, W_fft))

            # Usiamo il vero asse per il crop
            start_h = true_shift_h
            start_w = true_shift_w
            y = y_full[..., start_h : start_h + H, start_w : start_w + W]

        if self.undersampling > 1:
            y = y[:, :, :: self.undersampling, :: self.undersampling]

        return y
    

    def adjoint(self, y):
        N, C = y.shape[0], y.shape[1]
        H = y.shape[2] * self.undersampling
        W = y.shape[3] * self.undersampling
        kH, kW = self.kernel.shape[-2], self.kernel.shape[-1]

        true_shift_h = (self.kwidth - 1) // 2 + self.undersampling // 2
        true_shift_w = (self.kwidth - 1) // 2 + self.undersampling // 2

        if self.undersampling > 1:
            x_up = torch.zeros(N, C, H, W, device=y.device, dtype=y.dtype)
            x_up[:, :, :: self.undersampling, :: self.undersampling] = y
            y = x_up

        if self.circular:
            # L'aggiunto ribalta il segno: shift positivo
            shift_h = true_shift_h
            shift_w = true_shift_w
            y_unrolled = torch.roll(y, shifts=(shift_h, shift_w), dims=(-2, -1))

            Y = torch.fft.rfft2(y_unrolled, s=(H, W))
            K = self._get_kernel_fft(H, W)
            X_hat = Y * torch.conj(K)
            x_adjoint = torch.fft.irfft2(X_hat, s=(H, W))
            return x_adjoint
        else:
            H_fft, W_fft = self._fast_shape(H, W, kH, kW)
            start_h = true_shift_h
            start_w = true_shift_w

            y_padded = torch.zeros(N, C, H_fft, W_fft, device=y.device, dtype=y.dtype)
            y_padded[..., start_h : start_h + H, start_w : start_w + W] = y

            Y = torch.fft.rfft2(y_padded, s=(H_fft, W_fft))
            K = self._get_kernel_fft(H_fft, W_fft)
            x_full = torch.fft.irfft2(Y * K.conj(), s=(H_fft, W_fft))

            return x_full[..., :H, :W]


    def forward(self, x):
        return self.fft_conv2d(x)


class Corruption_train_fft(dinv.physics.Physics):
    def __init__(self, kwidth = 61, ksigma = 7.93, alpha = 12, alpha3=1, alpha2 = 1, undersampling = 6, noise_level=5, M = 27.72173500061035,  do_blur=True, double_poisson = False, circular=False, allow_negative_values = True, device= "cpu"):
        super().__init__()
        self.device = device
        self.kwidth = kwidth
        self.ksigma = ksigma
        self.undersampling = undersampling
        self.circular = circular
        self.conv = Conv2dConstKernel_fft_fast(kwidth= self.kwidth, ksigma=self.ksigma, undersampling=self.undersampling, circular=self.circular, device = self.device)
        self.noise_level = noise_level
        self.L = 6
        self.do_blur = do_blur
        self.relu = nn.ReLU()
        self.M = M
        self.alpha = alpha
        self.alpha3= alpha3
        self.alpha2 = alpha2
        self.allow_negative_values = allow_negative_values
        self.double_poisson = double_poisson

    def forward(self, x, b):

        if not self.double_poisson:
            if self.do_blur:
                out = self.conv(x)
            else: 
                out = x
        else:
            x_variance_base = torch.clamp(self.alpha2 * x.detach(), min=1e-6)
            
            # 2. Il gradiente fluirà SOLO attraverso il primo termine (self.alpha2 * x)
            out = self.alpha2 * x + torch.sqrt(x_variance_base/self.M) * torch.randn_like(x)
            if self.do_blur:
                out = self.conv(out)  # <-- Usa 'out', non 'x'
        out = out + b  
        #out = PoissonGaussianNoise(gain = 1., sigma = self.noise_level)(out)  
        eps_noise = torch.randn_like(out)
        out_safe = out.clamp(min=1e-6)
        out_pre_relu = self.alpha3*out + self.alpha3*torch.sqrt(out_safe/(self.M*self.alpha))*eps_noise + self.noise_level*torch.randn_like(out)/self.M
        if not self.allow_negative_values:
            out = self.relu(out_pre_relu)
        else:
            out = out_pre_relu
        return out, eps_noise, out_pre_relu
    
    def A_t(self, y_obs):
        x = self.conv.adjoint(y_obs)
        return x

class Corruption_train_fft_ext(Corruption_train_fft):
    """
    Come Corruption_train_fft, ma la convoluzione circolare è applicata
    su un dominio esteso di p pixel HR per lato (p multiplo di L, p >= kwidth//2 + 1).
    forward(x, b): x (N,C,H,W) HR -> y (N,C,H/L,W/L), come prima.
    """
    def __init__(self, p=36, **kwargs):
        kwargs["circular"] = True              # il dominio esteso ha senso solo col circolare
        super().__init__(**kwargs)
        assert p % self.L == 0, "p deve essere multiplo di L"
        assert p >= self.kwidth // 2 + 1, "p troppo piccolo per il kernel"
        self.p = p
        self.q = p // self.L

    # --- operatori sul dominio esteso ---
    def A_ext(self, x_ext):                    # (N,C,H+2p,W+2p) -> (N,C,H/L,W/L)
        y = self.conv(x_ext)                   # conv circolare + sottocampionamento
        q = self.q
        return y[..., q:-q, q:-q]

    def A_ext_t(self, y):                      # (N,C,H/L,W/L) -> (N,C,H+2p,W+2p)
        q = self.q
        return self.conv.adjoint(F.pad(y, (q, q, q, q)))

    # --- interfaccia originale ---
    def forward(self, x, b):
        p = self.p
        if self.do_blur:
            x_ext = F.pad(x, (p, p, p, p), mode="reflect")   # richiede p < H, W
            out = self.A_ext(x_ext)
        else:
            out = x
        out = out + b

        eps_noise = torch.randn_like(out)
        out_safe = out.clamp(min=1e-6)
        out_pre_relu = self.alpha3*out + self.alpha3*torch.sqrt(out_safe/(self.M*self.alpha))*eps_noise + self.noise_level*torch.randn_like(out)/self.M
        if not self.allow_negative_values:
            out = self.relu(out_pre_relu)
        else:
            out = out_pre_relu
        return out, eps_noise, out_pre_relu

    def A_t(self, y_obs):
        # aggiunto di A_ext, poi crop centrale: restituisce (N,C,H,W)
        p = self.p
        return self.A_ext_t(y_obs)[..., p:-p, p:-p]

class PoissonProcess(nn.Module):
    def __init__(self,alpha):
        super().__init__()
        self.alpha = alpha
    def forward(self,x):
        return torch.poisson(self.alpha * x)

class GaussianNoise(nn.Module) :
    def __init__(self,sigma) :
        super().__init__()
        self.sigma = sigma
    def forward(self,x):
        return x + torch.normal(0, self.sigma,
                                dtype=x.dtype,
                                size=x.shape,
                                device=x.device)


