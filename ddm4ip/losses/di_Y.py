from typing import Any, Mapping
import warnings

import torch
import torch.nn as nn

from ddm4ip.data.base import Batch, Datasplit
from ddm4ip.losses.base import AbstractLoss, OptState, OptimizationState
from ddm4ip.losses.flow_matching import FlowMatchingLoss
from ddm4ip.losses.utils import create_optimizer
from ddm4ip.trainers.base import BaseTrainer
from ddm4ip.utils import distributed
from ddm4ip.utils.torch_utils import center, equate_kernel_shapes, normalize
from ddm4ip.degradations.degradation import load_filter_from_file

def gaussian_reg(kernel):
    if kernel.dim() == 5:
        # kernel.shape: B, H*W, kc, kh, kw
        kernel = kernel.reshape(-1, *kernel.shape[2:])
    B, C, H, W = kernel.shape

    #  Compute center of gravity (x_c, y_c)
    x_coords = torch.linspace(-1, 1, W, device=kernel.device).view(1, 1, 1, W)
    y_coords = torch.linspace(-1, 1, H, device=kernel.device).view(1, 1, H, 1)

    x_c = (kernel * x_coords).sum(dim=(-2, -1), keepdim=True)  # [-1, 1]
    y_c = (kernel * y_coords).sum(dim=(-2, -1), keepdim=True)  # [-1, 1]

    # Compute spread loss (variance around center of gravity, normalized)
    spread_loss = ((x_coords - x_c) ** 2 + (y_coords - y_c) ** 2) * kernel
    return spread_loss.mean()

def gradient_reg(background):
    # background: C, H, W
    grad_x = background[ :,:, :, 1:] - background[ :,:, :, :-1]
    grad_y = background[ :,:, 1:, :] - background[ :,:, :-1, :]
    
    return (grad_x**2).mean() + (grad_y**2).mean()

def TV_reg(x, eps=1e-6):
    dy = x.roll(-1, -2) - x
    dx = x.roll(-1, -1) - x
    return torch.sqrt(dy**2 + dx**2 + eps).mean()
    

def Hessian_reg(x :torch.Tensor):
    Ixx = x[..., 2:, 1:-1] - 2*x[..., 1:-1, 1:-1] + x[..., :-2, 1:-1]
    Iyy = x[..., 1:-1, 2:] - 2*x[..., 1:-1, 1:-1] + x[..., 1:-1, :-2]
    Ixy = (x[..., 2:, 2:] - x[..., 2:, :-2] - x[..., :-2, 2:] + x[..., :-2, :-2]) / 4
    # penalizza blob (entrambi autovalori grandi) ma non ridge (un solo autovalore grande)
    return (Ixx*Iyy - Ixy**2).abs().mean()

def sparse_reg(kernel: torch.Tensor):
    return torch.abs(kernel).mean()


def com(kernel):
    xx, yy = torch.meshgrid(
        torch.arange(kernel.shape[-2], device=kernel.device),
        torch.arange(kernel.shape[-1], device=kernel.device),
        indexing='ij'
    )
    xx = xx[None, None, ...]
    yy = yy[None, None, ...]
    ksum = kernel.sum((-1, -2, -3)) + 1e-5
    center_of_mass = (kernel * xx).sum((-1, -2, -3)) / ksum, (kernel * yy).sum((-1, -2, -3)) / ksum
    return center_of_mass


def center_reg(kernel: torch.Tensor):
    kernel_com = com(kernel)
    com_reg = (kernel.shape[-2] // 2 - kernel_com[-2]) ** 2 + (kernel.shape[-1] // 2 - kernel_com[-1]) ** 2
    return com_reg.mean()


def sum_to_one_reg(kernel_pre_norm: torch.Tensor):
    return torch.abs(kernel_pre_norm.sum(dim=(-1, -2)) - 1).mean()


class DiffInstructOnY(AbstractLoss):
    def __init__(
        self,
        config: Mapping[str, Any],
        x,
        b,
        aux_flow_nn: nn.Module,
        prtr_flow_nn: nn.Module,
        physics,
         ):
        super().__init__(has_val_loss=True, n_accum_steps=config['loss']['n_accum_steps'])
        self.x = x
        self.b = b
        #self.b.requires_grad_(False)
        self.mean = 0 
        self.std = 1
        self.opt_state = OptimizationState(0, OptState.INNER)
        self.lr_scheduler = None
        gs_enabled = False
        self.grad_scaler = torch.amp.grad_scaler.GradScaler(enabled=gs_enabled)
       
        # Note that clean_images does not imply true clean images. The images used for
        # the auxiliary loss are the ones generated with `kernel_nn`
        config['loss']['aux']['n_accum_steps'] = self.n_accum_steps  # Forced to be equal!
        aux_cfg = dict(loss=config['loss']['aux'], optim=config['optim']['aux'])
        self.aux_flow_loss = FlowMatchingLoss(
            config=aux_cfg, flow_nn=aux_flow_nn, clean_images=True, pretr_flow_nn=prtr_flow_nn
        )
        self.aux_flow_nn = aux_flow_nn
        self.noise_level = config["loss"]["noise_level"]
        self.prtr_flow_nn = prtr_flow_nn.requires_grad_(False)
        self.M = config["dataset"]["M"]
        # Regularizers
        self.boundary_mask = None
        self.sparse_reg = config['loss'].get('sparse_reg', None)
        self.symmetry_reg = config['loss'].get('symmetry_reg', None)
        self.center_reg = config['loss'].get('center_reg', None)
        self.sum_to_one_reg = config['loss'].get('sum_to_one_reg', None)
        self.gaussian_reg = config['loss'].get('gaussian_reg', None)
        self.physics = physics
        self.alpha = 12
        self.conditioning_acc = []
        self.lambda1 = config["loss"]["lambda1"]
        self.lambda1_tv = config["loss"].get("lambda1_tv",0)
        self.lambda2 = config["loss"]["lambda2"]
        self.optim_kernel = config['optim']['kernel']
        #self.kernel_opt, self.lr_scheduler = create_optimizer(self.optim_kernel, [self.x])#[self.x, self.b])
        self.kernel_opt, self.lr_scheduler = create_optimizer(self.optim_kernel, [self.x, self.b])

    def grad_di_func(self, physics,flow_diff, x, b, eps_noise, out_pre_relu):   #aggiungi sigma!!! (no, quello non dipende nè da x nè da b, è una costante che si aggiunge al resto, non è un peso da mettere dentro la formula)
        A = physics.conv
        x_fix = x.detach()
        b_fix = b.detach()
        A_adjoint = physics.conv.adjoint
        out_originale = A(x_fix) + b_fix

        den = 2*torch.sqrt(out_originale.clamp(min=1e-6)*self.M*self.alpha)

        derivata_clamp = (out_originale > 1e-6).float()
        
        relu_mask = (out_pre_relu > 0).float()

        masked_diff = relu_mask * flow_diff        # <- maschera applicata QUI, prima di tutto
        w = eps_noise / den * derivata_clamp
        low_res = masked_diff + w * masked_diff
        grad_x = A_adjoint(low_res)
        grad_b = low_res

        return grad_x, grad_b


    def loss_di(self, x, b, conditioning, mean = 0.0, std = 1.0, **kernel_kwargs):     #problema: dove inizializzare x e b?  mettere self magari...
        dev = x.device
        y1, eps_noise, out_pre_relu = self.physics(x,b)
        corrupt_shape = y1.shape
        y0 = torch.randn(corrupt_shape, device= dev)
        y1 = center(y1)
        #y1 = normalize(y1, mean, std)
        t = torch.rand([corrupt_shape[0], 1, 1, 1], device = dev)
        yt = (1-t)*y0 +t*y1
        with torch.no_grad():
            cuda_rng_state = torch.cuda.get_rng_state()
            prtr_velocity = self.prtr_flow_nn(yt, t, conditioning)
            torch.cuda.set_rng_state(cuda_rng_state)
            aux_velocity = self.aux_flow_nn(yt, t, conditioning)   #cerca clean_label in flow
            flow_diff = aux_velocity - prtr_velocity

        loss_di = (flow_diff * y1).mean() 
        reg_x = TV_reg(x) 
        reg_b =  gradient_reg(b)
        #grad_x, grad_b = self.grad_di_func(self.physics, flow_diff, x, b, eps_noise, out_pre_relu)
        #reg_hess = Hessian_reg(x)
        sum_loss = loss_di + self.lambda2*reg_b  + self.lambda1_tv*reg_x #potresti anche aggiungere std
        loss_info = {
        	"loss/di": loss_di.detach(),
        	"loss/reg_x" : reg_x.detach(),
        	"loss/reg_b" : reg_b.detach(),
        	"loss/sum_loss" : sum_loss.detach(),
                #"loss/reg_hess" : reg_hess.detach()
        }

        return sum_loss, loss_info#, 2*grad_x/y1.numel(), 2*grad_b/y1.numel()
        
    def kernel_regularization(self, kernel: torch.Tensor):
        loss_info = {}
        loss_sparse = 0
        if self.sparse_reg is not None and self.sparse_reg > 0:
            loss_sparse = self.sparse_reg * sparse_reg(kernel)
            loss_info["loss/sparse"] = loss_sparse.detach()
        loss_center = 0
        if self.center_reg is not None and self.center_reg > 0:
            loss_center = self.center_reg * center_reg(kernel)
            loss_info["loss/center"] = loss_center.detach()
        loss_sum_to_one = 0
        if self.sum_to_one_reg is not None and self.sum_to_one_reg > 0:
            loss_sum_to_one = self.sum_to_one_reg * sum_to_one_reg(kernel)
            loss_info["loss/sum_to_one"] = loss_sum_to_one.detach()
        loss_gaussian = 0
        if self.gaussian_reg is not None and self.gaussian_reg > 0:
            loss_gaussian = self.gaussian_reg * gaussian_reg(kernel)
            loss_info['loss/gaussian'] = loss_gaussian.detach()

        total_reg = loss_sparse + loss_center + loss_sum_to_one + loss_gaussian
        return total_reg, loss_info

    def __call__(self, trainer: BaseTrainer, batch: Batch):
        batch_dev = batch.cuda()
        assert batch_dev.clean is not None
        assert batch_dev.noise_level is not None

        if self.opt_state.state == OptState.OUTER:  # Outer loss (learn the kernel)
            with self.optimizer_step(self.kernel_opt, self.lr_scheduler, None, grad_scaler=self.grad_scaler):
                loss_di, loss_info = self.loss_di(
                    self.x,
                    self.b,
                    batch_dev.clean_conditioning,
                    self.mean,
                    self.std,
                    **batch_dev.meta
                )
                loss_di = loss_di / self.n_accum_steps
                self.grad_scaler.scale(loss_di).backward()
                
                ### calcolo norma
                loss_info["grad_norm"] = torch.linalg.norm(self.x.grad.flatten()).detach() + torch.linalg.norm(self.b.grad.flatten()).detach()
                loss_info["grad_norm/grad_b"] = torch.linalg.norm(self.b.grad.flatten()).detach()
                loss_info["grad_norm/grad_x"] = torch.linalg.norm(self.x.grad.flatten()).detach()
                #loss_info["grad_norm/grad_manual_x"] = torch.linalg.norm(grad_x.flatten()).detach()
                #loss_info["grad_norm/grad_manual_b"] = torch.linalg.norm(grad_b.flatten()).detach()
                #loss_info["grad_norm/difference x"] = torch.linalg.norm(self.x.grad.flatten() - grad_x.flatten()).detach() 
                #loss_info["grad_norm/difference b"] = torch.linalg.norm(self.b.grad.flatten() - grad_b.flatten()).detach()
                #cos_sim_x = torch.nn.functional.cosine_similarity(self.x.grad.flatten(), grad_x.flatten(), dim=0).item()
                #cos_sim_b = torch.nn.functional.cosine_similarity(self.b.grad.flatten(), grad_b.flatten(), dim=0).item()
                #loss_info["cosine_sim_x"] = cos_sim_x
                #loss_info["cosine_sim_b"] = cos_sim_b

            with torch.no_grad():
                lr = self.kernel_opt.param_groups[0]['lr']
                self.x.sub_(self.lambda1*lr)
                self.x.clamp_(min=0)
                self.b.clamp_(min=0)
            loss_info["learning_rate"] = lr
            if super().is_full_step_complete():
                self.opt_state.increment_state()
        else:   # Inner loss (learn the auxiliary score network)
            with torch.no_grad():
                if self.x is None:
                    self.init_params(batch_dev.clean)
                y1, eps_noise, out_pre_relu = self.physics(self.x, self.b)
            ker_batch = Batch(
                clean=y1,
                clean_label=batch_dev.clean_label,
                clean_conditioning=batch_dev.clean_conditioning,
                corrupt=None,
                corrupt_label=None,
                corrupt_conditioning=None,
                noise_level=None,
            )
            aux_loss_info = self.aux_flow_loss(trainer, ker_batch)
            loss_info = {
                k.replace("loss/", "loss/aux_"): v for k, v in aux_loss_info.items()
            }
            if self.aux_flow_loss.is_full_step_complete():
                self.opt_state.increment_state()

        return loss_info

    def is_full_step_complete(self) -> bool:
        # full_step: both inner and outer
        # Always called after `__call__`
        return super().is_full_step_complete() and self.opt_state.state == OptState.INNER

    def state_dict(self):
        state = {
            "aux": self.aux_flow_loss.state_dict(),
            "kernel_opt": self.kernel_opt.state_dict(),
            "kernel_lr": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "gs": self.grad_scaler.state_dict(),
        }
        if self.x is not None:
            state["x"] = self.x[0:1].detach().cpu()
        if self.b is not None:
            state["b"] = self.b[0:1].detach().cpu()
        return state

    def load_state_dict(self, state_dict):
        self.aux_flow_loss.load_state_dict(state_dict["aux"])
        self.kernel_opt.load_state_dict(state_dict["kernel_opt"])
        if self.lr_scheduler is not None:
            if "kernel_lr" not in state_dict:
                raise ValueError(
                    "Created loss with lr scheduler but state-dict has no lr scheduler state."
                )
            self.lr_scheduler.load_state_dict(state_dict["kernel_lr"])
        self.grad_scaler.load_state_dict(state_dict["gs"]) 
        if "x" in state_dict and self.x is not None:
            saved_x = state_dict["x"].to(self.x.device)
            if saved_x.dim() == 3:
                saved_x = saved_x.unsqueeze(0)
            if self.x.shape[0]>1 and saved_x.shape[0] == 1:
                B = self.x.shape[0]
                saved_x= saved_x.expand(B,-1,-1,-1)
            with torch.no_grad():
                self.x.copy_(saved_x)
        if "b" in state_dict and self.b is not None:
            saved_b = state_dict["b"].to(self.b.device)
            if saved_b.dim() == 3:
                saved_b = saved_b.unsqueeze(0)
            if self.b.shape[0]>1 and saved_b.shape[0] == 1:
                B = self.b.shape[0]
                saved_b= saved_b.expand(B,-1,-1,-1)
            with torch.no_grad():
                self.b.copy_(saved_b)


    @torch.no_grad()
    def val_loss(self, trainer: BaseTrainer, batch: Batch) -> Mapping[str, float | torch.Tensor]:
        vl = {}
        print("Computing validation loss..... to be fixed")        
        return vl
