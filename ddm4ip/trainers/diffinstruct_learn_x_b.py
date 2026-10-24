import os
import warnings
from omegaconf import DictConfig
import torch
from ddm4ip.networks.corrupt import Corruption_train_fft, Corruption_train_fft_ext
from ddm4ip.data.base import Batch, Datasplit, init_dataset
from ddm4ip.losses.di_Y import DiffInstructOnY
from ddm4ip.networks import init_kernel_net
from ddm4ip.networks.unets import init_unet_with_defaults
from ddm4ip.trainers.base import BaseTrainer, init_trts_dataloaders
from ddm4ip.trainers.plots import plot_conditioning_hist, plot_explicit_kernels, plot_imggen, plot_implicit_kernels, plot_spacevarying_kernels, plot_xb, tensor_grid_to_fig_with_cbar, plot_xb_p_cbars
from ddm4ip.utils import distributed
from ddm4ip.utils.easy_dict import print_module_summary
from ddm4ip.degradations.degradation import load_filter_from_file

class DiffinstructOpTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.plot_batch = None
        self.corrupt_img_size: tuple[int, int, int] | None = None

    def init_datasets(self, cfg: DictConfig):
        if self.is_training:
            return {
                Datasplit.TRAIN: init_dataset(cfg, split=Datasplit.TRAIN, is_paired=False),
                Datasplit.TEST: init_dataset(cfg, split=Datasplit.TEST, is_paired=True)
            }
        return {
            Datasplit.TEST: init_dataset(cfg, split=Datasplit.TEST, is_paired=True)
        }

    def init_dataloaders(self, cfg, dsets):
        # No infinite data-loader during evaluation
        return init_trts_dataloaders(
            cfg, dsets, start_idx=self.start_global_step, is_infinite=self.is_training
        )
        
    def init_x_and_b(self, batch, device):
        #ancora meglio sarebbe fare la media 
        #x0 = A_t(y_obs[0:1]).squeeze(0).to(device)
        physics = self.models["physics"]
        batch_clean = batch.clean.to(device)
        B = batch_clean.shape[0]
        with torch.no_grad():   #serve?
            x0 = physics.A_t(batch_clean[0:1]).squeeze(0).expand(B, -1, -1, -1).to(device)#/489.2048344026573  #era con batch_clean intera (prima ancora facevo expand e partivo da batch_clean[0:1] e senza expand..  controlla che tutti i canali siano uguali e la shape sia okay	 ... 
        x = torch.nn.Parameter(x0.detach().clone())
        torch.save(x,"x_init.pt")
        b = torch.nn.Parameter(0.05*torch.ones(batch_clean.shape, device = device))  #controllare di aver preso bene e non rompa tutto il grafo. prova pls a aggiungere la batch
        torch.save(b,"b_init.pt")
        #b = torch.nn.Parameter(torch.mean(batch_clean, dim = 0,device = device)+ 0.0001* torch.ones(batch_clean.shape, device = device)  #non vengono degli 0? aggiungerci un 1e-6 magari
        #b = torch.load("/leonardo_work/IscrC_GOLD-FM_0/cozzani/flow_fluo/ddm4ip/networks/background.pt").unsqueeze(0).unsqueeze(0).expand(B,-1,-1,-1).to(device)/489.2048344026573
        return x, b        

    def init_models(self, cfg, dsets, device):
        dataset = self.get_any_dset(dsets)
        assert dataset is not None
        # Load the pre-trained model
        prt_flow_cfg = cfg["models"]["pretrained_flow"]
        if "path" not in prt_flow_cfg:
            raise ValueError(
                "'pretrained_flow' configuration must contain "
                "'path' key pointing to the pretrained model checkpoint."
            )
        if not os.path.isfile(prt_path := prt_flow_cfg["path"]):
            raise FileNotFoundError(
                f"Pretrained flow model path '{prt_path}' is not a file."
            )
        ckpt = torch.load(prt_path, map_location="cpu", weights_only=False)
        if "ema" in ckpt:
            state = ckpt["ema"]["emas"][-1]
            print(f"Loading pretrained flow model. EMA state with std: {ckpt['ema']['stds'][-1]}")
        elif "flow_nn" in ckpt:
            # This is the output of `finetune_flow`
            state = ckpt["flow_nn"]["state_dict"]
        else:
            raise ValueError(list(ckpt.keys()))


        
        std = 1
        #physics = Corruption_train(noise_level=cfg["loss"]["noise_level"], do_blur = True, device = device).to(device)#circular = False, device = device)
        alpha3 = cfg["dataset"].get("alpha3",1)
        allow_negative_vals = cfg["dataset"].get("allow_negative_values", False)
        ext = cfg["dataset"].get("extended_domain",False)
        if ext:
            physics = Corruption_train_fft_ext(kwidth = cfg["dataset"]["kwidth"], ksigma = cfg["dataset"]["ksigma"], alpha = cfg["dataset"]["alpha"], alpha2 = cfg["dataset"].get("alpha2", 1), alpha3 = alpha3, undersampling = cfg["dataset"]["undersampling"], noise_level=cfg["loss"]["noise_level"], M = cfg["dataset"]["M"], allow_negative_values= allow_negative_vals, circular = True, do_blur = True, double_poisson = cfg["dataset"].get("double_poisson",False),device = device).to(device)
        else:
            physics = Corruption_train_fft(kwidth = cfg["dataset"]["kwidth"], ksigma = cfg["dataset"]["ksigma"], alpha = cfg["dataset"]["alpha"], alpha2 = cfg["dataset"].get("alpha2", 1), alpha3 = alpha3, undersampling = cfg["dataset"]["undersampling"], noise_level=cfg["loss"]["noise_level"], M = cfg["dataset"]["M"], allow_negative_values= allow_negative_vals, circular = True, do_blur = True, double_poisson = cfg["dataset"].get("double_poisson",False),device = device).to(device)
        C, H, W = dataset.clean_img_size
        physics_fake_inputs = [
            torch.zeros([self.batch_size, C, H*cfg["dataset"]["undersampling"], W*cfg["dataset"]["undersampling"]], device=device),
            torch.zeros([self.batch_size, C, H, W ], device=device)
        ] 
        physics_fake_output, eps_noise, out_pre_relu = physics(*physics_fake_inputs)
        pred_corrupt_size = (physics_fake_output.shape[1],physics_fake_output.shape[2], physics_fake_output.shape[3])
        self.corrupt_img_size = pred_corrupt_size
        prtr_flow_nn = init_unet_with_defaults(
            cfg=prt_flow_cfg,
            img_size=pred_corrupt_size,
            cond_ch=dataset.corrupt_conditioning_channels,
            label_dim=dataset.label_dim,
        )
        prtr_flow_nn.load_state_dict(state)
        prtr_flow_nn = prtr_flow_nn.to(device)
        
        # Create the auxiliary model to match the pre-trained model
        init_from_pretrained =True 
        if (aux_flow_cfg := cfg["models"].get("aux_flow")) is None:
            aux_flow_cfg = prt_flow_cfg
            init_from_pretrained = True
        aux_flow_nn = init_unet_with_defaults(
            aux_flow_cfg,
            img_size=pred_corrupt_size,
            cond_ch=dataset.corrupt_conditioning_channels,
            label_dim=dataset.label_dim,
        )
        if init_from_pretrained:
            aux_flow_nn.load_state_dict(state)
        aux_flow_nn = aux_flow_nn.to(device)

        if distributed.get_rank() == 0:
            print("-----------------------AUXILIARY FLOW NN------------------------")
            cond_ch = dataset.corrupt_conditioning_channels
            print_module_summary(aux_flow_nn, [
                torch.zeros([self.batch_size, *pred_corrupt_size], device=device),
                torch.ones([self.batch_size], device=device),
                torch.zeros([self.batch_size, dataset.label_dim], device=device),
                torch.zeros([self.batch_size, cond_ch, *pred_corrupt_size[1:]], device=device)
            ], max_nesting=2)
            print("-------------------------KERNEL FLOW NN-------------------------")
            #print_module_summary(kernel_nn, kernel_nn_fake_inputs, max_nesting=2)
        return {
            "aux_flow_nn": aux_flow_nn,
            "prtr_flow_nn": prtr_flow_nn,
            "physics": physics,
        }

    def init_loss(self, cfg, models, x=None, b=None):
        loss = DiffInstructOnY(cfg, x, b,  **models)
        return loss

    def validate_batch(self, val_batch):
        return self.loss_optim.val_loss(self, val_batch)

    def get_any_dset(self, dsets):
        return next(d for d in dsets.values() if d is not None)

    def make_plots(self, cfg, dsets, dloaders, models, tb_writer, global_step):
        if self.plot_batch is None:
            # This ensures we always use the same batch for plotting.
            if (ds := dsets.get(Datasplit.TEST)) is None:
                warnings.warn("Using training batch for plotting.")
                ds = dsets[Datasplit.TRAIN]
                assert ds is not None
            # Don't use the dataloader to get around data-sampling artefacts.
            # messing around with indexes to get decent images for plotting
            self.plot_batch = Batch.collate_fn([ds[i] for i in range(self.batch_size)])

        dset = self.get_any_dset(dsets)
        assert self.corrupt_img_size is not None
        mean = 0
        std = 1

        p1 = plot_imggen(cfg, models["aux_flow_nn"], self.plot_batch, self.corrupt_img_size, dset, self.device, mean, std, cfg["dataset"]["M"])
        if p1 is not None and distributed.get_rank() == 0:
            fig = tensor_grid_to_fig_with_cbar(p1, cmap='hot')
            self.log_image(fig, 'diffusion/generated', tb_writer) #prima era p1


        p2 = None
        p2 = plot_xb(self.plot_batch, self.loss_optim.x, self.loss_optim.b, self.models["physics"], 16, cfg["dataset"]["M"])
        if p2 is not None and distributed.get_rank() == 0:
            self.log_image(p2, 'obtained_imgs/optimized', tb_writer)
        p3 = None
        p3 = plot_xb_p_cbars(cfg, self.plot_batch, self.loss_optim.x, self.loss_optim.b, self.models["physics"], 16, cfg["dataset"]["M"])
        if p3 is not None and distributed.get_rank() == 0:
            self.log_image(p3, 'obtained_imgs/multiple_cbars', tb_writer)

        
