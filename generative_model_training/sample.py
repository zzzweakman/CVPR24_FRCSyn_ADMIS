import os,sys
sys.path.insert(1, "../")

from typing import Any
import math
import numpy as np
import hydra
import torch
from pytorch_lightning.lite import LightningLite
import torch.nn.functional as F
import omegaconf
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from models.autoencoder.vqgan import VQEncoderInterface, VQDecoderInterface
from models.diffusion.ddpm import DenoisingDiffusionProbabilisticModel
from utils.helpers import ensure_path_join, denormalize_to_zero_to_one, normalize_to_neg_one_to_one


from tqdm import tqdm


class DiffusionSamplerLite(LightningLite):
    def run(self, cfg) -> Any:
        train_cfg_path = os.path.join("./configs/model/unet_cond_ca_cpd25.yaml")    
        train_cfg = omegaconf.OmegaConf.load(train_cfg_path) 

        # do not set seed to get different samples from each device
        self.seed_everything(cfg.sampling.seed * (1 + self.global_rank))

        # instantiate stuff from restoration config
        diffusion_model = instantiate(train_cfg)    

        # iDiff ckpt
        checkpoint_path = cfg.checkpoint.path
   
        # loading
        weights = torch.load(checkpoint_path, map_location="cpu")
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace('eps_model.', '') if 'eps_model' in k else k
            weights_dict[new_k] = v
        weights_dict_2 = {}
        for k, v in weights_dict.items():
            new_k = k.replace('module.', '') if 'module.' in k else k
            weights_dict_2[new_k] = v
        diffusion_model.load_state_dict(weights_dict_2, strict = False)
        diffusion_model = DenoisingDiffusionProbabilisticModel(eps_model=diffusion_model)
        
        # registrate model in lite
        diffusion_model = self.setup(diffusion_model)
      
        # sample size
        size = (3,128,128)
        # create VQGAN encoder and decoder for training in its latent space
        latent_encoder = VQEncoderInterface(
            first_stage_config_path=os.path.join("./", "models", "autoencoder",
                                                    "first_stage_config.yaml"),
            # according to your path
            encoder_state_dict_path=os.path.join("./ckpt/autoencoder/first_stage_encoder_state_dict.pt")                 
        )

        size = latent_encoder(torch.ones([1, *size])).shape[-3:]
        # del latent_encoder
        latent_encoder = self.setup(latent_encoder)
        latent_encoder.eval()
        latent_decoder = VQDecoderInterface(
            first_stage_config_path=os.path.join("./", "models", "autoencoder",
                                                    "first_stage_config.yaml"),
            # according to your path                                        
            decoder_state_dict_path=os.path.join("./ckpt/autoencoder/first_stage_decoder_state_dict.pt")
        )
        latent_decoder = self.setup(latent_decoder)
        latent_decoder.eval()
      
        # build path to save the samples
        assert cfg.sampling.contexts_file is not None
        
        # numpy array
        contexts = np.load(cfg.sampling.contexts_file)
        contexts = contexts[:10000]
        contexts_norm = np.linalg.norm(contexts,axis=1)
        contexts = contexts / contexts_norm[:, np.newaxis]
        print(contexts.shape)

        context_ids = list(i for i in range(0, contexts.shape[0]))


        model_name = cfg.checkpoint.path.split("/")[-1]
        if cfg.checkpoint.use_non_ema:
            model_name += "_non_ema"
        elif cfg.checkpoint.global_step is not None:
            model_name += f"_{cfg.checkpoint.global_step}"

        samples_dir = cfg.sampling.save_dir

        length_before_filter = len(context_ids)
        context_ids = list(filter(lambda i: not os.path.isfile(os.path.join(samples_dir, f"{i}.png")), context_ids))
        print(f"Skipped {length_before_filter - len(context_ids)} context ids, because for them files already seem to "
              f"exist!")
        context_ids = self.split_across_devices(context_ids)

        if self.global_rank == 0:
            with open(ensure_path_join(f"{samples_dir}.yaml"), "w+") as f:
                OmegaConf.save(config=cfg, f=f.name)
        
        np.random.seed(1337)
        # for id_name in context_ids:
        for id_index in range(0,len(context_ids)):

            prefix = str(context_ids[id_index])   
            print("sample " + prefix)
            # id condition
            context = torch.from_numpy(contexts[context_ids[id_index]]).float()  
            context = context.repeat(cfg.sampling.batch_size, 1).cuda()
            while not isinstance(diffusion_model, DenoisingDiffusionProbabilisticModel):
                diffusion_model = diffusion_model.module
            

            start_noise = None
            self.perform_sampling(
                diffusion_model=diffusion_model,
                n_samples=cfg.sampling.n_samples_per_context,
                size=size,
                batch_size=cfg.sampling.batch_size,
                samples_dir=samples_dir,
                prefix=prefix,
                context=context,
                latent_encoder=latent_encoder,
                latent_decoder=latent_decoder,
                start_batch = start_noise
            )

    @staticmethod
    def perform_sampling(
            diffusion_model, n_samples, size, batch_size, samples_dir,
            prefix: str = None, context: torch.Tensor = None, latent_encoder: torch.nn.Module = None, 
            latent_decoder: torch.nn.Module = None, start_batch: torch.Tensor = None):

        n_batches = math.ceil(n_samples / batch_size)

        samples_for_grid = []

        if context is not None:
            assert prefix is not None

        with torch.no_grad():    
            for _ in range(n_batches):
                # ddim sample
                batch_samples, _ = diffusion_model.sample_ddim(batch_size, size, x_T = start_batch, context=context)
                with torch.no_grad():
                    if latent_decoder:
                        batch_samples = latent_decoder(batch_samples).cpu()

                batch_samples = denormalize_to_zero_to_one(batch_samples)
                samples_for_grid.append(batch_samples)
                samples = torch.cat(samples_for_grid, dim=0)[:n_samples]

                # (b, 3, 112, 112)
                samples = F.interpolate(samples, size = [112,112], mode='bilinear')
                save_path = os.path.join(samples_dir, prefix)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                for sample_index in range(len(samples)):
                    save_image(samples[sample_index], ensure_path_join(save_path, str(sample_index) + ".png"))
                    
    def split_across_devices(self, L):
        if type(L) is int:
            L = list(range(L))

        chunk_size = math.ceil(len(L) / self.world_size)
        L_per_device = [L[idx: idx + chunk_size] for idx in range(0, len(L), chunk_size)]
        while len(L_per_device) < self.world_size:
            L_per_device.append([])

        return L_per_device[self.global_rank]

    @staticmethod
    def spherical_interpolation(value, start, target):
        start = torch.nn.functional.normalize(start)
        target = torch.nn.functional.normalize(target)
        omega = torch.acos((start * target).sum(1))
        so = torch.sin(omega)
        res = (torch.sin((1.0 - value) * omega) / so).unsqueeze(1) * start + (
                torch.sin(value * omega) / so).unsqueeze(1) * target
        return res


@hydra.main(config_path='./configs', config_name='sample_ddim_config', version_base=None)
def sample(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    sampler = DiffusionSamplerLite(devices="auto", accelerator="auto")
    sampler.run(cfg)


if __name__ == "__main__":
    sample()
