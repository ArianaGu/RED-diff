import datetime
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard as tb
import torchvision.utils as tvu
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from algos import build_algo
from datasets import build_loader
from models import build_model
from models.classifier_guidance_model import ClassifierGuidanceModel
from models.diffusion import Diffusion
from utils.distributed import get_logger, init_processes, common_init
from utils.functions import get_timesteps, postprocess, preprocess, strfdt
from utils.degredations import get_degreadation_image
from utils.save import save_result
from PIL import Image



def main(cfg):
    model, classifier = build_model(cfg)
    model.eval()
    if classifier is not None:
        classifier.eval()
    diffusion = Diffusion(**cfg.diffusion)
    cfg.algo.deg = "deblur_gauss"
    cfg.exp.num_steps = 1000
    cfg.algo.grad_term_weight = 0.25                #lambda
    cfg.algo.lr = 0.2
    cg_model = ClassifierGuidanceModel(model, classifier, diffusion, cfg)   #?? what is the easiest way to call stable diffusion?

    algo = build_algo(cg_model, cfg)
        
    H = algo.H
    # load a single image 
    fpath = '_exp/00003.png'
    img = Image.open(fpath).convert('RGB')
    # convert img to tensor
    x = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    n, c, h, w = x.size()
    x = x.cuda()

    x = preprocess(x)
    ts = get_timesteps(cfg)

    kwargs = {}
            
    y_0 = H.H(x)
    y_0 = y_0 + torch.randn_like(y_0) * cfg.algo.sigma_y * 2
    kwargs["y_0"] = y_0
    yo = torch.reshape(y_0, (1, 3, 256, 256))
    yo = postprocess(yo).detach().cpu()
    img = tvu.make_grid(yo, nrow=1).permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save('_exp/00003_degraded.png')

    start_time = time.time()
    xt_s, _ = algo.sample(x, y_0, ts, **kwargs)
    finish_time_sample = time.time() - start_time
    print('finish_time_sample', finish_time_sample)
    
    xo = postprocess(xt_s).detach().cpu()
    # convert tensor to img
    img = tvu.make_grid(xo, nrow=1).permute(1, 2, 0).numpy()
    img = (img/img.max()* 255).astype(np.uint8)
    img = Image.fromarray(img)
    # save the image
    img.save('_exp/00003_sampled.png')


@hydra.main(version_base="1.2", config_path="_configs", config_name="ddrmpp")
def main_dist(cfg: DictConfig):
    cwd = HydraConfig.get().runtime.output_dir
    init_processes(0, 1, main, cfg, cwd)
    

if __name__ == "__main__":
    main_dist()


# cfg = OmegaConf.load('_configs/ffhq256_uncond.yaml')
# cfg.dist = OmegaConf.create()
# cfg.dist.backend = 'nccl'
# cfg.dist.master_address = 'localhost'
# cfg.dist.port = '29500'
# cwd = os.getcwd()