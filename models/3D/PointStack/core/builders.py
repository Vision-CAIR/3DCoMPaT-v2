import os
import torch

from . import datasets
from .networks import networks

def build_network(cfg, topology = None):   
    net = networks.__all__[cfg.NETWORK.NAME](cfg, topology = topology)

    return net

def build_dataset(cfg, split = 'train'):
    Dataset = datasets.__all__[cfg.DATASET.NAME](cfg, split = split)
    return Dataset

def build_optimizer(cfg, params, data_loader_length, mode = None):
    opt_cfg = cfg.OPTIMIZER
    
    if mode == None:
        if (opt_cfg.NAME.lower() == 'adam'):
            opt = torch.optim.Adam(params, lr=opt_cfg.LR, betas=opt_cfg.BETAS, weight_decay=opt_cfg.WEIGHT_DECAY)
        elif (opt_cfg.NAME.lower() == 'sgd'):
            opt = torch.optim.SGD(params, lr=opt_cfg.LR, momentum=opt_cfg.MOMENTUM, weight_decay=opt_cfg.WEIGHT_DECAY, nesterov=opt_cfg.NESTEROV)

        if (opt_cfg.SCHEDULER is not None):
            if opt_cfg.SCHEDULER == 'cosine_annealing':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0 = opt_cfg.WARM_RESTART_EVERY * data_loader_length, eta_min = opt_cfg.MIN_LR)
    else:
        
        if (getattr(opt_cfg, mode).NAME.lower() == 'adam'):
            opt = torch.optim.Adam(params, lr=getattr(opt_cfg, mode).LR, betas=getattr(opt_cfg, mode).BETAS, weight_decay=getattr(opt_cfg, mode).WEIGHT_DECAY)
        elif (getattr(opt_cfg, mode).NAME.lower() == 'sgd'):
            opt = torch.optim.SGD(params, lr=getattr(opt_cfg, mode).LR, momentum=getattr(opt_cfg, mode).MOMENTUM, weight_decay=getattr(opt_cfg, mode).WEIGHT_DECAY, nesterov=getattr(opt_cfg, mode).NESTEROV)

        if (getattr(opt_cfg, mode).SCHEDULER is not None):
            if getattr(opt_cfg, mode).SCHEDULER.lower()== 'cosine_annealing':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = opt_cfg.MAX_EPOCH//2)
    
    return opt, scheduler
        