import os
from idna import valid_contextj
import torch
import argparse
import datetime
import numpy as np
import random
import shutil

from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader

from core.builders import build_dataset, build_network, build_optimizer
from utils.runtime_utils import cfg, cfg_from_yaml_file, validate
from utils.vis_utils import visualize_numpy, visualize_part

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

args, cfg = parse_config()
exp_dir = ('/').join(args.ckpt.split('/')[:-2])


# Build Dataloader
val_dataset = build_dataset(cfg, split='val')
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)

# Build Network
net = build_network(cfg)
state_dict = torch.load(args.ckpt)
epoch = state_dict['epoch']
net.load_state_dict(state_dict['model_state_dict'])
net = net.cuda()
net.eval()

print('Evaluating Epoch: ', epoch)
visualize_part(net, val_dataloader)