import os
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
from utils.vis_utils import visualize_numpy
from compat_loader import CompatLoader3D as CompatSeg
import provider 

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--data_name', type=str, default="fine", help ="data_name")

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

args, cfg = parse_config()
exp_dir = ('/').join(args.ckpt.split('/')[:-2])

random_seed = cfg.RANDOM_SEED # Setup seed for reproducibility
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# Build Dataloader
root = os.path.join(os.getcwd(), '../data/'+args.data_name+'_grained/')
val_dataset = CompatSeg(data_root =root, num_points=cfg.DATASET.NUM_POINTS, split='test', transform=None)
val_dataloader = DataLoader(val_dataset, batch_size=cfg.OPTIMIZER.BATCH_SIZE, shuffle=False, drop_last=False)

# Build Network and Optimizer
net = build_network(cfg)
state_dict = torch.load(args.ckpt)
epoch = state_dict['epoch']
net.load_state_dict(state_dict['model_state_dict'])
net = net.cuda()
net.eval()

print('Evaluating Epoch: ', epoch)
val_dict = validate(net, val_dataloader, net.get_loss, 'cuda', args.data_name, is_segmentation = cfg.DATASET.IS_SEGMENTATION)

if cfg.DATASET.IS_SEGMENTATION:
    best_inctance_avg_iou = np.round(val_dict['miou'], 4)
    print('test_inctance_avg_iou', best_inctance_avg_iou)

    best_class_avg_iou = np.round(val_dict['class_avg_miou'], 4)
    print('test_class_avg_iou', best_class_avg_iou)

    best_general_miou = np.round(val_dict['general_miou'], 4)
    print('test_general_miou', best_general_miou)

    acc = np.round(val_dict['accuracy'], 4)
    print('acc', acc)

else:
    val_loss    = np.round(val_dict['loss'], 4)
    val_acc     = np.round(val_dict['acc'], 2)
    val_acc_avg = np.round(val_dict['acc_avg'], 2)

    print('val_loss', val_loss)
    print('val_acc', val_acc)
    print('val_acc_avg', val_acc_avg)


if cfg.DATASET.IS_SEGMENTATION:
    with open(exp_dir + '/eval_best.txt', 'w') as f:
        f.write('Best Epoch: ' + str(epoch))
        f.write('\nBest inctance_avg_iou: ' + str(best_inctance_avg_iou))
        f.write('\nBest class_avg_iou: ' + str(best_class_avg_iou))
        f.write('\nBest general_miou: ' + str(best_general_miou))
        f.write('\nBest acc: ' + str(acc))

else:
    with open(exp_dir + '/eval_best.txt', 'w') as f:
        f.write('Best Epoch: ' + str(epoch))
        f.write('\nBest Acc: ' + str(val_acc))
        f.write('\nBest Mean Acc: ' + str(val_acc_avg))
        f.write('\nBest Loss: ' + str(val_loss))


torch.save(state_dict['model_state_dict'], exp_dir + '/ckpt_model_only.pth')

