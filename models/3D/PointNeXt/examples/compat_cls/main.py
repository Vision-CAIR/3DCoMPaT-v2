"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
import pdb
import torch.nn.functional as F
from collections import defaultdict, Counter
import yaml
from torch.utils.tensorboard import SummaryWriter
from compat_loader import CompatLoader3DCls as Compat
from compat_utils import *

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))
from openpoints.models import build_model_from_cfg
from openpoints.models.layers import torch_grouping_operation, knn_point
from openpoints.loss import build_criterion_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.dataset import build_dataloader_from_cfg, get_class_weights, get_features_by_keys
from openpoints.transforms import build_transforms_from_cfg
from openpoints.utils import AverageMeter, ConfusionMatrix
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port
from openpoints.models.layers import furthest_point_sample

from pathlib import Path
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

coarse_seg_classes = {"0": [10, 41, 0], "1": [10, 41, 19], "2": [10, 19, 24], "3": [19, 4, 3], "4": [10, 5, 3, 26], "5": [2, 32, 26, 3], "6": [32, 10, 41], "7": [31, 10, 6, 7], "8": [10, 9, 25, 42], "9": [40, 21, 3], "10": [10, 12], "11": [10, 41, 13, 42], "12": [2, 32, 26, 3], "13": [14, 10], "14": [15, 28, 3], "15": [16, 28, 20], "16": [40, 21, 19, 3], "17": [40, 21, 19, 3], "18": [10, 11, 8], "19": [10, 18, 3], "20": [10, 24, 19, 3], "21": [31, 40, 3], "22": [10, 19, 3], "23": [38, 22], "24": [23, 11, 3], "25": [32, 2, 1, 26, 3], "26": [2, 32, 26, 3], "27": [10, 19, 3], "28": [37, 10, 27], "29": [39, 40, 21], "30": [34, 10, 33], "31": [35, 17, 3], "32": [41, 21, 36], "33": [32, 2, 1, 26, 3], "34": [29, 30, 21, 3], "35": [2, 32, 3], "36": [2, 32, 1, 3], "37": [40, 21, 3], "38": [10, 32, 24], "39": [10, 19], "40": [10, 19, 3], "41": [10, 27]}
fine_seg_classes   = {"0": [6, 4, 10, 9, 2, 5, 3, 273, 8, 274, 7, 1, 271, 11], "1": [25, 24, 22, 20, 201, 271, 219, 273, 23, 138, 155, 21, 167, 168], "2": [172, 26, 262, 27, 92, 138, 155, 271, 167], "3": [252, 251, 268, 174, 172, 227, 28, 262, 201, 253, 29, 211, 261, 269, 138, 155, 266, 167], "4": [35, 253, 31, 141, 33, 37, 34, 39, 269, 252, 268, 172, 138, 155, 32, 40, 251, 227, 201, 30, 212, 36, 38, 267, 266], "5": [14, 17, 253, 224, 18, 223, 225, 269, 13, 252, 268, 172, 16, 219, 138, 155, 226, 251, 201, 12, 15, 212, 211, 92, 266], "6": [43, 50, 42, 52, 63, 41, 57, 44, 46, 59, 47, 56, 53, 60, 62, 64, 66, 45, 54, 51, 55, 48, 65, 58, 49, 61], "7": [252, 67, 154, 268, 172, 174, 69, 171, 211, 220, 68, 140, 269, 267, 153, 266, 70], "8": [86, 87, 83, 80, 223, 82, 169, 85, 79, 252, 78, 90, 75, 16, 74, 77, 226, 84, 73, 88, 76, 71, 211, 72, 89, 81, 91, 167], "9": [174, 253, 141, 158, 261, 269, 173, 252, 120, 268, 172, 262, 140, 138, 155, 251, 227, 201, 166, 92, 211, 267, 266, 167], "10": [252, 174, 97, 95, 94, 171, 96, 157, 201, 211, 155], "11": [98, 107, 110, 116, 223, 114, 101, 169, 103, 119, 105, 104, 111, 113, 109, 115, 100, 16, 108, 106, 102, 112, 203, 118, 12, 99, 117, 167], "12": [14, 202, 224, 18, 271, 223, 225, 169, 156, 13, 252, 16, 219, 155, 226, 19, 251, 170, 201, 12, 15, 212, 211, 266, 179], "13": [125, 123, 121, 268, 129, 127, 166, 128, 122, 270, 211, 124, 266, 126], "14": [252, 251, 154, 201, 131, 211, 132, 219, 138, 155, 216, 130], "15": [134, 136, 135, 270, 137, 133, 179], "16": [268, 174, 139, 172, 211, 140, 269, 266, 167], "17": [174, 253, 141, 158, 261, 269, 173, 252, 268, 172, 262, 140, 138, 155, 251, 227, 201, 203, 166, 211, 92, 267, 266, 167], "18": [142, 145, 93, 143, 270, 219, 147, 144, 146], "19": [0, 150, 151, 215, 219, 149, 148, 152, 228], "20": [161, 159, 201, 160, 211, 273, 155, 271, 162, 167, 168], "21": [174, 253, 222, 223, 220, 252, 165, 262, 163, 16, 164, 155, 226, 251, 201, 211, 221, 153, 266], "22": [175, 172, 211, 176, 155, 265, 177, 167], "23": [184, 251, 174, 181, 253, 183, 155, 180, 182], "24": [93, 270, 185, 271, 190, 198, 192, 199, 196, 200, 197, 194, 195, 187, 188, 252, 193, 219, 155, 251, 227, 201, 203, 191, 211, 186, 189], "25": [253, 18, 223, 225, 13, 252, 172, 16, 219, 138, 155, 226, 19, 251, 201, 12, 15, 212, 211, 266], "26": [13, 226, 252, 223, 251, 172, 16, 201, 225, 12, 224, 211, 212, 219, 269, 138, 155], "27": [252, 205, 154, 207, 210, 208, 209, 211, 204, 206, 216, 167], "28": [252, 251, 201, 213, 211, 245, 214, 155, 265, 263, 167], "29": [252, 158, 174, 172, 227, 262, 268, 166, 253, 140, 141, 269, 267, 138, 266, 167], "30": [234, 252, 231, 233, 0, 150, 151, 270, 215, 232, 149, 148, 152, 229, 228, 230], "31": [236, 241, 0, 150, 151, 215, 211, 235, 240, 149, 148, 152, 237, 239, 228, 238], "32": [244, 272, 219, 271, 242, 243], "33": [253, 223, 225, 173, 13, 252, 172, 16, 138, 155, 226, 251, 19, 201, 12, 15, 212, 267, 266], "34": [252, 246, 174, 250, 262, 218, 201, 247, 249, 248, 155, 217], "35": [13, 252, 226, 223, 251, 19, 16, 201, 225, 12, 224, 156, 211, 219, 18, 202, 155, 271], "36": [224, 18, 271, 223, 225, 169, 13, 252, 157, 16, 219, 155, 226, 251, 19, 201, 12, 15, 211, 266], "37": [174, 178, 253, 141, 261, 269, 173, 252, 120, 268, 172, 262, 157, 140, 138, 155, 251, 227, 201, 211, 92, 267, 266, 167], "38": [256, 258, 259, 260, 255, 254, 257], "39": [252, 251, 172, 174, 262, 201, 211, 155, 167], "40": [174, 253, 141, 271, 261, 269, 252, 120, 268, 172, 262, 157, 140, 138, 155, 251, 227, 166, 201, 92, 267, 266, 167], "41": [264, 213, 201, 211, 155, 265, 263]}
seg_classes  = {}

classes = ['airplane', 'bag', 'cap', 'car', 'chair',
               'earphone', 'guitar', 'knife', 'lamp', 'laptop',
               'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']

cls_parts = {'earphone': [16, 17, 18], 'motorbike': [30, 31, 32, 33, 34, 35], 'rocket': [41, 42, 43],
                'car': [8, 9, 10, 11], 'laptop': [28, 29], 'cap': [6, 7], 'skateboard': [44, 45, 46], 'mug': [36, 37],
                'guitar': [19, 20, 21], 'bag': [4, 5], 'lamp': [24, 25, 26, 27], 'table': [47, 48, 49],
                'airplane': [0, 1, 2, 3], 'pistol': [38, 39, 40], 'chair': [12, 13, 14, 15], 'knife': [22, 23]}
cls2parts = []
cls2partembed = torch.zeros(16, 275)
for i, cls in enumerate(classes):
    idx = cls_parts[cls]
    cls2parts.append(idx)
    cls2partembed[i].scatter_(0, torch.LongTensor(idx), 1)

def mean_iou(label, predicted_label, eps=1e-10, num_part=275): 
    with torch.no_grad(): 
        predicted_label =  predicted_label.contiguous().view(-1) 
        label = label.contiguous().view(-1)

        iou_single_class = [] 
        for class_number in range(0, num_part): 
            true_predicted_class = predicted_label == class_number 
            true_label = label == class_number 
            if true_label.long().sum().item() == 0: 
                iou_single_class.append(np.nan) 
            else: 
                intersection = torch.logical_and(true_predicted_class, true_label).sum().float().item() 
                union = torch.logical_or(true_predicted_class, true_label).sum().float().item() 

                iou = (intersection + eps) / (union + eps) 
                iou_single_class.append(iou) 

        return iou_single_class


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_part_seg_ssg', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=251, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=4096, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--shape_prior', action='store_true', default=False, help='use shape prior')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--data_name', type=str, default="fine", help ="data_name")

    return parser.parse_args()


def main(args, cfg):
    def log_string(str):
        logging.info(str)
        print(str)
        
    def validate_fn(model, testDataLoader, cfg):
        with torch.no_grad():

            model = model.eval()

            mean_correct = []
            class_acc = np.zeros((num_classes, 3))

            for j, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):

                points, target = points.cuda(), target.cuda()

                # points = points.transpose(2, 1)
                pred = model(points)
                
                pred_choice = pred.data.max(1)[1]

                for cat in np.unique(target.cpu()):
                    classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                    class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
                    class_acc[cat, 1] += 1

                correct = pred_choice.eq(target.long().data).cpu().sum()
                mean_correct.append(correct.item() / float(points.size()[0]))

            class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
            class_acc = np.mean(class_acc[:, 2])
            instance_acc = np.mean(mean_correct)
            log_string('Validation Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            return instance_acc

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.rank == 0:
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        writer = SummaryWriter(log_dir=cfg.run_dir)
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)


    root = '../data/fine_grained/'
    print(root)
    TRAIN_DATASET = Compat(data_root=root, num_points=args.npoint, split='train', transform=None)
    TEST_DATASET = Compat(data_root =root, num_points=args.npoint, split='valid', transform=None)
    VALID_DATASET = Compat(data_root=root, num_points=args.npoint, split='test', transform=None)

    validDataLoader = torch.utils.data.DataLoader(VALID_DATASET, batch_size=cfg.batch_size, shuffle=True, num_workers=10, drop_last=True)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=cfg.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=cfg.batch_size, shuffle=False, num_workers=10)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    # num_classes = 43
    # num_part = 235
    num_classes = 42
    num_part    = 275 
    seg_classes = fine_seg_classes
    if(cfg.dataset.common.NAME == "coarse"):
        num_part = 43
        logging.info("using 43 parts")
        seg_classes = coarse_seg_classes
    
    # cls2partembed = torch.zeros(num_classes, num_part)
    # for cls, idx in seg_classes.items():
    #     cls2partembed[int(cls)].scatter_(0, torch.LongTensor(idx), 1)

    # if cfg.model.get('decoder_args', False):
    #     cfg.model.decoder_args.cls2partembed = cls2partembed
    if not cfg.model.get('criterion_args', False):
        cfg.model.criterion_args = cfg.criterion_args
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    model = build_model_from_cfg(cfg.model).cuda()
    
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info(type(model))
    logging.info('Number of params: %.4f M' % (model_size / 1e6))


    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(0)
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')
    
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    
    if cfg.pretrained_path is not None:
        checkpoint = torch.load(cfg.pretrained_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_metrics = validate_fn(model, validDataLoader, cfg)
        return test_metrics
    else:
        logging.info('Training from scratch')

    # criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()


    best_acc = 0
    global_epoch = 0
    best_instance_acc = 0
    best_class_acc = 0

    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        mean_correct = []

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, cfg.epoch))
        model = model.train()
        # '''learning one epoch'''
        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):

            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()

            # seg_pred = model(points, cls0= label)
            logits, loss = model.get_logits_loss(points, target)
            logging.info(target.shape)
            logging.info(logits.data.max(1)[1].shape)
            pred_choice = logits.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (cfg.batch_size))

            loss.backward()
            optimizer.step()

        train_instance_acc = np.mean(mean_correct)
        logging.info(mean_correct)
        log_string('Train accuracy is: %.5f' % train_instance_acc)
        assert True==False

        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
            general_miou = []

            model = model.eval()

            mean_correct = []
            class_acc = np.zeros((num_classes, 3))

            for j, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):

                points, target = points.cuda(), target.cuda()

                # points = points.transpose(2, 1)
                pred = model(points)
                
                pred_choice = pred.data.max(1)[1]

                for cat in np.unique(target.cpu()):
                    classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                    class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
                    class_acc[cat, 1] += 1

                correct = pred_choice.eq(target.long().data).cpu().sum()
                mean_correct.append(correct.item() / float(points.size()[0]))

            class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
            class_acc = np.mean(class_acc[:, 2])
            instance_acc = np.mean(mean_correct)
            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1
                best_class_acc = class_acc
            log_string('Validation Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logging.info('Save model...')
                savepath = str(cfg.ckpt_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            # return instance_acc, class_acc
        scheduler.step(epoch)
        global_epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('3DCompat classification training')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--data_name', type=str, default="fine", help ="data_name")

    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)
    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # logger
    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]  # task/dataset name, \eg s3dis, modelnet40_cls
    cfg.cfg_basename = args.cfg.split('.')[-2].split('/')[-1]  # cfg_basename, \eg pointnext-xl
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        str(cfg.mode)+"cls",
        cfg.cfg_basename,
    ]
    opt_list = [] # for checking experiment configs from logging file
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            opt_list.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    cfg.opts = '-'.join(opt_list)

    cfg.is_training = cfg.mode not in ['test', 'testing', 'val', 'eval', 'evaluation']

    if cfg.mode in ['resume', 'test', 'val']:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = [cfg.mode]
    else:
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
        cfg.wandb.tags = tags
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path

    # wandb config
    cfg.wandb.name = cfg.run_name
    # pdb.set_trace()
    main(args, cfg)
