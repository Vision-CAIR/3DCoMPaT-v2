import os
import torch
import argparse
import datetime
import numpy as np
import random
import shutil
import json 

from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader

from core.builders import build_network, build_optimizer
from utils.runtime_utils import cfg, cfg_from_yaml_file, validate
from utils.vis_utils import visualize_numpy
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))
from compat_loader import CompatLoader3D as CompatSeg
import provider 

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--exp_name', type=str, default=None, help='specify experiment name for saving outputs')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--random_seed', type=int, default=0, help='random seed number')
    parser.add_argument('--val_steps', type=int, default=1, help='perform validation every n steps')
    parser.add_argument('--pretrained_ckpt', type = str, default = None, help='path to pretrained ckpt')
    parser.add_argument('--data_name', type=str, default="fine", help ="data_name")

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    exp_dir = cfg.ROOT_DIR / 'experiments' / cfg.DATASET.NAME / args.exp_name
    os.makedirs(exp_dir, exist_ok=True)
    shutil.copy2(args.cfg_file, exp_dir)

    return args, cfg

args, cfg = parse_config()

random_seed = cfg.RANDOM_SEED # Setup seed for reproducibility
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
root = os.path.join(os.getcwd(), '../data/'+args.data_name+'_grained/')
# Build Dataloader
train_dataset = CompatSeg(data_root=root, num_points=cfg.DATASET.NUM_POINTS, split='train', transform=None)
train_dataloader = DataLoader(train_dataset, batch_size=cfg.OPTIMIZER.BATCH_SIZE, shuffle=True, drop_last=True)

val_dataset = CompatSeg(data_root =root, num_points=cfg.DATASET.NUM_POINTS, split='valid', transform=None)
val_dataloader = DataLoader(val_dataset, batch_size=cfg.OPTIMIZER.BATCH_SIZE, shuffle=False, drop_last=False)

# Build Network and Optimizer
net = build_network(cfg)
if args.pretrained_ckpt is not None:
    pretrained_state_dict = torch.load(args.pretrained_ckpt)['model_state_dict']
    
    for k, v in net.state_dict().items():
        if (v.shape != pretrained_state_dict[k].shape):
            del pretrained_state_dict[k]

    net.load_state_dict(pretrained_state_dict, strict = False)
    
net = net.cuda()
opt, scheduler = build_optimizer(cfg, net.parameters(), len(train_dataloader))


from torch.utils.tensorboard import SummaryWriter
ckpt_dir = cfg.ROOT_DIR / 'experiments' / cfg.DATASET.NAME / args.exp_name / 'ckpt'
tensorboard_dir = cfg.ROOT_DIR / 'experiments' / cfg.DATASET.NAME / args.exp_name / 'tensorboard'
log_dir = cfg.ROOT_DIR / 'experiments' / cfg.DATASET.NAME / args.exp_name / 'logs'

os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(tensorboard_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


writer = SummaryWriter(tensorboard_dir)

min_loss = 1e20
max_acc = 0





steps_cnt = 0
epoch_cnt = 0
best_acc = 0
best_class_avg_iou = 0
best_inctance_avg_iou = 0
best_general_miou = 0
import logging
logger = logging.getLogger("Model")
logger.setLevel(logging.INFO) 
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.data_name))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

if(args.data_name == "coarse"):
    metadata = json.load(open('metadata/coarse_seg_classes.json'))        
else:
    metadata = json.load(open('metadata/fine_seg_classes.json'))
        
num_classes = metadata["num_classes"]
num_part    = metadata["num_part"]
seg_classes = metadata["seg_classes"]

def log_string(str):
    logger.info(str)
    print(str)
        

for epoch in tqdm(range(1, cfg.OPTIMIZER.MAX_EPOCH + 1)):
    opt.zero_grad()
    net.zero_grad()
    net.train()
    loss = 0
    count  = 0.0
    accuracy = []
    for (points, label, target, shape_id) in tqdm(train_dataloader, total=len(train_dataloader), smoothing=0.9):
        points = points.data.numpy()
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        points = torch.Tensor(points)
        points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            
        data_dic = {
            'points'    : points,
            'seg_id'    : target,
            'cls_tokens': label,
            'cls_id': label,
        }

        data_dic = net(data_dic) 
        loss, loss_dict = net.get_loss(data_dic, smoothing = True, is_segmentation = cfg.DATASET.IS_SEGMENTATION)
        loss = loss
        loss.backward()
        steps_cnt += 1
        
        # if (steps_cnt)%(cfg.OPTIMIZER.GRAD_ACCUMULATION) == 0:
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.OPTIMIZER.GRAD_CLIP)
        opt.step()
        opt.zero_grad()
        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        writer.add_scalar('steps/loss', loss, steps_cnt)
        writer.add_scalar('steps/lr', lr, steps_cnt)
        if cfg.DATASET.IS_SEGMENTATION:
            seg_pred = data_dic['pred_score_logits'].contiguous().view(-1, num_part) 
            target = data_dic['seg_id'].view(-1, 1)[:, 0]
            # Loss

            pred_choice = seg_pred.data.max(1)[1]  # b*n
            correct = pred_choice.eq(target.data).sum()  # torch.int64: total number of correct-predict pts

            count += data_dic['pred_score_logits'].shape[0]  # count the total number of samples in each iteration
            accuracy.append(correct.item() / (data_dic['pred_score_logits'].shape[0] * data_dic['points'].shape[1]))  # append the accuracy of each iteration
        else:
            pred_choice = data_dic['pred_score_logits'].data.max(1)[1]
            correct = pred_choice.eq(label.long().data).cpu().sum()
            accuracy.append(correct.item() / float(points.size()[0]))
        for k,v in loss_dict.items():
            writer.add_scalar('steps/loss_' + k, v, steps_cnt)
    train_instance_acc = np.mean(accuracy)
    log_string('Train accuracy is: %.5f' % train_instance_acc) 

    if (epoch % args.val_steps) == 0:
        val_dict = validate(net, val_dataloader, net.get_loss, 'cuda', args.data_name, is_segmentation = cfg.DATASET.IS_SEGMENTATION)
        
        log_string('='*20 + 'Epoch ' + str(epoch+1)+ '='*20)

        if cfg.DATASET.IS_SEGMENTATION:
            writer.add_scalar('epochs/val_miou', val_dict['miou'], epoch_cnt)
            log_string('Val mIoU: ' + str(val_dict['miou']))
            writer.add_scalar('epochs/val_class_avg_miou', val_dict['class_avg_miou'], epoch_cnt)
            log_string('Val class_avg_miou: '+ str(val_dict['class_avg_miou']))
            writer.add_scalar('epochs/val_general_miou', val_dict['general_miou'], epoch_cnt)
            log_string('Val general_miou: '+ str(val_dict['general_miou']))
            writer.add_scalar('epochs/val_accuracy', val_dict['accuracy'], epoch_cnt)
            log_string('Val accuracy: '+ str(val_dict['accuracy']))

            if val_dict['accuracy'] > best_acc:
                best_acc = val_dict['accuracy']
            if val_dict['class_avg_miou'] > best_class_avg_iou:
                best_class_avg_iou = val_dict['class_avg_miou']
            if val_dict['miou'] > best_inctance_avg_iou:
                best_inctance_avg_iou = val_dict['miou']
            if val_dict['general_miou'] > best_general_miou:
                best_general_miou = val_dict['general_miou']
            log_string('Best accuracy is: %.4f' % best_acc)
            log_string('Best class avg mIOU is: %.4f' % best_class_avg_iou)
            log_string('Best inctance avg mIOU is: %.4f' % best_inctance_avg_iou)
            log_string('Best best_general_miou is: %.4f' % best_general_miou)
 
        else:
            writer.add_scalar('epochs/val_loss', val_dict['loss'], epoch_cnt)
            writer.add_scalar('epochs/val_acc', val_dict['acc'], epoch_cnt)
            writer.add_scalar('epochs/val_acc_avg', val_dict['acc_avg'], epoch_cnt)
            log_string('Val Loss: '+ str(val_dict['loss'])+ '\nVal Accuracy: '+ str(val_dict['acc'])+ '\nVal Avg Accuracy: '+ str(val_dict['acc_avg']))

            if val_dict['acc'] > best_acc:
                best_acc = val_dict['acc']
            log_string('Best accuracy is: %.4f' % best_acc)
            
            for k,v in val_dict['loss_dic'].items():
                writer.add_scalar('epochs/val_loss_'+ k, v, epoch_cnt)

        epoch_cnt += 1

        
        if cfg.DATASET.IS_SEGMENTATION:
            if val_dict['miou'] > max_acc:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    }, ckpt_dir / 'ckpt-best.pth')
                
                max_acc = val_dict['miou']
        else:

            if val_dict['acc'] > max_acc:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'loss': val_dict['loss'],
                    }, ckpt_dir / 'ckpt-best.pth')
                
                max_acc = val_dict['acc']

    torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                }, ckpt_dir / 'ckpt-last.pth')
