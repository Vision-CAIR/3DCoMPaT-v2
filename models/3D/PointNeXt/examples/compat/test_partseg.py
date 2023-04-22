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
import json
from compat_utils import *
from pathlib import Path
from tqdm import tqdm
from compat_loader import CompatLoader3D as CompatSeg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    parser.add_argument('--model', type=str, default='pointnet2_part_seg_ssg', help='model name')
    parser.add_argument('--npoint', type=int, required=True, default=1024, help='point Number')
    parser.add_argument('--shape_prior', action='store_true', default=False, help='use shape prior')
    parser.add_argument('--data_name', type=str, default="fine", help ="data_name")

    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)
    
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/part_seg/' + args.log_dir
    
    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    root = os.path.join(os.getcwd(), 'data/'+args.data_name+'_grained/')
    TEST_DATASET = CompatSeg(data_root=root, num_points=args.npoint, split='test', transform=None)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    if(args.data_name == "coarse"):
        metadata = json.load(open('metadata/coarse_seg_classes.json'))        
    else:
        metadata = json.load(open('metadata/fine_seg_classes.json'))
        
    num_classes = metadata["num_classes"]
    num_part    = metadata["num_part"]
    seg_classes = metadata["seg_classes"] 
    
    
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    '''MODEL LOADING'''
    

    shape_prior = args.shape_prior
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    # pdb.set_trace()
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, shape_prior=shape_prior, normal_channel=args.normal).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
        general_miou = []
        general_miou_method = []

        classifier = classifier.eval()

        for batch_id, (points, label, target, shape_id) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            seg_pred, _ = classifier(points, to_categorical(label, num_classes))
            if shape_prior:
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
            else:
                seg_pred, _ = classifier(points)
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val = np.argmax(cur_pred_val, -1)
            target = target.cpu().data.numpy()

            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            # calculate the mIoU given shape prior knowledge and without it
            miou = compute_overall_iou(cur_pred_val, target, num_part)
            general_miou = general_miou + miou
            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                shape = str(label[i].item())
                part_ious = {}
                for l in seg_classes[shape]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l] = 1.0
                    else:
                        part_ious[l] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                #Convert the dictionary to a list 
                part_ious = list(part_ious.values())
                shape_ious[shape].append(np.mean(part_ious))
        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        
        for cat in sorted(shape_ious.keys()):
            log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
        test_metrics['avg_iou_wihtout_shape'] = np.nanmean(general_miou)
        
    log_string('Best accuracy is: %.5f' % test_metrics['accuracy'])
    log_string('Best class avg mIOU is: %.5f' % test_metrics['class_avg_iou'])
    log_string('Best inctance avg mIOU is: %.5f' % test_metrics['inctance_avg_iou'])
    log_string('Best general avg mIOU is: %.5f' % test_metrics['avg_iou_wihtout_shape'])
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
