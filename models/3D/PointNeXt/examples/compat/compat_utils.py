import argparse
import pdb
import torch.nn.functional as F
import torch
import numpy as np


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


def compute_overall_iou(pred_np, target_np, num_classes):
        shape_ious = []
        for shape_idx in range(pred_np.shape[0]):   # sample_idx
            part_ious = []
            for part in range(num_classes): 
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))

                F = np.sum(target_np[shape_idx] == part)

                if F != 0:
                    iou = I / float(U)    
                    part_ious.append(iou)   
            shape_ious.append(np.mean(part_ious))  
        return shape_ious

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