"""
Utility functions for 3D models training.
"""
import numpy as np
import torch


def compute_overall_iou(pred_np, target_np, num_classes):
    """
    Compute overall IoU given prediction and target numpy arrays.
    """
    shape_ious = []
    for shape_idx in range(pred_np.shape[0]):  # sample_idx
        part_ious = []
        for part in range(num_classes):
            intersect = np.sum(
                np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part)
            )
            union = np.sum(
                np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part)
            )

            n_preds = np.sum(target_np[shape_idx] == part)

            if n_preds != 0:
                iou = intersect / float(union)
                part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious


def inplace_relu(m):
    """
    Set all ReLU layers in a model to be inplace.
    """
    classname = m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True


def to_categorical(y, num_classes):
    """
    Convert a class vector (integers) to a one-hot vector.
    """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if y.is_cuda:
        return new_y.cuda()
    return new_y
