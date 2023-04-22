"""
Custom metrics for 2D segmentation models.
"""
import numpy as np


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


def compute_overall_precision(y_predicted, y_truth):
    """
    Compute overall precision given prediction and target numpy arrays.
    """
    precisions = []
    for i in range(y_predicted.shape[0]):
        true_positives = np.sum(np.logical_and(y_predicted[i], y_truth[i]))
        false_positives = np.sum(
            np.logical_and(y_predicted[i], np.logical_not(y_truth[i]))
        )

        precision = true_positives / (true_positives + false_positives)
        precisions.append(precision)
    return precisions
