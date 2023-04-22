"""
Utility functions for shape classification.
"""
import os
import random

import numpy as np
import torch

cudnn_deterministic = True


def seed_everything(seed=0):
    """
    Fix all random seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic


def compute_topk_acc(pred, targets, topk):
    """
    Compute top-k accuracy given prediction and target vectors.

    Args:
    ----
    pred:       Network prediction
    targets:    Ground truth labels
    topk:       k value
    """
    topk = min(topk, pred.shape[1])
    _, pred = pred.topk(topk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    hits_tag = correct[:topk].reshape(-1).float().sum(0)

    return hits_tag


def calculate_metrics(outputs, targets):
    """
    Compute top-1 and top-5 accuracy metrics.

    Args:
    ----
    outputs:    Network outputs list
    targets:    Ground truth labels
    """
    pred = outputs

    # Top-k prediction for TAg
    hits_tag_top5 = compute_topk_acc(pred, targets, 5)
    hits_tag_top1 = compute_topk_acc(pred, targets, 1)

    return hits_tag_top5.item(), hits_tag_top1.item()
