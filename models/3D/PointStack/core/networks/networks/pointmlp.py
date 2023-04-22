from .. import heads, encoders
from .network_template import NetworkTemplate

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class PointMLP(NetworkTemplate):
    def __init__(self, cfg, topology = None):
        super().__init__(cfg, topology=topology)
        self.build_networks()
        self.cfg = cfg


    def get_loss(self, data_dic, smoothing=True, is_segmentation = False):
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''
        
        if is_segmentation:
            pred_logits = data_dic['pred_score_logits'].contiguous().view(-1, self.cfg.DATASET.NUM_CLASS) # (Batch size * Num Points, Num Classes)
            gt_cls_id = data_dic['seg_id'].contiguous().view(-1, 1).long() # (Batch size * Num Points, 1)

        else:
            pred_logits = data_dic['pred_score_logits'] # (Batch size, Num Classes)
            gt_cls_id = data_dic['cls_id'] # (Batch size, 1)

        if smoothing:
            eps = 0.2
            n_class = pred_logits.size(1)

            one_hot = torch.zeros_like(pred_logits).scatter(1, gt_cls_id.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred_logits, dim=1)
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss[torch.isfinite(loss)].mean()
        else:
            loss = F.cross_entropy(pred_logits, gt_cls_id, reduction='mean')

        loss_dict = {
            'Cls': loss.item(),
        }

        return loss, loss_dict

    def compute_overall_iou(self, pred, target, num_classes):
        shape_ious = []
        pred = pred.max(dim=2)[1]    # (batch_size, num_points)
        pred_np = pred.cpu().data.numpy()

        target_np = target.cpu().data.numpy()
        for shape_idx in range(pred.size(0)):   # sample_idx
            part_ious = []
            for part in range(num_classes):   
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))

                F = np.sum(target_np[shape_idx] == part)

                if F != 0:
                    iou = I / float(U) 
                    part_ious.append(iou)   
            shape_ious.append(np.mean(part_ious)) 
        return shape_ious   # [batch_size]