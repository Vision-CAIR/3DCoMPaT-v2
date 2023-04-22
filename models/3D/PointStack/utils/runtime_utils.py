import os
import yaml
import torch
import datetime
import json

import sklearn.metrics as metrics
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
from pathlib import Path

import torch.nn.functional as F

def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.safe_load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config

def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)

        merge_new_config(config=config, new_config=new_config)

    return config


cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def validate(net, testloader, criterion, device, data_name, is_segmentation = False):
    net.eval()
    num_params = get_n_params(net)
     
    if(data_name == "coarse"):
        metadata = json.load(open('metadata/coarse_seg_classes.json'))        
    else:
        metadata = json.load(open('metadata/fine_seg_classes.json'))
    num_classes = metadata["num_classes"]
    num_part    = metadata["num_part"]
    seg_classes = metadata["seg_classes"]  

    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = []
    mious = []
    accuracy = []
    shape_ious = 0.0
    count  = 0.0
    shape_ious = {cat: [] for cat in seg_classes.keys()}
    general_mious = []

    with torch.no_grad():
        for batch_idx, (points, label, target, shape_id) in enumerate(tqdm(testloader, total=len(testloader), smoothing=0.9)):
            start_time = datetime.datetime.now()

            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            data_dic = {
                'points'    : points,
                'seg_id'    : target,
                'cls_tokens': label,
                'cls_id': label,
            }

            data_dic = net(data_dic)
            time_cost.append(float((datetime.datetime.now() - start_time).total_seconds()))
            target = target.cpu().data.numpy()
            if is_segmentation:
                pred, target, label = data_dic['pred_score_logits'], data_dic['seg_id'], data_dic['cls_tokens']
                general_miou = net.compute_overall_iou(pred, target, num_part)
                general_mious = general_mious + general_miou
                pred = pred.max(dim=2)[1]    # (batch_size, num_points)
                cur_batch_size = pred.shape[0]
                cur_pred_val = pred.cpu().data.numpy()
                target = target.cpu().data.numpy()
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
                # total iou of current batch in each process:

                # prepare seg_pred and target for later calculating loss and acc:

                seg_pred = data_dic['pred_score_logits'].contiguous().view(-1, num_part) # ShapeNetPart has 50 classes

                target = data_dic['seg_id'].view(-1, 1)[:, 0]
                # Loss

                pred_choice = seg_pred.data.max(1)[1]  # b*n
                correct = pred_choice.eq(target.data).sum()  # torch.int64: total number of correct-predict pts

                count += data_dic['pred_score_logits'].shape[0]  # count the total number of samples in each iteration
                accuracy.append(correct.item() / (data_dic['pred_score_logits'].shape[0] * data_dic['points'].shape[1]))  # append the accuracy of each iteration

            else:
                preds = data_dic['pred_score_logits'].max(dim=1)[1]
                test_true.append(data_dic['cls_id'].cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
                total += data_dic['cls_id'].size(0)
                correct += preds.eq(data_dic['cls_id']).sum().item()
                loss, loss_dict = criterion(data_dic)
                test_loss += loss.item()
        
        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))

        total_samples = len(time_cost)
        time_cost = np.mean(time_cost[total_samples//5:total_samples*4//5])
        
        if is_segmentation:
            general_miou_value = np.mean(general_mious)
            overall_miou = np.mean(all_shape_ious)
            class_avg_iou = np.mean(mean_shape_ious)
            overall_acc = np.mean(accuracy)
            return {
                    "miou": float("%.4f" % (100. * overall_miou)),
                    "general_miou": float("%.4f" % (100. * general_miou_value)),
                    "class_avg_miou": float("%.4f" % (100. * class_avg_iou)),
                    "accuracy": float("%.4f" % (100. * overall_acc)),
                    "time": time_cost,
                    "num_params": num_params,
                    'peak_memory': torch.cuda.memory_stats('cuda')['allocated_bytes.all.peak']
                }
        else:
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)


            return {
                "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
                "loss_dic": loss_dict,
                "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
                "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
                "time": time_cost,
                "num_params": num_params,
                'peak_memory': torch.cuda.memory_stats('cuda')['allocated_bytes.all.peak']
            }

def validate_voting(net, testloader, device = 'cuda', num_repeat = 300, num_vote = 10):
    net.eval()
    best_acc = 0
    best_mean_acc = 0
    pointscale = PointcloudScale(scale_low=0.85, scale_high=1.15)

    for i in tqdm(range(num_repeat)):
        test_true = []
        test_pred = []

        for batch_idx, data_dic in enumerate(tqdm(testloader)):
            pred = 0
            for v in range(num_vote):
                new_data = data_dic
                if v > 0:
                    new_data['points'] = pointscale(new_data['points'])
                with torch.no_grad():
                    pred += F.softmax(net(data_dic)['pred_score_logits'], dim=1)  # sum 10 preds
            pred /= num_vote  # avg the preds!
            pred_choice = pred.max(dim=1)[1]
            test_true.append(data_dic['cls_id'].cpu().numpy())
            test_pred.append(pred_choice.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = 100. * metrics.accuracy_score(test_true, test_pred)
        test_mean_acc = 100. * metrics.balanced_accuracy_score(test_true, test_pred)
        if test_acc > best_acc:
            best_acc = test_acc
        if test_mean_acc > best_mean_acc:
            best_mean_acc = test_mean_acc
        outstr = 'Voting %d, test acc: %.3f, test mean acc: %.3f,  [current best(all_acc: %.3f mean_acc: %.3f)]' % \
                 (i, test_acc, test_mean_acc, best_acc, best_mean_acc)
        print(outstr)
    best_acc = best_acc*100
    best_mean_acc = best_mean_acc*100
    print('Final Test Acc: ', best_acc)
    print('Final Test Avg Acc: ', best_mean_acc)

    val_dict = {
        'acc': best_acc,
        'acc_avg': best_mean_acc
    }

    return val_dict

class PointcloudScale(object):  # input random scaling
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda())

        return pc
