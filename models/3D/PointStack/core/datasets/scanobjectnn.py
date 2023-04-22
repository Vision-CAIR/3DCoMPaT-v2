#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Based on codes originally written by An Tao (ta19@mails.tsinghua.edu.cn)
"""

import os
import torch
import json
import h5py
from glob import glob
import numpy as np

from .dataset_template import DatasetTemplate

CLS_NAME_TO_ID_DIC = {'bag': 0, 'bin': 1, 'box': 2, 'cabinet': 3, 'chair': 4, 'desk': 5, 'display': 6, 'door': 7, 'shelf': 8, 'table': 9, 'bed': 10, 'pillow': 11, 'sink': 12, 'sofa': 13, 'toilet': 14}
CLS_ID_TO_NAME_DIC = {0: 'bag', 1: 'bin', 2: 'box', 3: 'cabinet', 4: 'chair', 5: 'desk', 6: 'display', 7: 'door', 8: 'shelf', 9: 'table', 10: 'bed', 11: 'pillow', 12: 'sink', 13: 'sofa', 14: 'toilet'}
MODELNET40_LIMITER = [246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246]

class ScanObjectNN(DatasetTemplate):
    def __init__(self, cfg, class_choice=None, split='train', load_name=True, load_file=True, random_rotate=False, random_jitter=False, random_translate=False):
        super().__init__(cfg = cfg, class_choice=None, split=split, load_name=True, load_file=True, random_rotate=False, random_jitter=False, random_translate=False)

        self.path_h5py_all = []
        self.path_h5py_nobg = []

        self.path_name_all = []
        
        self.path_file_all = []
        self.path_file_nobg = []

        if self.cfg.DATASET.DIRSPLIT == 'main':
            self.dirsplit_all = ['main_split']
            self.dirsplit_nobg = ['main_split_nobg']
        elif self.cfg.DATASET.DIRSPLIT == 'supplementary':
            self.dirsplit_all = ['split1', 'split2', 'split3', 'split4']
            self.dirsplit_nobg = ['split1_nobg', 'split2_nobg', 'split3_nobg', 'split4_nobg']
        else:
            self.dirsplit_all = ['main_split', 'split1', 'split2', 'split3', 'split4']
            self.dirsplit_nobg = ['main_split_nobg', 'split1_nobg', 'split2_nobg', 'split3_nobg', 'split4_nobg']

        if self.cfg.DATASET.AUGSPLIT == 'OBJ_ONLY':
            self.augsplit = 'objectdataset'
        elif self.cfg.DATASET.AUGSPLIT == 'PB_T25':
            self.augsplit = 'objectdataset_augmented25_norot'
        elif self.cfg.DATASET.AUGSPLIT == 'PB_T25_R':
            self.augsplit = 'objectdataset_augmented25rot'
        elif self.cfg.DATASET.AUGSPLIT == 'PB_T50_R':
            self.augsplit = 'objectdataset_augmentedrot'
        elif self.cfg.DATASET.AUGSPLIT == 'PB_T50_RS':
            self.augsplit = 'objectdataset_augmentedrot_scale75'
        
        if self.split in ['train','trainval','all']:   
            self.get_path('training', self.dirsplit_all, self.augsplit)
        if self.split in ['val', 'test', 'all']:
            self.get_path('test', self.dirsplit_all, self.augsplit)

        self.path_h5py_all.sort()
        self.path_h5py_nobg.sort()


        data, label, seg = self.load_h5py(self.path_h5py_all)

        if self.load_name or self.class_choice != None:
            self.path_name_all.sort()
            self.name = np.array(self.load_json(self.path_name_all))    # load label name

        if self.load_file:
            self.path_file_all.sort()
            self.file = np.array(self.load_json(self.path_file_all))    # load file name
        
        self.data = np.concatenate(data, axis=0)

        self.label = np.concatenate(label, axis=0) 

        if self.segmentation:
            self.seg = np.concatenate(seg, axis=0) 

        if self.class_choice != None:
            indices = (self.name == class_choice)
            self.data = self.data[indices]
            self.label = self.label[indices]

            self.name = self.name[indices]
            if self.load_file:
                self.file = self.file[indices]
        
        if (cfg.DATASET.LIMIT_TO_MODELNET40) and (split == 'train'):
            chosen_data = []
            chosen_label = []

            for cls_id in range(15):
                indices = np.where(self.label == cls_id)
                label = self.label[indices]
                data = self.data[indices]

                chosen_data.extend(data[:MODELNET40_LIMITER[cls_id]])
                chosen_label.extend(label[:MODELNET40_LIMITER[cls_id]])

            self.data = np.array(chosen_data)
            self.label = np.array(chosen_label)


    def get_path(self, type, dirsplits, augsplit):
        for dirsplit in dirsplits:
            path_h5py = os.path.join(self.data_dir, dirsplit, type+'*'+augsplit+'.h5')
            path_h5py_nobg = os.path.join(self.data_dir, dirsplit + '_nobg', type+'*'+augsplit+'.h5')

            self.path_h5py_all += glob(path_h5py)
            self.path_h5py_nobg += glob(path_h5py_nobg)

        return 
    
    def __getitem__(self, item):
        point_set = self.data[item][:self.num_points]
        label = self.label[item]

        if self.segmentation:
            seg = self.seg[item][:self.num_points]

        pose_target = {
            'theta': 0,
            'dx': 0,
            'dy': 0,
            'dz': 0
        }

        if self.random_rotate:
            point_set, theta = self.rotate_pointcloud(point_set)
            pose_target['theta'] = theta

        if self.random_jitter:
            point_set, sigma, clip = self.jitter_pointcloud(point_set)

        if self.random_translate:
            point_set, xyz1, xyz2 = self.translate_pointcloud(point_set)
            pose_target['dx'], pose_target['dy'], pose_target['dz'] = xyz2

        if self.split == 'train':
            if self.cfg.DATASET.USE_RANDOM_SHUFFLE:
                p = np.random.permutation(len(point_set))
                point_set = point_set[p]
                    
                if self.segmentation:
                    assert len(point_set) == len(seg)
                    seg = seg[p]        
        point_set = torch.from_numpy(point_set)
        
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        label = label.squeeze(0)
        
        data_dic = {
            'points': point_set.cuda(),
            'cls_id': label.cuda(),
        }

        if self.segmentation:
            seg = torch.from_numpy(seg).cuda()
            data_dic['seg_id'] = seg
            
        return data_dic