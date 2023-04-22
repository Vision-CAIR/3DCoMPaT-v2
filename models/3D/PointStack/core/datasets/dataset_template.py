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
import torch.utils.data as data


shapenetpart_cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
shapenetpart_seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
shapenetpart_seg_start_index = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]


class DatasetTemplate(data.Dataset):
    def __init__(self, cfg, class_choice=None, split='train', load_name=False, load_file=False, random_rotate=False, random_jitter=False, 
            random_translate=False):
        assert cfg.DATASET.NUM_POINTS <= 2048        
        if cfg.DATASET.USE_RANDOM_SHUFFLE: 
            print('using random shuffle per point set')

        self.cfg = cfg
        self.data_dir = os.path.join(cfg.ROOT_DIR, 'data', cfg.DATASET.NAME.lower())
        print('Loading dataset: ', self.data_dir)
        self.class_choice = class_choice
        self.split = split
        self.load_name = load_name
        self.load_file = load_file

        self.dataset_name = cfg.DATASET.NAME
        self.num_points = cfg.DATASET.NUM_POINTS
        self.segmentation = cfg.DATASET.IS_SEGMENTATION
        self.random_rotate = cfg.DATASET.USE_AUG_ROT if self.split in ['train'] else False
        self.random_jitter = cfg.DATASET.USE_AUG_JIT if self.split in ['train'] else False
        self.random_translate = cfg.DATASET.USE_AUG_TRANS if self.split in ['train'] else False

    def get_path(self, type):
        path_h5py = os.path.join(self.data_dir, '*%s*.h5'%type)
        self.path_h5py_all += glob(path_h5py)
        if self.load_name:
            path_json = os.path.join(self.data_dir, '%s*_id2name.json'%type)
            self.path_name_all += glob(path_json)
        if self.load_file:
            path_json = os.path.join(self.data_dir, '%s*_id2file.json'%type)
            self.path_file_all += glob(path_json)
        return 

    def load_h5py(self, path):
        all_data = []
        all_label = []
        all_seg = []
        for h5_name in path:
            f = h5py.File(h5_name, 'r+')
                
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            if self.segmentation:
                if 'mask' in f.keys():
                    seg = f['mask'][:].astype('int64')
                    seg[np.where(seg > -1)] = 1 # Foreground
                    seg[np.where(seg == -1)] = 0 # Background
                else:
                    seg = f['seg'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
            if self.segmentation:
                all_seg.append(seg)
        return all_data, all_label, all_seg

    def load_json(self, path):
        all_data = []
        for json_name in path:
            j =  open(json_name, 'r+')
            data = json.load(j)
            all_data += data
        return all_data

    def __getitem__(self, item):
        point_set = self.data[item][:self.num_points]
        # visualize_numpy(point_set)
        label = self.label[item]
        if self.segmentation:
            seg = self.seg[item][:self.num_points]    

        if self.random_rotate:
            point_set, theta = self.rotate_pointcloud(point_set)
        if self.random_jitter:
            point_set, sigma, clip = self.jitter_pointcloud(point_set)
        if self.random_translate:
            point_set, xyz1, xyz2 = self.translate_pointcloud(point_set)
        
        if self.split == 'train':
            if self.cfg.DATASET.USE_RANDOM_SHUFFLE:
                p = np.random.permutation(len(point_set))
                # np.random.shuffle(point_set)
                point_set = point_set[p]
                    
                if self.segmentation:
                    assert len(point_set) == len(seg)
                    seg = seg[p]
                    
        # convert numpy array to pytorch Tensor
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

    def __len__(self):
        return self.data.shape[0]

    def __repr__(self):
        return ('Name=%s, Class Choice=%s, Num Points=%s, Split=%s, Random Rotate=%s, Random Jitter=%s, Random Translate=%s, )' 
                % (repr(self.dataset_name), repr(self.class_choice), repr(self.num_points),
                repr(self.split), repr(self.random_rotate), repr(self.random_jitter), repr(self.random_translate)))

    def translate_pointcloud(self, pointcloud, xyz1 = None, xyz2 = None):
        if xyz1 == None:
            xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
        if xyz2 == None:
            xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        
        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
        return translated_pointcloud, xyz1, xyz2


    def jitter_pointcloud(self, pointcloud, sigma=0.01, clip=0.02):
        N, C = pointcloud.shape
        pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
        return pointcloud, sigma, clip


    def rotate_pointcloud(self, pointcloud, theta = None):
        if theta == None:
            theta = np.pi*2 * np.random.rand()
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
        return pointcloud, theta