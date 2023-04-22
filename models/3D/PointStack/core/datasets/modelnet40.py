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

class ModelNet40(DatasetTemplate):
    def __init__(self, cfg, class_choice=None, split='train', load_name=True, load_file=True, random_rotate=False, random_jitter=False, random_translate=False):
        super().__init__(cfg = cfg, class_choice=None, split=split, load_name=True, load_file=True, random_rotate=False, random_jitter=False, random_translate=False)
       
        self.path_h5py_all = []
        self.path_name_all = []
        self.path_file_all = []

        if self.split in ['train','trainval','all']:   
            self.get_path('train')
        if self.split in ['val', 'test', 'all']:
            self.get_path('test')

        self.path_h5py_all.sort()
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