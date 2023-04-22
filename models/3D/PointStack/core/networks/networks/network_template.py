'''
* author: Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: kevin.tirta@kaist.ac.kr
'''

from .. import heads, encoders

import torch
import torch.nn as nn
import torch.nn.functional as F

class NetworkTemplate(nn.Module):
    def __init__(self, cfg, topology = None):
        super().__init__()
        self.cfg = cfg
        self.num_class = cfg.DATASET.NUM_CLASS
        
        if topology == None:
            self.module_topology = ['encoder',  'head']
        else:
            self.module_topology = topology

        self.module_list = self.build_networks()
        
    def build_networks(self):
        model_info_dict = {
            'module_list': [],
        }
        for module_name in self.module_topology:
            modules, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            

        return nn.ModuleList(model_info_dict['module_list'])

    def build_encoder(self, model_info_dict):
        if self.cfg.NETWORK.get('ENCODER', None) is None:
            return None, model_info_dict
        encoder_modules = []
        encoder_module = encoders.__all__[self.cfg.NETWORK.ENCODER.NAME](cfg=self.cfg)
        model_info_dict['module_list'].append(encoder_module)
        encoder_modules.append(encoder_module)
        return nn.ModuleList(encoder_modules), model_info_dict
    
    
    def build_head(self, model_info_dict):
        if self.cfg.NETWORK.get('HEAD', None) is None:
            return None, model_info_dict

        head_modules = []
        if self.cfg.NETWORK.HEAD.get('CLASSIFIER', None) is None:
            classifier_module = None
        else:
            classifier_module = heads.__all__[self.cfg.NETWORK.HEAD.CLASSIFIER.NAME](cfg=self.cfg)
            model_info_dict['module_list'].append(classifier_module)
            head_modules.append(classifier_module)

        if self.cfg.NETWORK.HEAD.get('SEGMENTATOR', None) is None:
            segmentator_module = None
        else:
            segmentator_module = heads.__all__[self.cfg.NETWORK.HEAD.SEGMENTATOR.NAME](cfg=self.cfg)
            model_info_dict['module_list'].append(segmentator_module)
            head_modules.append(segmentator_module)

        return nn.ModuleList(head_modules), model_info_dict

    def forward(self, data_dic):
        for curr_module in self.module_list:
            data_dic = curr_module(data_dic)
        
        return data_dic