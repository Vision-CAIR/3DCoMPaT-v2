import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, cfg):
        super(LinearClassifier, self).__init__()
        self.cfg = cfg
        self.in_channels = self.cfg.NETWORK.HEAD.CLASSIFIER.IN_CHANNELS + self.cfg.NETWORK.HEAD.CLASSIFIER.DIMS[0:-1]
        self.out_channels = self.cfg.NETWORK.HEAD.CLASSIFIER.DIMS

        self.classifier = []
        for c_in, c_out in zip(self.in_channels, self.out_channels):
            self.classifier.extend(nn.Sequential(
                nn.Linear(c_in, c_out),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
                nn.Dropout(0.5)
            ))

        self.classifier.extend(nn.Sequential(
            nn.Linear(self.cfg.NETWORK.HEAD.CLASSIFIER.DIMS[-1], self.cfg.DATASET.NUM_CLASS)
        ))

        self.classifier = nn.ModuleList(self.classifier)
    
    def forward(self, data_dic):
        logits = data_dic['point_features']
        
        for cur_module in self.classifier:
            logits = cur_module(logits)
        
        data_dic['pred_score_logits'] = logits
        return data_dic