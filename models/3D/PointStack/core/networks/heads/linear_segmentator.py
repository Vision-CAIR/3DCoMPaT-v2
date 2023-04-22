import torch
import torch.nn as nn

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda(non_blocking=True)
    return new_y

class LinearSegmentator(nn.Module):
    def __init__(self, cfg):
        super(LinearSegmentator, self).__init__()
        self.cfg = cfg
        self.in_channels = self.cfg.NETWORK.HEAD.SEGMENTATOR.IN_CHANNELS + self.cfg.NETWORK.HEAD.SEGMENTATOR.DIMS[0:-1]
        self.out_channels = self.cfg.NETWORK.HEAD.SEGMENTATOR.DIMS

        self.segmentator = []
        for c_in, c_out in zip(self.in_channels, self.out_channels):
            self.segmentator.extend(nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=1),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
                nn.Dropout(0.5)
            ))

        self.segmentator.extend(nn.Sequential(
            nn.Conv1d(self.cfg.NETWORK.HEAD.SEGMENTATOR.DIMS[-1], self.cfg.DATASET.NUM_CLASS, kernel_size=1)
        ))

        self.segmentator = nn.ModuleList(self.segmentator)

        self.cls_map = nn.Sequential(
            nn.Conv1d(in_channels=42, out_channels=64, kernel_size=1, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
    
    def forward(self, data_dic):
        cls_tokens = data_dic['cls_tokens'] # (B, 1, 1)
        cls_tokens = to_categorical(cls_tokens, 42).unsqueeze(1) # (B, 1, 16)
        cls_tokens = self.cls_map(cls_tokens.permute(0, 2, 1)) # B, 1, 64

        logits = data_dic['point_features']
        logits = torch.cat([logits, cls_tokens.repeat(1, 1, logits.shape[-1])], dim = 1)

        for cur_module in self.segmentator:
            logits = cur_module(logits)
        data_dic['pred_score_logits'] = logits.permute(0,2,1)
        return data_dic
