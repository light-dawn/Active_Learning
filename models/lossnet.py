# encoding=utf-8

import torch
import torch.nn as nn 
import torch.nn.functional as F 


# LossNet to learn the classification loss with feature maps
class LossNet(nn.Module):
    def __init__(self, feature_sizes, num_channels, interm_dim=128):
        super(LossNet, self).__init__()
        self.num_feature_maps = len(feature_sizes)
        # Must define the list as torch.nn.ModuleList, otherwise the list cannot be moved to cuda.
        self.feat_gap_group = nn.ModuleList([nn.AvgPool2d(feature_sizes[i]) for i in range(self.num_feature_maps)])
        self.feat_fc_group = nn.ModuleList([nn.Linear(num_channels[i], interm_dim) for i in range(self.num_feature_maps)])
        self.linear = nn.Linear(self.num_feature_maps * interm_dim, 1)

    def get_feature_vector(self, stage, feat_map):
        x = self.feat_gap_group[stage](feat_map)
        x = x.view(x.size(0), -1)
        x = F.relu(self.feat_fc_group[stage](x))
        return x

    def forward(self, features):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # feat_vector = torch.tensor([], device=device)
        # for stage in range(self.num_feature_maps):
        #     x = self.feat_gap_group[stage](features[stage])
        #     x = x.view(x.size(0), -1)
        #     x = self.feat_fc_group[stage](x)
        #     x = F.relu(x)
        #     feat_vector = torch.cat([feat_vector, x])
        feat_vector = torch.cat([self.get_feature_vector(i, features[i]) for i in range(self.num_feature_maps)], 1)
        out = self.linear(feat_vector)
        return out

if __name__ == "__main__":
    lossnet = LossNet(feature_sizes=(56, 28, 14, 7), num_channels=(64, 128, 256, 512))
    feat = [
        torch.zeros(size=(12, 64, 56, 56)),
        torch.zeros(size=(12, 128, 28, 28)),
        torch.zeros(size=(12, 256, 14, 14)),
        torch.zeros(size=(12, 512, 7, 7))
    ]
    result = lossnet(feat)
    print("Output size: ", result.size())