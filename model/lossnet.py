import torch
import torch.nn as nn 
import torch.nn.functional as F 


# LossNet to learn the classification loss with feature maps
class LossNet(nn.Module):
    def __init__(self, feature_sizes, num_channels, interm_dim=128):
        super(LossNet, self).__init__()
        self.feature_sizes = (56, 28, 14, 7) if not feature_sizes else feature_sizes
        self.num_channels = (64, 128, 256, 512) if not num_channels else num_channels
        self.num_feature_maps = len(feature_sizes)
        self.feat_gap_group = [nn.AvgPool2d(self.feature_sizes[i]) for i in range(self.num_feature_maps)]
        self.feat_fc_group = [nn.Linear(self.num_channels[i], interm_dim) for i in range(self.num_feature_maps)]
        self.linear = nn.Linear(self.num_feature_maps * interm_dim, 1)

    def get_feature_vector(self, stage, feat_map):
        x = self.feat_gap_group[stage](feat_map)
        x = x.view(x.size(0), -1)
        x = F.relu(self.feat_fc_group[stage](x))
        return x

    def forward(self, features):
        feat_vector = torch.cat([self.get_feature_vector(i, features[i]) for i in range(self.num_feature_maps)], 1)
        print(feat_vector.shape)
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