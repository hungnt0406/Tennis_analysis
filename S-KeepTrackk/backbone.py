import torch
import torch.nn as nn
import torchvision.models as models

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        # Use a pretrained ResNet50 as base
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Low-level Branch: preserve spatial features (e.g. up to layer2, feature stride 8)
        self.low_level = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2
        )
        
        # High-level Branch: layer3 (feature stride 16)
        self.high_level = resnet.layer3

    def forward(self, x):
        feat_low = self.low_level(x)
        feat_high = self.high_level(feat_low)
        return feat_low, feat_high
