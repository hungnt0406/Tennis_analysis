import torch
import torch.nn as nn
import torch.nn.functional as F

class TargetClassifier(nn.Module):
    def __init__(self, in_channels=1024):
        super(TargetClassifier, self).__init__()
        # Several 3x3 Convs to process high-level features into a heatmap
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)

def extract_candidates(heatmap, top_k=10, pool_size=3):
    """
    Extract top_k local maxima from the heatmap using MaxPool2d as NMS.
    """
    B, C, H, W = heatmap.shape
    pad = pool_size // 2
    # NMS via MaxPool
    hmax = F.max_pool2d(heatmap, pool_size, stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    heatmap = heatmap * keep
    
    heatmap = heatmap.view(B, -1) # [B, H*W]
    top_scores, top_indices = torch.topk(heatmap, top_k, dim=1)
    
    top_y = torch.div(top_indices, W, rounding_mode='floor')
    top_x = top_indices % W
    
    candidates = torch.stack([top_x, top_y], dim=-1).float() # [B, top_k, 2]
    return candidates, top_scores
