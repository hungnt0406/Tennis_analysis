import torch
import torch.nn as nn
from torchvision.ops import roi_align

class FeatureEncoding(nn.Module):
    def __init__(self, feat_stride, roi_size=5):
        super(FeatureEncoding, self).__init__()
        self.feat_stride = feat_stride
        self.roi_size = roi_size

    def forward(self, feature_map, candidates):
        """
        feature_map: [B, C, H, W]
        candidates: [B, N, 2] (x, y) coordinates relative to input image scale. 
        """
        B, N, _ = candidates.shape
        C = feature_map.shape[1]
        
        # Prepare rois for roi_align: [batch_index, x1, y1, x2, y2]
        box_size = 4.0 # pixels to crop in feature space
        
        rois = []
        for b in range(B):
            for n in range(N):
                cx, cy = candidates[b, n, 0], candidates[b, n, 1]
                x1, y1 = cx - box_size/2, cy - box_size/2
                x2, y2 = cx + box_size/2, cy + box_size/2
                rois.append([b, x1, y1, x2, y2])
                
        rois = torch.tensor(rois, dtype=torch.float32, device=feature_map.device)
        
        # Apply RoI Align
        aligned_features = roi_align(
            feature_map, 
            rois, 
            output_size=(self.roi_size, self.roi_size), 
            spatial_scale=1.0  # Assumes candidates passed match the scale of the feature map
        )
        
        aligned_features = aligned_features.view(B, N, C, self.roi_size, self.roi_size)
        
        # flatten the spatial dimensions of RoI
        return aligned_features.view(B, N, -1)
