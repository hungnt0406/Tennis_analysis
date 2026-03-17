import torch
import torch.nn as nn
import torch.nn.functional as F

def focal_loss(pred, target, alpha=2, beta=4):
    """
    pred: [B, C, H, W]
    target: [B, C, H, W] Gaussian heatmap
    """
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    
    neg_weights = torch.pow(1 - target, beta)
    
    pred = torch.clamp(pred, 1e-4, 1 - 1e-4)
    
    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds
    
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    
    if num_pos == 0:
        return -neg_loss
    return -(pos_loss + neg_loss) / num_pos


def association_loss(A, current_gt_index, prev_gt_index):
    """
    Cross-entropy equivalent penalizing incorrect matches
    A: [B, N, N] - Association scores between frame t-1 and frame t
    current_gt_index: [B] - Index of true candidate in frame t
    prev_gt_index: [B] - Index of true candidate in frame t-1
    """
    B = A.shape[0]
    loss = 0.0
    for b in range(B):
        true_t1 = prev_gt_index[b]
        true_t = current_gt_index[b]
        
        if true_t1 < A.shape[1] and true_t < A.shape[2]:
            score = A[b, true_t1, true_t]
            loss += -torch.log(score + 1e-6)
    return loss / B
