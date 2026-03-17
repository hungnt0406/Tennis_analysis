import torch
import torch.nn as nn
import torch.nn.functional as F

class ObjectAssociation(nn.Module):
    def __init__(self, embed_dim=256):
        super(ObjectAssociation, self).__init__()
        self.w_low = nn.Parameter(torch.tensor(0.0)) # Starts at logit 0 -> sigmoid 0.5

    def compute_association_matrix(self, embed_t_minus_1, embed_t):
        """
        embed: [B, N, embed_dim]
        Returns: [B, N, N] association matrix
        """
        # Candidate correlation between frames
        matrix = torch.bmm(embed_t_minus_1, embed_t.transpose(1, 2))
        return matrix

    def forward(self, low_t1, low_t, high_t1, high_t):
        # Calculate parallel associations from both branches
        A_low = self.compute_association_matrix(low_t1, low_t)
        A_high = self.compute_association_matrix(high_t1, high_t)
        
        # Weighted Fusion 
        w = torch.sigmoid(self.w_low)
        A = w * A_low + (1 - w) * A_high
        A = F.softmax(A, dim=-1) # Softmax over candidates in current frame t
        
        return A
