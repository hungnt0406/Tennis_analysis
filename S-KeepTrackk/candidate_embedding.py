import torch
import torch.nn as nn

class CandidateEmbedding(nn.Module):
    def __init__(self, in_dim, embed_dim=256, num_heads=4, num_layers=2):
        super(CandidateEmbedding, self).__init__()
        # Project RoI features down to embed length
        self.proj = nn.Linear(in_dim, embed_dim)
        
        # Self-attention module to learn spatio-temporal relationships between candidates
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim*2, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        x: [B, N, in_dim]
        """
        x = self.proj(x)
        x = self.transformer(x)
        return x
