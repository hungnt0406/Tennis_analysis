import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import local modules (assuming they are in the same directory)
from dataloader import TennisDataset
from backbone import Backbone
from candidate_extraction import TargetClassifier, extract_candidates
from feature_encoding import FeatureEncoding
from candidate_embedding import CandidateEmbedding
from association import ObjectAssociation
from loss import focal_loss, association_loss

def evaluate_checkpoints(root_dir, checkpoint_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Setup Val DataLoader
    try:
        val_dataset = TennisDataset(root_dir, mode='val', seq_len=3)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    except Exception as e:
        print(f"Error initializing dataloader: {e}")
        return

    # 2. Initialize Model (Architecture must match exactly)
    backbone = Backbone().to(device)
    target_classifier = TargetClassifier(in_channels=1024).to(device)
    feat_encode_low = FeatureEncoding(feat_stride=8, roi_size=5).to(device)
    feat_encode_high = FeatureEncoding(feat_stride=16, roi_size=5).to(device)
    embed_low = CandidateEmbedding(in_dim=512*5*5, embed_dim=256).to(device)
    embed_high = CandidateEmbedding(in_dim=1024*5*5, embed_dim=256).to(device)
    association = ObjectAssociation(embed_dim=256).to(device)

    # 3. Find and sort checkpoints
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    # Sort by epoch number to see the curve properly
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if 'epoch' in x else 0)

    results = []

    print(f"\nStarting evaluation of {len(checkpoints)} checkpoints...")
    
    for ckpt_name in checkpoints:
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        
        # Load Weights
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        backbone.load_state_dict(checkpoint['backbone'])
        target_classifier.load_state_dict(checkpoint['target_classifier'])
        feat_encode_low.load_state_dict(checkpoint['feat_encode_low'])
        feat_encode_high.load_state_dict(checkpoint['feat_encode_high'])
        embed_low.load_state_dict(checkpoint['embed_low'])
        embed_high.load_state_dict(checkpoint['embed_high'])
        association.load_state_dict(checkpoint['association'])
        
        # Evaluate
        backbone.eval()
        target_classifier.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                heatmaps = batch['heatmaps'].to(device)
                centers = batch['centers'].to(device)
                B, T, C, H, W = images.shape
                
                v_loss = 0
                prev_low, prev_high = None, None
                
                for t in range(T):
                    feat_low, feat_high = backbone(images[:, t])
                    pred_heatmap = target_classifier(feat_high)
                    
                    gt_hm = torch.nn.functional.interpolate(heatmaps[:, t], size=pred_heatmap.shape[-2:], mode='bilinear', align_corners=False)
                    loss_hm = focal_loss(pred_heatmap, gt_hm)
                    
                    candidates, _ = extract_candidates(pred_heatmap, top_k=10)
                    candidates[:, 0, :] = centers[:, t] / 16.0
                    
                    roi_low = feat_encode_low(feat_low, candidates * 2.0)
                    roi_high = feat_encode_high(feat_high, candidates)
                    emb_low, emb_high = embed_low(roi_low), embed_high(roi_high)
                    
                    if t > 0 and prev_low is not None:
                        A = association(prev_low, emb_low, prev_high, emb_high)
                        v_loss += loss_hm + association_loss(A, torch.zeros(B, dtype=torch.long).to(device), torch.zeros(B, dtype=torch.long).to(device))
                    else:
                        v_loss += loss_hm
                    prev_low, prev_high = emb_low, emb_high
                
                total_val_loss += v_loss.item()
        
        avg_val = total_val_loss / (len(val_loader) + 1e-5)
        print(f"{ckpt_name} -> Val Loss: {avg_val:.4f}")
        results.append((ckpt_name, avg_val))

    # Summary
    print("\n--- Final Results (Sorted by Loss) ---")
    results.sort(key=lambda x: x[1])
    for name, loss in results[:5]:
        print(f"Top Candidate: {name} (Loss: {loss:.4f})")

if __name__ == "__main__":
    # Update these paths for your local machine or Kaggle environment
    dataset_root = '/Users/hungcucu/Documents/usth/computer_vision/Tennis ball dataset/Dataset'
    checkpoints_path = '/Users/hungcucu/Documents/usth/computer_vision/tennis_analysis/S-KeepTrackk/checkpoint'
    evaluate_checkpoints(dataset_root, checkpoints_path)
