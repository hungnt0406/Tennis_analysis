import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import TennisDataset
from backbone import Backbone
from candidate_extraction import TargetClassifier, extract_candidates
from feature_encoding import FeatureEncoding
from candidate_embedding import CandidateEmbedding
from association import ObjectAssociation
from loss import focal_loss, association_loss

def find_gt_candidate_index(candidates, gt_center, stride=16):
    gt = gt_center.unsqueeze(1) / stride
    dists = torch.norm(candidates - gt, dim=-1)
    return dists.argmin(dim=-1)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    root_dir = '/Users/hungcucu/Documents/usth/computer_vision/Tennis ball dataset/Dataset'
    
    try:
        train_dataset = TennisDataset(root_dir, mode='train', seq_len=3)
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
        
        val_dataset = TennisDataset(root_dir, mode='val', seq_len=3)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    except Exception as e:
        print(f"Error initializing dataloader: {e}")
        train_loader = []
        val_loader = []

    # ... (modules and optimizer initialization)
    # [Rest of module initialization remains the same]
    backbone = Backbone().to(device)
    target_classifier = TargetClassifier(in_channels=1024).to(device)
    feat_encode_low = FeatureEncoding(feat_stride=8, roi_size=5).to(device)
    feat_encode_high = FeatureEncoding(feat_stride=16, roi_size=5).to(device)
    embed_low = CandidateEmbedding(in_dim=512*5*5, embed_dim=256).to(device)
    embed_high = CandidateEmbedding(in_dim=1024*5*5, embed_dim=256).to(device)
    association = ObjectAssociation(embed_dim=256).to(device)
    
    optimizer = optim.Adam(
        list(backbone.parameters()) + 
        list(target_classifier.parameters()) +
        list(feat_encode_low.parameters()) +
        list(feat_encode_high.parameters()) +
        list(embed_low.parameters()) +
        list(embed_high.parameters()) +
        list(association.parameters()), 
        lr=1e-4
    )

    epochs = 10
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        print(f"Epoch [{epoch+1}/{epochs}] - Training")
        backbone.train()
        target_classifier.train()
        
        total_train_loss = 0
        pbar = tqdm(train_loader) if len(train_loader) > 0 else []
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(device)
            heatmaps = batch['heatmaps'].to(device)
            centers = batch['centers'].to(device)
            B, T, C, H, W = images.shape
            
            optimizer.zero_grad()
            batch_loss = 0
            prev_low, prev_high = None, None
            prev_gt_idx = None
            
            for t in range(T):
                img_t = images[:, t] 
                gt_heatmap_t = heatmaps[:, t]
                feat_low, feat_high = backbone(img_t)
                pred_heatmap = target_classifier(feat_high)
                gt_heatmap_resized = torch.nn.functional.interpolate(gt_heatmap_t, size=pred_heatmap.shape[-2:], mode='bilinear', align_corners=False)
                loss_hm = focal_loss(pred_heatmap, gt_heatmap_resized)
                
                candidates, _ = extract_candidates(pred_heatmap, top_k=10)
                curr_gt_idx = find_gt_candidate_index(candidates, centers[:, t], stride=16)
                
                roi_feat_low = feat_encode_low(feat_low, candidates * 2.0)
                roi_feat_high = feat_encode_high(feat_high, candidates)
                emb_low, emb_high = embed_low(roi_feat_low), embed_high(roi_feat_high)
                
                if t > 0 and prev_low is not None:
                    A = association(prev_low, emb_low, prev_high, emb_high)
                    loss_assoc = association_loss(A, current_gt_index=curr_gt_idx, prev_gt_index=prev_gt_idx)
                    batch_loss += loss_hm + loss_assoc
                else:
                    batch_loss += loss_hm
                prev_low, prev_high = emb_low, emb_high
                prev_gt_idx = curr_gt_idx
                
            batch_loss.backward()
            optimizer.step()
            total_train_loss += batch_loss.item()
            if isinstance(pbar, tqdm): pbar.set_description(f"Train Loss: {batch_loss.item():.4f}")

        # --- VALIDATION PHASE ---
        print(f"Epoch [{epoch+1}/{epochs}] - Validation")
        backbone.eval()
        target_classifier.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            vbar = tqdm(val_loader) if len(val_loader) > 0 else []
            for batch in vbar:
                images, heatmaps, centers = batch['images'].to(device), batch['heatmaps'].to(device), batch['centers'].to(device)
                B, T, C, H, W = images.shape
                v_loss = 0
                prev_low, prev_high = None, None
                prev_gt_idx = None
                for t in range(T):
                    feat_low, feat_high = backbone(images[:, t])
                    pred_heatmap = target_classifier(feat_high)
                    gt_hm = torch.nn.functional.interpolate(heatmaps[:, t], size=pred_heatmap.shape[-2:], mode='bilinear', align_corners=False)
                    loss_hm = focal_loss(pred_heatmap, gt_hm)
                    candidates, _ = extract_candidates(pred_heatmap, top_k=10)
                    curr_gt_idx = find_gt_candidate_index(candidates, centers[:, t], stride=16)
                    roi_low = feat_encode_low(feat_low, candidates * 2.0)
                    roi_high = feat_encode_high(feat_high, candidates)
                    emb_low, emb_high = embed_low(roi_low), embed_high(roi_high)
                    if t > 0 and prev_low is not None:
                        A = association(prev_low, emb_low, prev_high, emb_high)
                        v_loss += loss_hm + association_loss(A, current_gt_index=curr_gt_idx, prev_gt_index=prev_gt_idx)
                    else:
                        v_loss += loss_hm
                    prev_low, prev_high = emb_low, emb_high
                    prev_gt_idx = curr_gt_idx
                total_val_loss += v_loss.item()
                if isinstance(vbar, tqdm): vbar.set_description(f"Val Loss: {v_loss.item():.4f}")

        avg_train = total_train_loss / (len(train_loader) + 1e-5)
        avg_val = total_val_loss / (len(val_loader) + 1e-5)
        print(f"Avg Train Loss: {avg_train:.4f} | Avg Val Loss: {avg_val:.4f}")
        
        checkpoint = {
            'epoch': epoch,
            'backbone': backbone.state_dict(),
            'target_classifier': target_classifier.state_dict(),
            'feat_encode_low': feat_encode_low.state_dict(),
            'feat_encode_high': feat_encode_high.state_dict(),
            'embed_low': embed_low.state_dict(),
            'embed_high': embed_high.state_dict(),
            'association': association.state_dict(),
            'val_loss': avg_val
        }
        
        # Save Latest
        torch.save(checkpoint, "latest_model.pth")
        
        # Save Best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(checkpoint, "best_model.pth")
            print(f"New Best Model saved (Val Loss: {avg_val:.4f})")


if __name__ == '__main__':
    train()
