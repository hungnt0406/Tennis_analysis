import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm

from backbone import Backbone
from candidate_extraction import TargetClassifier, extract_candidates
from feature_encoding import FeatureEncoding
from candidate_embedding import CandidateEmbedding
from association import ObjectAssociation

def run_inference(video_path, checkpoint_path, output_path):
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # Initialize Modules
    backbone = Backbone().to(device)
    target_classifier = TargetClassifier(in_channels=1024).to(device)
    feat_encode_low = FeatureEncoding(feat_stride=8, roi_size=5).to(device)
    feat_encode_high = FeatureEncoding(feat_stride=16, roi_size=5).to(device)
    embed_low = CandidateEmbedding(in_dim=512*5*5, embed_dim=256).to(device)
    embed_high = CandidateEmbedding(in_dim=1024*5*5, embed_dim=256).to(device)
    association = ObjectAssociation(embed_dim=256).to(device)

    # Load Weights
    print(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone.load_state_dict(checkpoint['backbone'])
    target_classifier.load_state_dict(checkpoint['target_classifier'])
    feat_encode_low.load_state_dict(checkpoint['feat_encode_low'])
    feat_encode_high.load_state_dict(checkpoint['feat_encode_high'])
    embed_low.load_state_dict(checkpoint['embed_low'])
    embed_high.load_state_dict(checkpoint['embed_high'])
    association.load_state_dict(checkpoint['association'])

    # Eval mode
    backbone.eval()
    target_classifier.eval()
    feat_encode_low.eval()
    feat_encode_high.eval()
    embed_low.eval()
    embed_high.eval()
    association.eval()

    # Image Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    target_w, target_h = 640, 360
    scale_x = width / target_w
    scale_y = height / target_h

    prev_low, prev_high = None, None
    best_idx_prev = 0

    print(f"Running inference on {video_path}...")
    pbar = tqdm(total=total_frames)
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(rgb_frame, (target_w, target_h))
            img_tensor = transform(resized_img).unsqueeze(0).to(device) # [1, C, H, W]

            # Forward Pass
            feat_low, feat_high = backbone(img_tensor)
            pred_heatmap = target_classifier(feat_high)
            candidates, top_scores = extract_candidates(pred_heatmap, top_k=10) # [1, 10, 2]
            
            # Phase 2: Feature encoding
            cand_low = candidates * 2.0
            cand_high = candidates
            
            roi_feat_low = feat_encode_low(feat_low, cand_low)
            roi_feat_high = feat_encode_high(feat_high, cand_high)
            
            emb_low = embed_low(roi_feat_low)
            emb_high = embed_high(roi_feat_high)

            best_idx_curr = 0
            
            # Phase 3: Spatial-Temporal Association matching
            if prev_low is not None:
                A = association(prev_low, emb_low, prev_high, emb_high) # [1, 10, 10]
                # A[0, best_idx_prev, :] gives the association scores from the previous best candidate to all current candidates
                association_scores = A[0, best_idx_prev, :]
                best_idx_curr = torch.argmax(association_scores).item()
            else:
                # First frame, just pick the max heatmap score (which is at index 0 because topk sorts it)
                best_idx_curr = 0

            # Store the current embeddings as previous for the next frame
            prev_low, prev_high = emb_low, emb_high
            best_idx_prev = best_idx_curr

            # Get the coordinates of the chosen candidate
            chosen_candidate = candidates[0, best_idx_curr] # [2] (x, y) at stride 16 feature map scale
            score = top_scores[0, best_idx_curr].item()

            if True: # Always draw for now, regardless of score
                pred_x = chosen_candidate[0].item() * 16 * scale_x
                pred_y = chosen_candidate[1].item() * 16 * scale_y
                cv2.circle(frame, (int(pred_x), int(pred_y)), 10, (0, 0, 255), -1)
                
                # Optional: draw the score text
                cv2.putText(frame, f"Conf: {score:.3f}", (int(pred_x)+15, int(pred_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            out.write(frame)
            pbar.update(1)

    cap.release()
    out.release()
    pbar.close()
    print(f"Inference complete! Video saved to {output_path}")

if __name__ == '__main__':
    video_input = "/Users/hungcucu/Documents/usth/computer_vision/Tennis_TrackingVideo_Input2.mp4"
    checkpoint = "/Users/hungcucu/Documents/usth/computer_vision/tennis_analysis/S-KeepTrackk/checkpoint/Latest Model.pth"
    output_video = "/Users/hungcucu/Documents/usth/computer_vision/tennis_analysis/S-KeepTrackk/inference_output.mp4"
    run_inference(video_input, checkpoint, output_video)
