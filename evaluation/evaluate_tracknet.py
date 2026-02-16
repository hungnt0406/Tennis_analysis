"""
Evaluate TrackNet ball detection model on test set.

TrackNet uses 3 consecutive frames as input and outputs a heatmap.

Metrics:
- Precision, Recall, F1-score
- Mean distance error (for correct detections)
- Detection rate by visibility class

Usage:
    python evaluate_tracknet.py --model path/to/Best_Model.pt --dataset path/to/Dataset
"""

import os
import sys
import argparse
import csv
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn

# Add TrackNet to path
sys.path.insert(0, '/Users/hungcucu/Documents/usth/computer_vision/tracknet_model/TrackNet')
from model import BallTrackerNet


def postprocess(feature_map: np.ndarray, scale: int = 2) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract ball position from TrackNet heatmap output.
    
    Args:
        feature_map: Model output (flattened heatmap)
        scale: Scale factor (TrackNet uses 360x640, images are 720x1280)
    
    Returns:
        (x, y) ball position in original image coordinates or (None, None)
    """
    feature_map = feature_map * 255
    feature_map = feature_map.reshape((360, 640))
    feature_map = feature_map.astype(np.uint8)
    
    ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        heatmap, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=1, 
        param1=50, 
        param2=2, 
        minRadius=2,
        maxRadius=7
    )
    
    x, y = None, None
    if circles is not None:
        if len(circles) >= 1:
            x = circles[0][0][0] * scale
            y = circles[0][0][1] * scale
    
    return x, y


def load_frames(
    frame_path: str, 
    prev_path: str, 
    preprev_path: str,
    input_width: int = 640,
    input_height: int = 360
) -> np.ndarray:
    """
    Load and preprocess 3 consecutive frames for TrackNet.
    
    Returns:
        Tensor of shape (1, 9, 360, 640)
    """
    img = cv2.imread(frame_path)
    img = cv2.resize(img, (input_width, input_height))
    
    img_prev = cv2.imread(prev_path)
    img_prev = cv2.resize(img_prev, (input_width, input_height))
    
    img_preprev = cv2.imread(preprev_path)
    img_preprev = cv2.resize(img_preprev, (input_width, input_height))
    
    # Concatenate: current, previous, pre-previous (9 channels)
    imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
    imgs = imgs.astype(np.float32) / 255.0
    imgs = np.rollaxis(imgs, 2, 0)  # (H, W, C) -> (C, H, W)
    imgs = np.expand_dims(imgs, 0)  # Add batch dimension
    
    return imgs


def load_dataset_labels(dataset_dir: str) -> Dict[str, Dict]:
    """
    Load all labels from the Tennis ball dataset.
    
    Returns:
        Dict mapping image path to label info
    """
    dataset_dir = Path(dataset_dir)
    labels = {}
    
    for label_csv in dataset_dir.rglob("Label.csv"):
        clip_dir = label_csv.parent
        
        with open(label_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        for i, row in enumerate(rows):
            filename = row['file name']
            img_path = str(clip_dir / filename)
            
            visibility = int(row['visibility'])
            
            if visibility > 0:
                try:
                    x = float(row['x-coordinate'])
                    y = float(row['y-coordinate'])
                except (ValueError, KeyError):
                    x, y = None, None
            else:
                x, y = None, None
            
            # Get previous frames
            prev_path = str(clip_dir / rows[i-1]['file name']) if i >= 1 else img_path
            preprev_path = str(clip_dir / rows[i-2]['file name']) if i >= 2 else prev_path
            
            labels[img_path] = {
                'x': x,
                'y': y,
                'visibility': visibility,
                'status': int(row['status']) if row['status'] else 0,
                'prev_path': prev_path,
                'preprev_path': preprev_path,
                'clip': str(clip_dir)
            }
    
    return labels


def pixel_distance(pred: Tuple[float, float], gt: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((pred[0] - gt[0])**2 + (pred[1] - gt[1])**2)


def evaluate_tracknet(
    model_path: str,
    dataset_dir: str,
    device: str = 'cpu',
    distance_threshold: float = 10.0,
    sample_ratio: float = 1.0,
    save_results: str = None
):
    """
    Evaluate TrackNet model on dataset.
    
    Args:
        model_path: Path to TrackNet weights
        dataset_dir: Path to Tennis ball dataset
        device: 'cpu' or 'cuda'
        distance_threshold: Max distance (pixels) for correct detection
        sample_ratio: Ratio of samples to evaluate (for faster testing)
        save_results: Path to save detailed results CSV
    """
    print("=" * 60)
    print("TrackNet Ball Detection Evaluation")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_dir}")
    print(f"Device: {device}")
    print(f"Distance threshold: {distance_threshold}px")
    print()
    
    # Load model
    model = BallTrackerNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load labels
    print("Loading dataset labels...")
    labels = load_dataset_labels(dataset_dir)
    print(f"Found {len(labels)} labeled frames")
    
    # Sample if needed
    all_paths = list(labels.keys())
    if sample_ratio < 1.0:
        np.random.shuffle(all_paths)
        all_paths = all_paths[:int(len(all_paths) * sample_ratio)]
        print(f"Evaluating on {len(all_paths)} samples ({sample_ratio*100:.0f}%)")
    print()
    
    # Metrics by visibility
    metrics = {
        0: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'distances': []},  # Not visible
        1: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'distances': []},  # Visible
        2: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'distances': []},  # Partially visible
    }
    
    results = []
    
    for img_path in tqdm(all_paths, desc="Evaluating"):
        label = labels[img_path]
        
        # Skip if files don't exist
        if not os.path.exists(img_path):
            continue
        if not os.path.exists(label['prev_path']):
            continue
        if not os.path.exists(label['preprev_path']):
            continue
        
        # Get ground truth
        gt_x, gt_y = label['x'], label['y']
        visibility = min(label['visibility'], 2)  # Clamp to 0, 1, 2
        
        # Load frames and run inference
        try:
            frames = load_frames(img_path, label['prev_path'], label['preprev_path'])
            frames_tensor = torch.from_numpy(frames).float().to(device)
            
            with torch.no_grad():
                output = model(frames_tensor, testing=True)
                output = output.argmax(dim=1).cpu().numpy()[0]
            
            pred_x, pred_y = postprocess(output, scale=2)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
        
        # Calculate metrics
        dist = None
        m = metrics[visibility]
        
        if pred_x is not None and gt_x is not None:
            dist = pixel_distance((pred_x, pred_y), (gt_x, gt_y))
            if dist <= distance_threshold:
                m['tp'] += 1
                m['distances'].append(dist)
            else:
                m['fp'] += 1
        elif pred_x is not None and gt_x is None:
            m['fp'] += 1
        elif pred_x is None and gt_x is not None:
            m['fn'] += 1
        else:
            m['tn'] += 1
        
        results.append({
            'image': img_path,
            'visibility': visibility,
            'gt_x': gt_x,
            'gt_y': gt_y,
            'pred_x': pred_x,
            'pred_y': pred_y,
            'distance': dist,
            'correct': dist is not None and dist <= distance_threshold
        })
    
    # Calculate overall metrics
    eps = 1e-10
    total_tp = sum(m['tp'] for m in metrics.values())
    total_fp = sum(m['fp'] for m in metrics.values())
    total_fn = sum(m['fn'] for m in metrics.values())
    total_tn = sum(m['tn'] for m in metrics.values())
    
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + eps)
    
    all_distances = []
    for m in metrics.values():
        all_distances.extend(m['distances'])
    
    mean_dist = np.mean(all_distances) if all_distances else 0
    std_dist = np.std(all_distances) if all_distances else 0
    
    # Print results
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Total evaluated: {len(results)}")
    print()
    
    print("Overall Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print()
    
    print("Confusion Matrix (Total):")
    print(f"  TP: {total_tp}, FP: {total_fp}, FN: {total_fn}, TN: {total_tn}")
    print()
    
    print("By Visibility Class:")
    for vis, m in metrics.items():
        vis_name = {0: 'Not visible', 1: 'Visible', 2: 'Partially visible'}[vis]
        total = m['tp'] + m['fp'] + m['fn'] + m['tn']
        if total > 0:
            p = m['tp'] / (m['tp'] + m['fp'] + eps)
            r = m['tp'] / (m['tp'] + m['fn'] + eps)
            f = 2 * p * r / (p + r + eps)
            md = np.mean(m['distances']) if m['distances'] else 0
            print(f"  {vis_name} (n={total}):")
            print(f"    Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}")
            print(f"    Mean distance: {md:.2f}px")
    
    print()
    print("Distance Error (correct detections):")
    print(f"  Mean: {mean_dist:.2f}px")
    print(f"  Std: {std_dist:.2f}px")
    print(f"  Samples: {len(all_distances)}")
    
    # Save detailed results
    if save_results:
        with open(save_results, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nDetailed results saved to: {save_results}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'mean_distance': mean_dist,
        'metrics_by_visibility': metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate TrackNet ball detection model")
    parser.add_argument(
        '--model',
        type=str,
        default='/Users/hungcucu/Documents/usth/computer_vision/tracknet_model/Best_Model.pt',
        help='Path to TrackNet model weights'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='/Users/hungcucu/Documents/usth/computer_vision/Tennis ball dataset/Dataset',
        help='Path to Tennis ball dataset'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to run inference on'
    )
    parser.add_argument(
        '--dist_thresh',
        type=float,
        default=10.0,
        help='Distance threshold for correct detection (pixels)'
    )
    parser.add_argument(
        '--sample_ratio',
        type=float,
        default=1.0,
        help='Ratio of samples to evaluate (0-1)'
    )
    parser.add_argument(
        '--save_results',
        type=str,
        default=None,
        help='Path to save detailed results CSV'
    )
    
    args = parser.parse_args()
    
    evaluate_tracknet(
        model_path=args.model,
        dataset_dir=args.dataset,
        device=args.device,
        distance_threshold=args.dist_thresh,
        sample_ratio=args.sample_ratio,
        save_results=args.save_results
    )


if __name__ == "__main__":
    main()
