"""
Evaluate YOLO on game10.

Usage:
    python evaluate_yolo_game10.py
    python evaluate_yolo_game10.py --yolo_conf 0.2
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
import time

from ultralytics import YOLO


# ============================================================================
# Utility Functions
# ============================================================================

def pixel_distance(pred: Tuple[float, float], gt: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((pred[0] - gt[0])**2 + (pred[1] - gt[1])**2)


def load_clip_data(clip_dir: Path) -> List[Dict]:
    """
    Load all frames and labels from a clip.
    
    Returns:
        List of dicts with frame info
    """
    label_csv = clip_dir / "Label.csv"
    if not label_csv.exists():
        return []
    
    frames = []
    with open(label_csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    for i, row in enumerate(rows):
        filename = row['file name']
        img_path = clip_dir / filename
        
        if not img_path.exists():
            continue
        
        visibility = int(row['visibility'])
        
        if visibility > 0:
            try:
                x = float(row['x-coordinate'])
                y = float(row['y-coordinate'])
            except (ValueError, KeyError):
                x, y = None, None
        else:
            x, y = None, None
        
        frames.append({
            'path': str(img_path),
            'gt_x': x,
            'gt_y': y,
            'visibility': visibility,
            'status': int(row['status']) if row['status'] else 0,
            'frame_idx': i,
            'clip': clip_dir.name
        })
    
    return frames


# ============================================================================
# YOLO Evaluation
# ============================================================================

def run_yolo(model: YOLO, img_path: str, conf_threshold: float = 0.3):
    """
    Run YOLO detection on a single image.
    
    Returns:
        (x, y, confidence) or (None, None, 0)
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, None, 0
    
    results = model(img, conf=conf_threshold, verbose=False)[0]
    
    if results.boxes is not None and len(results.boxes) > 0:
        best_conf = -1
        best_pos = None
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                best_pos = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                best_conf = conf
        return best_pos[0], best_pos[1], best_conf
    
    return None, None, 0


def evaluate_yolo_game10(
    yolo_model_path: str,
    game_dir: str,
    distance_threshold: float = 10.0,
    yolo_conf: float = 0.3,
    save_results: str = None
):
    """
    Evaluate YOLO on game10.
    """
    print("=" * 70)
    print("YOLO Evaluation on Game10")
    print("=" * 70)
    print(f"YOLO model: {yolo_model_path}")
    print(f"Game directory: {game_dir}")
    print(f"Distance threshold: {distance_threshold}px")
    print(f"YOLO confidence threshold: {yolo_conf}")
    print()
    
    # Load model
    print("Loading YOLO model...")
    yolo_model = YOLO(yolo_model_path)
    
    # Load all clips from game10 (sorted numerically)
    game_dir = Path(game_dir)
    clips = sorted(
        [d for d in game_dir.iterdir() if d.is_dir() and d.name.startswith('Clip')],
        key=lambda x: int(x.name.replace('Clip', ''))
    )
    
    print(f"\nFound {len(clips)} clips")
    
    # Load all frames
    all_frames = []
    for clip in clips:
        frames = load_clip_data(clip)
        all_frames.extend(frames)
        print(f"  {clip.name}: {len(frames)} frames")
    
    print(f"\nTotal frames: {len(all_frames)}")
    print()
    
    # Metrics
    metrics = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'distances': [], 'times': []}
    
    results = []
    
    # Evaluate
    for frame in tqdm(all_frames, desc="Evaluating YOLO"):
        gt_x, gt_y = frame['gt_x'], frame['gt_y']
        has_gt = gt_x is not None and gt_y is not None
        
        # Run YOLO
        t0 = time.time()
        yolo_x, yolo_y, yolo_conf_val = run_yolo(yolo_model, frame['path'], yolo_conf)
        yolo_time = time.time() - t0
        metrics['times'].append(yolo_time)
        
        yolo_dist = None
        if yolo_x is not None and has_gt:
            yolo_dist = pixel_distance((yolo_x, yolo_y), (gt_x, gt_y))
            if yolo_dist <= distance_threshold:
                metrics['tp'] += 1
                metrics['distances'].append(yolo_dist)
            else:
                metrics['fp'] += 1
        elif yolo_x is not None and not has_gt:
            metrics['fp'] += 1
        elif yolo_x is None and has_gt:
            metrics['fn'] += 1
        else:
            metrics['tn'] += 1
        
        # Store result
        results.append({
            'clip': frame['clip'],
            'frame_idx': frame['frame_idx'],
            'visibility': frame['visibility'],
            'gt_x': gt_x,
            'gt_y': gt_y,
            'yolo_x': yolo_x,
            'yolo_y': yolo_y,
            'yolo_conf': yolo_conf_val,
            'yolo_dist': yolo_dist,
            'yolo_correct': yolo_dist is not None and yolo_dist <= distance_threshold,
        })
    
    # Calculate final metrics
    eps = 1e-10
    precision = metrics['tp'] / (metrics['tp'] + metrics['fp'] + eps)
    recall = metrics['tp'] / (metrics['tp'] + metrics['fn'] + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    accuracy = (metrics['tp'] + metrics['tn']) / (metrics['tp'] + metrics['tn'] + metrics['fp'] + metrics['fn'] + eps)
    mean_dist = np.mean(metrics['distances']) if metrics['distances'] else 0
    std_dist = np.std(metrics['distances']) if metrics['distances'] else 0
    avg_time = np.mean(metrics['times']) * 1000  # ms
    fps = 1.0 / np.mean(metrics['times']) if metrics['times'] else 0
    
    # Print results
    print()
    print("=" * 70)
    print("RESULTS - YOLO")
    print("=" * 70)
    print(f"Total frames evaluated: {len(all_frames)}")
    print(f"Frames with ball (ground truth): {sum(1 for f in all_frames if f['gt_x'] is not None)}")
    print()
    
    print("-" * 50)
    print(f"{'Metric':<30} {'Value':>15}")
    print("-" * 50)
    print(f"{'Precision':<30} {precision:>15.4f}")
    print(f"{'Recall':<30} {recall:>15.4f}")
    print(f"{'F1-Score':<30} {f1:>15.4f}")
    print(f"{'Accuracy':<30} {accuracy:>15.4f}")
    print("-" * 50)
    print(f"{'True Positives':<30} {metrics['tp']:>15}")
    print(f"{'False Positives':<30} {metrics['fp']:>15}")
    print(f"{'False Negatives':<30} {metrics['fn']:>15}")
    print(f"{'True Negatives':<30} {metrics['tn']:>15}")
    print("-" * 50)
    print(f"{'Mean Distance (px)':<30} {mean_dist:>15.2f}")
    print(f"{'Std Distance (px)':<30} {std_dist:>15.2f}")
    print("-" * 50)
    print(f"{'Avg Time (ms)':<30} {avg_time:>15.1f}")
    print(f"{'FPS':<30} {fps:>15.1f}")
    print("-" * 50)
    
    # Save results
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
        'mean_dist': mean_dist,
        'std_dist': std_dist,
        'avg_time_ms': avg_time,
        'fps': fps,
        **metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO on game10")
    parser.add_argument(
        '--yolo_model',
        type=str,
        default='/Users/hungcucu/Documents/usth/computer_vision/tennis_analysis/models/kaggle/working/runs/detect/tennis_yolo11n/weights/best.pt',
        help='Path to YOLO model'
    )
    parser.add_argument(
        '--game_dir',
        type=str,
        default='/Users/hungcucu/Documents/usth/computer_vision/Tennis ball dataset/Dataset/game10',
        help='Path to game10 directory'
    )
    parser.add_argument(
        '--dist_thresh',
        type=float,
        default=10.0,
        help='Distance threshold for correct detection (pixels)'
    )
    parser.add_argument(
        '--yolo_conf',
        type=float,
        default=0.3,
        help='YOLO confidence threshold'
    )
    parser.add_argument(
        '--save_results',
        type=str,
        default='yolo_game10_results.csv',
        help='Path to save detailed results CSV'
    )
    
    args = parser.parse_args()
    
    evaluate_yolo_game10(
        yolo_model_path=args.yolo_model,
        game_dir=args.game_dir,
        distance_threshold=args.dist_thresh,
        yolo_conf=args.yolo_conf,
        save_results=args.save_results
    )


if __name__ == "__main__":
    main()
