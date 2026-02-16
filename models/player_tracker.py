"""
Player Tracking using YOLO pretrained model.

Uses YOLOv11m pretrained on COCO dataset to detect and track players (class 0 = person).
"""

import cv2
from ultralytics import YOLO
from typing import List, Tuple, Optional
import numpy as np


class PlayerTracker:
    def __init__(self, model_path: str = "yolo11m.pt", conf_threshold: float = 0.3):
        """
        Initialize player tracker.
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_id = 0  # COCO class 0 = person
    
    def detect(self, frame: np.ndarray) -> List[dict]:
        """
        Detect players in a single frame.
        
        Args:
            frame: BGR image (numpy array)
        
        Returns:
            List of detections, each containing:
                - bbox: (x1, y1, x2, y2)
                - confidence: float
                - center: (cx, cy)
        """
        results = self.model(frame, conf=self.conf_threshold, classes=[self.class_id], verbose=False)[0]
        
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': conf,
                    'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
                })
        
        return detections
    
    def track(self, frame: np.ndarray) -> List[dict]:
        """
        Track players across frames (with persistent IDs).
        
        Args:
            frame: BGR image (numpy array)
        
        Returns:
            List of tracked players, each containing:
                - bbox: (x1, y1, x2, y2)
                - confidence: float
                - center: (cx, cy)
                - track_id: int (persistent across frames)
        """
        results = self.model.track(
            frame, 
            conf=self.conf_threshold, 
            classes=[self.class_id], 
            persist=True, 
            verbose=False
        )[0]
        
        tracked = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                
                tracked.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': conf,
                    'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                    'track_id': track_id
                })
        
        return tracked
    
    def draw(self, frame: np.ndarray, players: List[dict], show_id: bool = True) -> np.ndarray:
        """
        Draw player bounding boxes on frame.
        
        Args:
            frame: BGR image
            players: List of player detections/tracks
            show_id: Whether to show track ID
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for player in players:
            x1, y1, x2, y2 = player['bbox']
            conf = player['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            if show_id and 'track_id' in player and player['track_id'] >= 0:
                label = f"Player {player['track_id']} ({conf:.2f})"
            else:
                label = f"Player ({conf:.2f})"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 6), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return annotated


def main():
    """Test player tracker on video."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test player tracker")
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--model', type=str, default='yolo11m.pt', help='YOLO model path')
    parser.add_argument('--output', type=str, default=None, help='Output video path')
    args = parser.parse_args()
    
    tracker = PlayerTracker(model_path=args.model)
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        players = tracker.track(frame)
        annotated = tracker.draw(frame, players)
        
        if out:
            out.write(annotated)
        
        cv2.imshow("Player Tracking", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
