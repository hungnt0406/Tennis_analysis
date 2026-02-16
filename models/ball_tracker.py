"""
Ball Tracking using custom YOLO model + Kalman Filter interpolation.

Detects tennis ball and smooths trajectory using Kalman filter.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional
from collections import deque


class BallTracker:
    def __init__(
        self, 
        model_path: str,
        conf_threshold: float = 0.3,
        trajectory_length: int = 30
    ):
        """
        Initialize ball tracker.
        
        Args:
            model_path: Path to trained YOLO ball detection model
            conf_threshold: Confidence threshold for detections
            trajectory_length: Number of past positions to keep for visualization
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        # Kalman filter removed as requested
        self.trajectory = deque(maxlen=trajectory_length)
        self.last_detection = None
    
    def detect(self, frame: np.ndarray) -> Optional[dict]:
        """
        Detect ball in frame (raw detection, no filtering).
        
        Args:
            frame: BGR image
        
        Returns:
            Detection dict or None if not detected
        """
        results = self.model.track(frame, conf=self.conf_threshold, verbose=False, persist=True)[0]
        
        if results.boxes is None or len(results.boxes) == 0:
            return None
        
        # Get best detection (highest confidence)
        best_box = None
        best_conf = -1
        
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                best_conf = conf
                best_box = {
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': conf,
                    'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
                }
        
        return best_box
    
    def track(self, frame: np.ndarray) -> Optional[dict]:
        """
        Track ball using YOLO detection only (no Kalman filter).
        
        Args:
            frame: BGR image
        
        Returns:
            Tracked position dict with:
                - center: (x, y) position
                - confidence: detection confidence
                - detected: True
                - velocity: None (not calculated without filter)
        """
        detection = self.detect(frame)
        
        if detection is not None:
            result = {
                'center': detection['center'],
                'raw_center': detection['center'],
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'detected': True,
                'velocity': None  # Velocity not available without Kalman filter
            }
            self.last_detection = result
            self.trajectory.append(result['center'])
            return result
        
        return None
    
    def get_trajectory(self) -> List[Tuple[int, int]]:
        """Get recent ball trajectory."""
        return list(self.trajectory)
    
    def draw(
        self, 
        frame: np.ndarray, 
        ball: Optional[dict],
        draw_trajectory: bool = True,
        draw_velocity: bool = False
    ) -> np.ndarray:
        """
        Draw ball detection and trajectory on frame.
        
        Args:
            frame: BGR image
            ball: Ball tracking result
            draw_trajectory: Whether to draw trajectory trail
            draw_velocity: Ignored (velocity not available)
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw trajectory
        if draw_trajectory and len(self.trajectory) > 1:
            points = list(self.trajectory)
            for i in range(1, len(points)):
                # Fade color based on age
                alpha = i / len(points)
                color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                thickness = max(1, int(3 * alpha))
                cv2.line(annotated, points[i-1], points[i], color, thickness)
        
        if ball is None:
            return annotated
        
        cx, cy = ball['center']
        
        # Draw ball position (Green circle)
        cv2.circle(annotated, (cx, cy), 8, (0, 255, 0), 2)
        cv2.circle(annotated, (cx, cy), 3, (0, 255, 0), -1)
        
        # Draw bbox if available
        if ball['bbox']:
            x1, y1, x2, y2 = ball['bbox']
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        label = f"Ball ({ball['confidence']:.2f})"
        
        # Draw label
        cv2.putText(annotated, label, (cx + 10, cy - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated


def main():
    """Test ball tracker on video."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ball tracker")
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--model', type=str, required=True, help='YOLO ball model path')
    parser.add_argument('--output', type=str, default=None, help='Output video path')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    args = parser.parse_args()
    
    tracker = BallTracker(model_path=args.model, conf_threshold=args.conf)
    
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
    
    frame_count = 0
    detected_count = 0
    predicted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        ball = tracker.track(frame)
        
        if ball:
            if ball['detected']:
                detected_count += 1
            else:
                predicted_count += 1
        
        annotated = tracker.draw(frame, ball, draw_trajectory=True, draw_velocity=True)
        
        # Add stats
        stats = f"Frame: {frame_count} | Detected: {detected_count} | Predicted: {predicted_count}"
        cv2.putText(annotated, stats, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if out:
            out.write(annotated)
        
        cv2.imshow("Ball Tracking", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"\nSummary:")
    print(f"Total frames: {frame_count}")
    print(f"Detected: {detected_count} ({detected_count/frame_count*100:.1f}%)")
    print(f"Predicted: {predicted_count} ({predicted_count/frame_count*100:.1f}%)")


if __name__ == "__main__":
    main()
