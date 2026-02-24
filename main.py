"""
Tennis Analysis Main Pipeline

Integrates:
- Court Detection (with minimap)
- Player Tracking (YOLO)
- Ball Tracking (YOLO)

Usage:
    python -m tennis_analysis.main --video INPUT_VIDEO --output OUTPUT_VIDEO
"""

import cv2
import numpy as np
import argparse
import os
import sys
from pathlib import Path
import time

from collections import deque

# Import our modules
from models.ball_tracker import BallTracker
from models.player_tracker import PlayerTracker
from court_keypoint import CourtKeypointDetector

#the minimap is not showing the playet-dots and the ball-dots(minimap should contain the dynamic dots according to the players and ball)
def create_court_minimap(width=200, height=350):
    """
    Create a 2D schematic court diagram.
    
    Args:
        width: Minimap width in pixels
        height: Minimap height in pixels
    
    Returns:
        Court minimap image (BGR)
    """
    # Create black background
    minimap = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Define court proportions
    margin = 10
    court_width = width - 2 * margin
    court_height = height - 2 * margin
    
    # Court boundaries
    x1, y1 = margin, margin
    x2, y2 = margin + court_width, margin + court_height
    
    # Draw outer rectangle (doubles court)
    cv2.rectangle(minimap, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
    # Singles sidelines (inner lines)
    singles_width = int(court_width * 0.75)
    singles_margin = (court_width - singles_width) // 2
    x_left_singles = margin + singles_margin
    x_right_singles = margin + court_width - singles_margin
    cv2.line(minimap, (x_left_singles, y1), (x_left_singles, y2), (255, 255, 255), 1)
    cv2.line(minimap, (x_right_singles, y1), (x_right_singles, y2), (255, 255, 255), 1)
    
    # Service boxes
    service_line_top = margin + int(court_height * 0.25)
    service_line_bottom = margin + int(court_height * 0.75)
    cv2.line(minimap, (x1, service_line_top), (x2, service_line_top), (255, 255, 255), 1)
    cv2.line(minimap, (x1, service_line_bottom), (x2, service_line_bottom), (255, 255, 255), 1)
    
    # Net (center line)
    net_y = margin + court_height // 2
    cv2.line(minimap, (x1, net_y), (x2, net_y), (200, 200, 200), 2)
    
    # Center service line
    center_x = margin + court_width // 2
    cv2.line(minimap, (center_x, service_line_top), (center_x, service_line_bottom), (255, 255, 255), 1)
    
    # Add border
    cv2.rectangle(minimap, (0, 0), (width-1, height-1), (255, 255, 255), 1)
    
    return minimap


def draw_objects_on_minimap(minimap_base, detector, players, ball, ball_history=None):
    """
    Draw players and ball on the minimap using inverse homography.
    
    Args:
        minimap_base: Base minimap image
        detector: CourtKeypointDetector with valid homography
        players: List of player bounding boxes
        ball: Ball bounding box or None
        ball_history: Deque of past ball positions on minimap
    
    Returns:
        Minimap with objects drawn
    """
    minimap = minimap_base.copy()
    if not detector.is_valid() or detector.H is None:
        return minimap
    
    try:
        inv_H = np.linalg.inv(detector.H)
    except np.linalg.LinAlgError:
        return minimap
    
    # Reference court bounds (from CourtReference)
    ref = detector.court_reference
    # Top-Left corner of the court lines (active playing area)
    x_min = ref.baseline_top[0][0]
    y_min = ref.baseline_top[0][1]
    # Bottom-Right corner
    x_max = ref.baseline_bottom[1][0]
    y_max = ref.baseline_bottom[1][1]
    
    ref_width = x_max - x_min
    ref_height = y_max - y_min
    
    # Minimap bounds (must match create_court_minimap logic)
    # width=200, height=350, margin=10
    map_margin = 10
    map_w_total = minimap.shape[1]
    map_h_total = minimap.shape[0]
    
    map_width = map_w_total - 2 * map_margin
    map_height = map_h_total - 2 * map_margin
    
    def frame_to_minimap(point):
        # Point is (x, y)
        pt = np.array([point[0], point[1], 1.0]).reshape(3, 1)
        pt_ref = np.dot(inv_H, pt)
        
        if pt_ref[2] != 0:
            pt_ref = pt_ref / pt_ref[2] # Normalize
        
        x_ref, y_ref = pt_ref[0][0], pt_ref[1][0]
        
        # Map to minimap
        # Formula: margin + (relative_pos / ref_dim) * map_dim
        x_map = map_margin + (x_ref - x_min) * (map_width / ref_width)
        y_map = map_margin + (y_ref - y_min) * (map_height / ref_height)
        
        return int(x_map), int(y_map)

    # Draw players (Limit to 2 players)
    if players:
        # Sort players by confidence (optional) or just take first 2
        players_to_draw = players[:2] 
        
        for player_data in players_to_draw:
            # Player is a dictionary with 'bbox' key: (x1, y1, x2, y2)
            if 'bbox' in player_data:
                x1, y1, x2, y2 = player_data['bbox']
                # Use bottom center for player position (feet)
                cx = (x1 + x2) / 2
                cy = y2 
                
                mx, my = frame_to_minimap((cx, cy))
                
                # Clip to be safe
                mx = np.clip(mx, 0, minimap.shape[1]-1)
                my = np.clip(my, 0, minimap.shape[0]-1)
                
                cv2.circle(minimap, (mx, my), 4, (0, 0, 255), -1) # Red dot for players

    # Draw ball
    if ball is not None:
        # Ball is a dictionary with 'bbox' key
        if 'bbox' in ball:
            x1, y1, x2, y2 = ball['bbox']
            # Use center
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            mx, my = frame_to_minimap((cx, cy))
            
            mx = np.clip(mx, 0, minimap.shape[1]-1)
            my = np.clip(my, 0, minimap.shape[0]-1)
            
            # Update history
            if ball_history is not None:
                ball_history.append((mx, my))
            
            # Draw trail as fading dots
            if ball_history:
                for i, point in enumerate(ball_history):
                    # Calculate fading color (yellow)
                    alpha = (i + 1) / len(ball_history)
                    # BGR: Yellow is (0, 255, 255)
                    # Start faint, end strong
                    color = (0, int(255 * alpha), int(255 * alpha)) 
                    # Fixed radius for dots
                    cv2.circle(minimap, point, 2, color, -1)

            # Draw current ball position
            cv2.circle(minimap, (mx, my), 3, (0, 255, 255), -1) # Yellow dot for ball
        
    return minimap


def draw_court_lines(frame, detector, color=(0, 255, 255), thickness=2):
    """
    Draw court lines on frame using homography.
    
    Args:
        frame: Input frame
        detector: CourtLineDetector with valid homography
        color: Line color (BGR)
        thickness: Line thickness
    
    Returns:
        Frame with court lines drawn
    """
    if not detector.is_valid():
        return frame
    
    output = frame.copy()
    court_ref = detector.court_reference
    
    # Get all court lines from reference
    lines_to_draw = [
        # Baselines
        (court_ref.baseline_top[0], court_ref.baseline_top[1]),
        (court_ref.baseline_bottom[0], court_ref.baseline_bottom[1]),
        # Sidelines
        (court_ref.left_court_line[0], court_ref.left_court_line[1]),
        (court_ref.right_court_line[0], court_ref.right_court_line[1]),
        # Service lines
        (court_ref.top_inner_line[0], court_ref.top_inner_line[1]),
        (court_ref.bottom_inner_line[0], court_ref.bottom_inner_line[1]),
        # Inner sidelines
        (court_ref.left_inner_line[0], court_ref.left_inner_line[1]),
        (court_ref.right_inner_line[0], court_ref.right_inner_line[1]),
        # Middle line
        (court_ref.middle_line[0], court_ref.middle_line[1]),
        # Net
        (court_ref.net[0], court_ref.net[1]),
    ]
    
    # Transform each line to frame coordinates and draw
    for pt1, pt2 in lines_to_draw:
        # Transform points using homography
        pt1_array = np.array([[pt1]], dtype=np.float32)
        pt2_array = np.array([[pt2]], dtype=np.float32)
        
        pt1_frame = cv2.perspectiveTransform(pt1_array, detector.H)
        pt2_frame = cv2.perspectiveTransform(pt2_array, detector.H)
        
        # Convert to integer coordinates
        p1 = tuple(pt1_frame[0][0].astype(int))
        p2 = tuple(pt2_frame[0][0].astype(int))
        
        # Draw line
        cv2.line(output, p1, p2, color, thickness, cv2.LINE_AA)
    
    return output


def draw_court_keypoints(frame, detector, color=(0, 0, 255), radius=5):
    """
    Draw court keypoints as dots on frame using homography.
    
    Args:
        frame: Input frame
        detector: CourtKeypointDetector with valid homography
        color: Dot color (BGR)
        radius: Dot radius
    
    Returns:
        Frame with keypoints drawn
    """
    if not detector.is_valid() or detector.H is None:
        return frame
    
    output = frame.copy()
    court_ref = detector.court_reference
    
    # Get all keypoints from reference
    kps = court_ref.key_points
    
    # Transform to frame coordinates
    # kps is list of tuples, convert to (N, 1, 2) array
    kps_array = np.array([kps], dtype=np.float32)
    
    try:
        kps_frame = cv2.perspectiveTransform(kps_array, detector.H)
        
        # Draw each point
        for pt in kps_frame[0]:
            center = (int(pt[0]), int(pt[1]))
            # Check if point is inside frame
            if 0 <= center[0] < frame.shape[1] and 0 <= center[1] < frame.shape[0]:
                cv2.circle(output, center, radius, color, -1)
    except Exception:
        pass
        
    return output


def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def choose_players(court_keypoints, players):
    """
    Selects one player from the top half and one from the bottom half of the court.
    Uses indices: [4,12,6] for top and [5,13,7] for bottom.
    """
    if not players:
        return []
        
    # Indices for Side A (Top/Far) and Side B (Bottom/Near)
    top_indices = [4,12,6]
    bottom_indices = [5,13,7]
    
    player_candidates = []
    for player in players:
        track_id = player.get('track_id', -1)
        if track_id == -1 or 'bbox' not in player:
            continue
        center = get_center_of_bbox(player['bbox'])
        
        # Min distance to top set
        dist_top = min([measure_distance(center, court_keypoints[i]) for i in top_indices if i < len(court_keypoints)])
        # Min distance to bottom set
        dist_bottom = min([measure_distance(center, court_keypoints[i]) for i in bottom_indices if i < len(court_keypoints)])
        
        player_candidates.append({
            'id': track_id,
            'dist_top': dist_top,
            'dist_bottom': dist_bottom
        })
        
    if not player_candidates:
        return []
        
    # Find best player for top half
    best_top = min(player_candidates, key=lambda x: x['dist_top'])
    top_id = best_top['id']
    
    # Find best player for bottom half (among remaining)
    remaining = [p for p in player_candidates if p['id'] != top_id]
    if remaining:
        best_bottom = min(remaining, key=lambda x: x['dist_bottom'])
        bottom_id = best_bottom['id']
        return [top_id, bottom_id]
    else:
        return [top_id]


def overlay_minimap(frame, minimap, position='top-right', margin=20):
    """
    Overlay minimap on frame.
    
    Args:
        frame: Main frame
        minimap: Minimap image
        position: 'top-right', 'top-left', 'bottom-right', 'bottom-left'
        margin: Margin from edge in pixels
    
    Returns:
        Frame with minimap overlaid
    """
    output = frame.copy()
    h_map, w_map = minimap.shape[:2]
    h_frame, w_frame = frame.shape[:2]
    
    # Calculate position
    if position == 'top-right':
        y1, y2 = margin, margin + h_map
        x1, x2 = w_frame - w_map - margin, w_frame - margin
    elif position == 'top-left':
        y1, y2 = margin, margin + h_map
        x1, x2 = margin, margin + w_map
    elif position == 'bottom-right':
        y1, y2 = h_frame - h_map - margin, h_frame - margin
        x1, x2 = w_frame - w_map - margin, w_frame - margin
    else:  # bottom-left
        y1, y2 = h_frame - h_map - margin, h_frame - margin
        x1, x2 = margin, margin + w_map
    
    # Overlay with slight transparency
    alpha = 0.9
    output[y1:y2, x1:x2] = cv2.addWeighted(
        output[y1:y2, x1:x2], 1 - alpha,
        minimap, alpha, 0
    )
    
    return output


def process_video(
    input_path: str,
    output_path: str,
    ball_model_path: str,
    player_model_path: str = "/Users/hungcucu/Documents/usth/computer_vision/model/yolo11m.pt",
    court_model_path: str = "/Users/hungcucu/Documents/usth/computer_vision/courtKeyPointDetection/model/model_tennis_court_det.pt",
    static_court: bool = True,
    show_preview: bool = False
):
    """
    Process video with full tennis analysis pipeline.
    
    Args:
        input_path: Input video path
        output_path: Output video path
        ball_model_path: Path to trained ball detection YOLO model
        player_model_path: Path to YOLO model for player detection
        court_model_path: Path to CNN court keypoint detection model
        static_court: If True, detect court once (for static cameras)
        show_preview: Show preview window during processing
    """
    print("\n" + "="*70)
    print("Tennis Analysis Pipeline")
    print("="*70)
    
    # Initialize trackers
    print("\n[1/4] Initializing components...")
    # trajectory_length=5 for shorter ball trace in video
    ball_tracker = BallTracker(model_path=ball_model_path, conf_threshold=0.3, trajectory_length=5)
    player_tracker = PlayerTracker(model_path=player_model_path, conf_threshold=0.3)
    court_detector = CourtKeypointDetector(model_path=court_model_path)
    
    # Create court minimap
    minimap = create_court_minimap(width=250, height=500)
    
    # Initialize ball history for minimap
    ball_minimap_history = deque(maxlen=1)
     
    # Store IDs of the 2 active players
    chosen_player_ids = None
    
    print("   ✓ Ball tracker initialized")
    print("   ✓ Player tracker initialized")
    print("   ✓ Court detector initialized")
    print("   ✓ Minimap created")
    
    # Open input video
    print(f"\n[2/4] Opening video: {os.path.basename(input_path)}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"   ✗ Error: Cannot open video: {input_path}")
        return False
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   Resolution: {frame_width}x{frame_height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   Court mode: {'STATIC (detect once)' if static_court else 'DYNAMIC (per-frame)'}")
    
    # Create video writer
    print(f"\n[3/4] Creating output video: {os.path.basename(output_path)}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print(f"   ✗ Error: Cannot create output video: {output_path}")
        cap.release()
        return False
    
    print("   ✓ Output writer ready")
    
    # Process video
    print(f"\n[4/4] Processing frames...")
    if show_preview:
        print("   Press 'q' to stop early")
    
    frame_count = 0
    court_detected = False
    ball_detections = 0
    player_detections = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        output_frame = frame.copy()
        
        # Court detection
        if static_court:
            if not court_detected:
                print(f"\n   Detecting court with CNN (frame {frame_count})...")
                court_detector.detect_robust(frame)
                if court_detector.is_valid():
                    court_detected = True
                    print(f"   ✓ Court detected! Config: {court_detector.best_conf}, Score: {court_detector.score:.0f}")
        else:
            # Dynamic mode: detect every frame
            court_detector.detect(frame)
        
        # Draw court keypoints
        if court_detector.is_valid():
            output_frame = draw_court_keypoints(output_frame, court_detector, color=(0, 0, 255), radius=5)
        
        # Track players
        players = player_tracker.track(frame)
        
        # Filter logic using chosen_player_ids
        if court_detector.is_valid():
            try:
                # Get keypoints for distance measurement
                ref_kps = np.array([court_detector.court_reference.key_points], dtype=np.float32)
                image_kps = cv2.perspectiveTransform(ref_kps, court_detector.H)[0]
                
                # Choose players ONCE
                if chosen_player_ids is None and players:
                    chosen_player_ids = choose_players(image_kps, players)
                    print(f"   ✓ Locked onto players: {chosen_player_ids}")
                
                # Filter players by ID
                if chosen_player_ids:
                    players = [p for p in players if p.get('track_id') in chosen_player_ids]
            except Exception:
                pass
        
        if players:
            player_detections += len(players)
            output_frame = player_tracker.draw(output_frame, players, show_id=True)
        
        # Track ball
        ball = ball_tracker.track(frame)
        if ball:
            ball_detections += 1
            output_frame = ball_tracker.draw(output_frame, ball, draw_trajectory=True)
        
        # Update and overlay minimap
        minimap_with_objects = draw_objects_on_minimap(minimap, court_detector, players, ball, ball_history=ball_minimap_history)
        output_frame = overlay_minimap(output_frame, minimap_with_objects, position='top-right', margin=20)
        
        # Add stats overlay
        stats_y = 30
        cv2.putText(output_frame, f"Frame: {frame_count}/{total_frames}", 
                   (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(output_frame, f"Players: {len(players) if players else 0}", 
                   (10, stats_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(output_frame, f"Ball: {'Yes' if ball else 'No'}", 
                   (10, stats_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write frame
        out.write(output_frame)
        
        # Show preview
        if show_preview:
            cv2.imshow('Tennis Analysis', output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n   Stopped by user")
                break
        
        # Progress update
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   Progress: {progress:.1f}% ({frame_count}/{total_frames}) | "
                  f"Ball: {ball_detections}/{frame_count} | "
                  f"Players: {player_detections}")
    
    # Cleanup
    cap.release()
    out.release()
    if show_preview:
        cv2.destroyAllWindows()
    
    # Summary
    print("\n" + "="*70)
    print("Processing Complete!")
    print("="*70)
    print(f"Output saved to: {output_path}")
    print(f"Total frames: {frame_count}")
    print(f"Court detected: {'Yes' if court_detected or court_detector.is_valid() else 'No'}")
    print(f"Ball detections: {ball_detections} ({ball_detections/frame_count*100:.1f}%)")
    print(f"Player detections: {player_detections}")
    print("="*70 + "\n")
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Tennis Analysis - Ball Tracking, Player Tracking, and Court Detection"
    )
    
    # Required arguments
    parser.add_argument(
        '--video', 
        type=str, 
        default='/Users/hungcucu/Documents/usth/computer_vision/Tennis_TrackingVideo_Input2.mp4',
        help='Path to input video'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='/Users/hungcucu/Documents/usth/computer_vision/tennis_analysis/output_main.mp4',
        help='Path to output video'
    )
    parser.add_argument(
        '--ball-model', 
        type=str, 
        default='/Users/hungcucu/Documents/usth/computer_vision/tennis_analysis/models/kaggle/working/runs/detect/tennis_yolo11n/weights/best.pt',
        help='Path to trained YOLO ball detection model'
    )
    
    # Optional arguments
    parser.add_argument(
        '--player-model', 
        type=str, 
        default='/Users/hungcucu/Documents/usth/computer_vision/model/yolo11m.pt',
        help='Path to YOLO player detection model (default: yolo11m.pt)'
    )
    parser.add_argument(
        '--court-model', 
        type=str, 
        default='/Users/hungcucu/Documents/usth/computer_vision/courtKeyPointDetection/model/model_tennis_court_det.pt',
        help='Path to CNN court keypoint detection model'
    )
    parser.add_argument(
        '--static-court', 
        action='store_true',
        help='Use static court detection (detect once, for fixed cameras)'
    )
    parser.add_argument(
        '--preview', 
        action='store_true',
        help='Show preview window during processing'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.video):
        print(f"Error: Input video not found: {args.video}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process video
    success = process_video(
        input_path=args.video,
        output_path=args.output,
        ball_model_path=args.ball_model,
        player_model_path=args.player_model,
        court_model_path=args.court_model,
        static_court=args.static_court,
        show_preview=args.preview
    )
    
    if not success:
        print("✗ Processing failed")
        sys.exit(1)
    
    print("✓ Success!")


if __name__ == "__main__":
    
    start_time  = time.time()
    main()
    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds")
