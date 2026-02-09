"""
Visualize bounding box to adjust ball_size parameter.
"""

import os 
import csv
import shutil
from PIL import Image
import cv2


def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size


def csv_to_yolo(x, y, img_width, img_height, ball_size=12):
    """
    Convert point annotation to YOLO bounding box format.
    
    Args:
        x: x-coordinate of ball center (pixels)
        y: y-coordinate of ball center (pixels)
        img_width: image width (pixels)
        img_height: image height (pixels)
        ball_size: estimated ball diameter (pixels), default 12
    
    Returns:
        tuple: (x_center, y_center, width, height) normalized to [0, 1]
    """
    # Normalize coordinates
    x_center = x / img_width
    y_center = y / img_height
    
    # Normalize ball size (assuming square bounding box)
    width = ball_size / img_width
    height = ball_size / img_height
    
    # Clamp values to [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    
    return x_center, y_center, width, height


def yolo_to_pixel(x_center, y_center, width, height, img_width, img_height):
    """
    Convert YOLO normalized format back to pixel coordinates for visualization.
    
    Returns:
        tuple: (x1, y1, x2, y2) pixel coordinates of bounding box corners
    """
    # Convert center + size to corner coordinates
    box_w = width * img_width
    box_h = height * img_height
    cx = x_center * img_width
    cy = y_center * img_height
    
    x1 = int(cx - box_w / 2)
    y1 = int(cy - box_h / 2)
    x2 = int(cx + box_w / 2)
    y2 = int(cy + box_h / 2)
    
    return x1, y1, x2, y2


def visualize_bbox(image_path, x, y, ball_size=12):
    """
    Visualize bounding box on image.
    
    Args:
        image_path: path to image
        x: ball x-coordinate (pixels)
        y: ball y-coordinate (pixels)
        ball_size: ball diameter in pixels
    """
    img_width, img_height = get_image_size(image_path)
    
    # Get YOLO format
    x_center, y_center, width, height = csv_to_yolo(x, y, img_width, img_height, ball_size)
    
    # Convert back to pixel coordinates for drawing
    x1, y1, x2, y2 = yolo_to_pixel(x_center, y_center, width, height, img_width, img_height)
    
    # Draw
    image = cv2.imread(image_path)
    
    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Draw center point
    cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
    
    # Add text info
    info = f"ball_size={ball_size}px | bbox: {x2-x1}x{y2-y1}px"
    cv2.putText(image, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Show YOLO format values
    yolo_info = f"YOLO: {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}"
    cv2.putText(image, yolo_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    cv2.imshow(f"Ball Size = {ball_size}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def compare_ball_sizes(image_path, x, y, sizes=[8, 10, 12, 15, 20]):
    """
    Compare different ball sizes side by side.
    """
    img_width, img_height = get_image_size(image_path)
    image = cv2.imread(image_path)
    
    colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
    ]
    
    for i, ball_size in enumerate(sizes):
        x_center, y_center, width, height = csv_to_yolo(x, y, img_width, img_height, ball_size)
        x1, y1, x2, y2 = yolo_to_pixel(x_center, y_center, width, height, img_width, img_height)
        
        color = colors[i % len(colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Label
        cv2.putText(image, f"{ball_size}px", (x2 + 5, y1 + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw center point
    cv2.circle(image, (x, y), 3, (255, 255, 255), -1)
    
    # Legend
    for i, ball_size in enumerate(sizes):
        color = colors[i % len(colors)]
        cv2.putText(image, f"ball_size={ball_size}px", (10, 30 + i*25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.imshow("Compare Ball Sizes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    image_path = "/Users/hungcucu/Documents/usth/computer_vision/Tennis ball dataset/Dataset/game1/Clip1/0037.jpg"
    x, y = 806,156  # Ball coordinates from CSV
    
    # Option 1: Visualize single ball size
    # visualize_bbox(image_path, x, y, ball_size=12)
    
    # Option 2: Compare multiple ball sizes
    compare_ball_sizes(image_path, x, y, sizes=[8])


if __name__ == "__main__":
    main()
