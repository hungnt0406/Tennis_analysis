import torch
import cv2
import numpy as np
import os
from dataloader import TennisDataset
import matplotlib.pyplot as plt

def visualize_heatmap():
    root_dir = '/Users/hungcucu/Documents/usth/computer_vision/Tennis ball dataset/Dataset'
    save_dir = '/Users/hungcucu/Documents/usth/computer_vision/tennis_analysis/S-KeepTrackk/debug'
    
    if not os.path.exists(root_dir):
        print(f"Error: Dataset directory {root_dir} not found.")
        return

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Initialize dataset
    dataset = TennisDataset(root_dir, mode='train', seq_len=3)
    if len(dataset) == 0:
        print("Error: Dataset is empty.")
        return

    # Get a sample
    sample = dataset[0]
    images = sample['images']   # [3, 3, 360, 640]
    heatmaps = sample['heatmaps'] # [3, 1, 90, 160]
    centers = sample['centers']   # [3, 2]

    # Select the first frame in the sequence
    img = images[0].permute(1, 2, 0).numpy()
    # Unnormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std + mean) * 255
    img = img.astype(np.uint8)

    heatmap = heatmaps[0, 0].numpy()
    center = centers[0].numpy()

    # Resize heatmap for overlay
    heatmap_resized = cv2.resize(heatmap, (640, 360))
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Correct alignment: Create overlay
    overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

    # Draw a circle at the ground truth center
    cv2.circle(overlay, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)

    # Plotting
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Frame (Resized)")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Generated Heatmap (160x90)")
    plt.imshow(heatmap, cmap='hot')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Overlay (Verification)")
    plt.imshow(overlay)
    plt.axis('off')

    save_path = os.path.join(save_dir, 'heatmap_verification.png')
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    plt.close()

if __name__ == '__main__':
    visualize_heatmap()
