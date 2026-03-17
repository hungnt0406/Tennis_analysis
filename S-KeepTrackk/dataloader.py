import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class TennisDataset(Dataset):
    def __init__(self, root_dir, mode='train', seq_len=3):
        self.root_dir = root_dir
        self.mode = mode
        self.seq_len = seq_len
        
        # 80/10/10 split
        if mode == 'train':
            self.games = [f'game{i}' for i in range(1, 9)]
        elif mode == 'val':
            self.games = ['game9']
        elif mode == 'test':
            self.games = ['game10']
        
        self.samples = self._parse_dataset()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _parse_dataset(self):
        samples = []
        for game in self.games:
            game_path = os.path.join(self.root_dir, game)
            if not os.path.exists(game_path):
                continue
            for clip in os.listdir(game_path):
                clip_path = os.path.join(game_path, clip)
                if not os.path.isdir(clip_path):
                    continue
                csv_path = os.path.join(clip_path, 'Label.csv')
                if not os.path.exists(csv_path):
                    continue
                
                df = pd.read_csv(csv_path)
                df = df[df['visibility'] > 0].reset_index(drop=True)
                
                for i in range(len(df) - self.seq_len + 1):
                    seq = []
                    for j in range(self.seq_len):
                        row = df.iloc[i+j]
                        img_path = os.path.join(clip_path, row['file name'])
                        x, y = int(row['x-coordinate']), int(row['y-coordinate'])
                        seq.append((img_path, x, y))
                    samples.append(seq)
        return samples

    def generate_heatmap(self, width, height, x, y, sigma=1):
        heatmap = np.zeros((height, width), dtype=np.float32)
        tmp_size = sigma * 3
        ul = [int(x - tmp_size), int(y - tmp_size)]
        br = [int(x + tmp_size + 1), int(y + tmp_size + 1)]
        
        size = 2 * tmp_size + 1
        x_mesh = np.arange(0, size, 1, np.float32)
        y_mesh = np.arange(0, size, 1, np.float32)
        x_mesh, y_mesh = np.meshgrid(x_mesh, y_mesh)
        
        gaussian = np.exp(- ((x_mesh - tmp_size)**2 + (y_mesh - tmp_size)**2) / (2 * sigma**2))
        
        g_x = max(0, -ul[0]), min(br[0], width) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], height) - ul[1]
        
        img_x = max(0, ul[0]), min(br[0], width)
        img_y = max(0, ul[1]), min(br[1], height)
        
        if img_x[0] >= img_x[1] or img_y[0] >= img_y[1]:
            return heatmap
            
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
            heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
            gaussian[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        )
        return heatmap

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        
        images = []
        heatmaps = []
        centers = []
        
        target_w, target_h = 640, 360
        
        for img_path, x, y in seq:
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            orig_h, orig_w = img.shape[:2]
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h
            
            img = cv2.resize(img, (target_w, target_h))
            
            scaled_x = int(x * scale_x)
            scaled_y = int(y * scale_y)
            
            heatmap = self.generate_heatmap(target_w // 4, target_h // 4, scaled_x // 4, scaled_y // 4)
            
            images.append(self.transform(img))
            heatmaps.append(torch.tensor(heatmap).unsqueeze(0))
            centers.append(torch.tensor([scaled_x, scaled_y], dtype=torch.float32))
            
        return {
            'images': torch.stack(images),
            'heatmaps': torch.stack(heatmaps),
            'centers': torch.stack(centers)
        }
