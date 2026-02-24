import cv2
import numpy as np
import torch
import torch.nn.functional as F
from .tracknet import BallTrackerNet
from .postprocess import postprocess, refine_kps
from .homography import get_trans_matrix
from .court_reference import CourtReference

class CourtKeypointDetector:
    def __init__(self, model_path, device=None, out_channels=15):
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        print(f"CourtKeypointDetector using device: {self.device}")
        
        self.model = BallTrackerNet(out_channels=out_channels)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.court_reference = CourtReference()
        self.H = None
        self.best_conf = None
        self.score = 0
        
    def detect_robust(self, image, use_refine_kps=False, use_homography=True):
        self.detect(image, use_refine_kps, use_homography)

    def detect(self, image, use_refine_kps=False, use_homography=True):
        OUTPUT_WIDTH = 640
        OUTPUT_HEIGHT = 360
        
        orig_height, orig_width = image.shape[:2]
        img = cv2.resize(image, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        
        inp = (img.astype(np.float32) / 255.)
        inp = torch.tensor(np.rollaxis(inp, 2, 0))
        inp = inp.unsqueeze(0)
        
        out = self.model(inp.float().to(self.device))[0]
        # Avoid user warning from F.sigmoid
        pred = torch.sigmoid(out).detach().cpu().numpy()
        
        points = []
        for kps_num in range(14):
            heatmap = (pred[kps_num]*255).astype(np.uint8)
            x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25, scale=1)
            if x_pred is not None and y_pred is not None:
                x_pred_orig = x_pred * orig_width / OUTPUT_WIDTH
                y_pred_orig = y_pred * orig_height / OUTPUT_HEIGHT
                
                if use_refine_kps and kps_num not in [8, 12, 9]:
                    x_pred_orig, y_pred_orig = refine_kps(image, int(y_pred_orig), int(x_pred_orig))
                points.append((x_pred_orig, y_pred_orig))
            else:
                points.append((None, None))
                
        if use_homography:
            matrix_trans = get_trans_matrix(points)
            if matrix_trans is not None:
                self.H = matrix_trans
                self.score = 100
                self.best_conf = "CNN"
            else:
                self.H = None
                
        return points
        
    def is_valid(self):
        return self.H is not None
