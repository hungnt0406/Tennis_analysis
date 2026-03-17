# S-KeepTrack Implementation Plan for Tennis Ball Dataset

This document outlines the detailed plan to implement and train the **S-KeepTrack** architecture using your local dataset located at `@[/Users/hungcucu/Documents/usth/computer_vision/Tennis ball dataset/Dataset]`.

## 1. Dataset Analysis & Data Loading Strategy

### Data Structure & Splitting Strategy
Your dataset consists of 10 distinct matches (`game1` to `game10`) containing various video clips. 

**Critical Requirement:** You must split the dataset at the **Game** level, not by Clip or Frame. Frames from the same game share identical lighting, court colors, and player visual appearance. Splitting by frame would cause severe data leakage, leading to false high accuracy during training that fails to generalize to new videos.

**Recommended Split (80/10/10):**
*   **Training Set (80%):** `game1` to `game8`. Used to train the model weights.
*   **Validation Set (10%):** `game9`. Used to tune hyperparameters, check for overfitting, and determine early stopping during the training loops.
*   **Test Set (10%):** `game10`. Used *only* for final performance evaluation on completely unseen court environments.

### Dataloader Pipeline (`dataloader.py`)
1. **Parsing:** Parse [Label.csv](file:///Users/hungcucu/Documents/usth/computer_vision/Tennis%20ball%20dataset/Dataset/game1/Clip1/Label.csv) based on whether the current game belongs to the Train, Val, or Test set split. Filter out frames where `visibility=0` (ball not visible).
2. **Input Generation:** For each sequence, extract $T$ consecutive frames (e.g., $T=3$) to provide temporal tracking context.
3. **Ground Truth Heatmap Generation:** Convert $(x, y)$ coordinates into a 2D Gaussian heatmap. 

---

## 2. S-KeepTrack Architecture Details (Layer by Layer)

The S-KeepTrack model improves upon standard trackers by utilizing a dual-parallel branch architecture to explicitly process both **low-level** (spatial) and **high-level** (semantic) features for small objects.

### 2.1 Base Tracker / Backbone (`backbone.py`)
*   **Layer 1-4 (ResNet/VGG Variant):** You can use a modified ResNet-50.
    *   *Branch 1 (Low-level):* Extract output from an earlier block (e.g., `Conv3_x` or `Res2`) to preserve fine-grained spatial details necessary for small tennis balls.
    *   *Branch 2 (High-level):* Extract output from a deeper block (e.g., `Conv4_x` or `Res3`) for broader structural context.

### 2.2 Target Candidate Extraction (`candidate_extraction.py`)
*   **Target Classifier (Heatmap Predictor):** A fully convolutional head appended to the backbone to predict the target score map $S$ for frame $t$. 
    *   *Implementation:* Several $3 \times 3$ Conv layers followed by a $1 \times 1$ Conv layer outputting a heatmap.
*   **Candidate Extraction Setup:** Apply Non-Maximum Suppression (NMS) to the predicted heatmap $S$ to extract the top $N$ peak coordinates $V = \{v_i\}_{i=1}^N$ as target candidates.

### 2.3 Parallel Feature Encoding (`feature_encoding.py`)
To robustly track the small ball, the network encodes features for each candidate $v_i$ using the parallel representations:
*   **Low-level Encoding:** Crop/align features from the low-level backbone feature map at the candidate locations $V$.
*   **High-level Encoding:** Crop/align features from the high-level backbone feature map at the candidate locations $V$.
*   *Implementation:* Use `RoIAlign` (Region of Interest Align) initialized at candidate points to extract fixed-size feature vectors (e.g., $3 \times 3$ or $5 \times 5$ grids) for each branch.

### 2.4 Candidate Embedding (`candidate_embedding.py`)
*   **Graph/Transformer Layers:** Both the lower and higher-level candidate features pass through embedding layers to learn relationship representations across candidates.
    *   *Implementation:* Multi-head Self-Attention layers (Transformer Encoders) or Graph Convolutional Networks (GCNs). They model the spatial relationships between the True ball and distractors (false positives).

### 2.5 Object Association Module (`association.py`)
*   **Parallel Association Network:** 
    *   Calculate association matrices $A_{low}$ and $A_{high}$ between candidates in frame $t-1$ and frame $t$ for both level branches.
    *   *Fusion:* The final association matrix $A$ is a weighted sum (or learned fusion) of $A_{low}$ and $A_{high}$.
*   **Final Output:** Multiply the candidate scores by the fused association matrix to determine the most likely trajectory over time, filtering out background noise.

---

## 3. Training Plan & Loss Functions

### Loss Components (`loss.py`)
1.  **Focal Loss (Heatmap):** Used to train the Target Classifier to predict the Gaussian heatmaps (handling the heavy class imbalance between the tiny ball and the massive background court).
2.  **Offset Loss (L1/Smooth L1):** Since applying max-pooling reduces resolution, an offset loss is used to recover the exact sub-pixel $(x, y)$ coordinates.
3.  **Association Loss (Cross-Entropy):** Used to train the Target Candidate Association Network. It penalizes incorrect matches of candidates between frames $t-1$ and $t$.

### Training Loop Steps
1.  **Initialize model** with pre-trained ImageNet weights for the backbone.
2.  **Phase 1 (Base Tracker):** Train only the Backbone + Target Classifier using the Focal Loss and Offset Loss to ensure it reliably proposes reasonable candidates.
3.  **Phase 2 (End-to-End):** Freeze/Lower learning rate for the backbone, and train the Feature Encoding, Embedding, and Association Modules using the Association Loss to learn the temporal tracking of the ball.

## 4. Verification Plan

### Automated Verification
*   **Unit Tests:** Write a script `test_dataloader.py` that loads 1 batch of images, plots the images, and overlays the corresponding $(x, y)$ coordinates from [Label.csv](file:///Users/hungcucu/Documents/usth/computer_vision/Tennis%20ball%20dataset/Dataset/game1/Clip1/Label.csv) as red dots. Verify alignments visually.
*   **Loss Convergence Check:** Write a dummy training loop script using just one Clip (`game1/Clip1`) and intentionally overfit the model. If the loss reaches ~0 and predictions perfectly overlap the ground truth video, the architecture wiring is correct.

### Manual Verification
*   Once trained, run inference on the remaining clips (e.g., `game10`).
*   Output the video with a drawn bounding box/circle on the predicted track.
*   *User Evaluation:* Visually review the generated video. The S-KeepTrack model should be capable of inferring the ball location even when the ball briefly becomes a blurred streak against a white player's shirt (a typical failure case for basic single-frame tracknets).
