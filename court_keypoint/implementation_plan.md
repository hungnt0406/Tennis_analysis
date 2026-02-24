# Replace Court Detector with CNN Keypoint Detector

This plan outlines the steps to replace the current raw image processing court detection logic in `tennis_analysis` with a more robust CNN-based keypoint detection model.

## Proposed Changes

### [NEW] `tennis_analysis/court_keypoint`

Create a new module to handle CNN-based court detection.

#### [NEW] [model.py](file:///Users/hungcucu/Documents/usth/computer_vision/tennis_analysis/court_keypoint/model.py)
- Port of [tracknet.py](file:///Users/hungcucu/Documents/usth/computer_vision/courtKeyPointDetection/TennisCourtDetector/tracknet.py) from the source. This contains the [BallTrackerNet](file:///Users/hungcucu/Documents/usth/computer_vision/courtKeyPointDetection/TennisCourtDetector/tracknet.py#16-85) architecture (which acts as the keypoint detector).

#### [NEW] [postprocess.py](file:///Users/hungcucu/Documents/usth/computer_vision/tennis_analysis/court_keypoint/postprocess.py)
- Port of [postprocess.py](file:///Users/hungcucu/Documents/usth/computer_vision/courtKeyPointDetection/TennisCourtDetector/postprocess.py) from the source. Handles heatmap-to-coordinate conversion and keypoint refinement.

#### [NEW] [detector.py](file:///Users/hungcucu/Documents/usth/computer_vision/tennis_analysis/court_keypoint/detector.py)
- A new `CourtKeypointDetector` class that:
  - Loads the model weights.
  - Preprocesses frames.
  - Predicts keypoints.
  - Computes the homography matrix using the `homography` logic.
  - Exposes an interface compatible with the existing pipeline (e.g., `.is_valid()`, `.H`).

### [MODIFY] [main.py](file:///Users/hungcucu/Documents/usth/computer_vision/tennis_analysis/main.py)
- Import `CourtKeypointDetector` from the new module.
- Replace `CourtLineDetector` initialization with `CourtKeypointDetector`.
- Ensure detection is triggered for the first frame and stored for subsequent frames when `static_court` is enabled.

## Verification Plan

### Automated Tests
- **Offline Detection Test**: Create a script `tennis_analysis/test_court_cnn.py` that:
  - Loads a sample frame.
  - Runs the `CourtKeypointDetector`.
  - Asserts that keypoints are detected and the homography matrix is non-None.
  - Saves an image with the detected keypoints and drawn court lines.

### Manual Verification
- Run the full pipeline on a sample video:
  ```bash
  python -m tennis_analysis.main --video [SAMPLE_VIDEO] --output output_cnn.mp4 --static-court
  ```
- Visually inspect `output_cnn.mp4` to ensure:
  - Court lines are precisely aligned with the court in the video.
  - The minimap correctly reflects player and ball positions relative to the court.
