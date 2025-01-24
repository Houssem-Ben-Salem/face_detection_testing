# Face Detection Algorithm Testing Suite

A comprehensive platform for evaluating and comparing face detection algorithms under various conditions and perturbations.
![image](https://github.com/user-attachments/assets/b11c4f46-3cb0-496d-b61e-84d660f68ff7)

![image](https://github.com/user-attachments/assets/d920552e-b241-4b16-a69a-43541e5532fe)

## Authors
- Ben Salem Houssem
- Sauvé Catherine

## Overview

An interactive testing environment for evaluating face detection algorithms across different scenarios, variations, and perturbations. Supports both traditional and modern deep learning-based approaches.

## Features

### Algorithms
1. Traditional Methods
   - Viola-Jones (OpenCV Haar Cascades)
   - HOG + SVM (DLib implementation)

2. Modern ML-Based Methods
   - MediaPipe Face
   - MTCNN (Multi-task Cascaded CNN)
   - RetinaFace

3. General Object Detection
   - YOLOv8 (with custom face detection model)

### Testing Conditions

#### Pose Variations
- Yaw: -90° to +90°
- Pitch: -45° to +45°
- Roll: -30° to +30°

#### Lighting Conditions
- Brightness: 0-200% adjustment
- Contrast: 0-200% adjustment
- Color temperature: 2000K-7000K

#### Environmental Perturbations
- Gaussian Noise: Adjustable intensity
- Snow Effect: Variable density
- Custom variations combining multiple effects

### Interface Features
- Real-time webcam testing
- Video file processing with progress tracking
- Frame capture with detection comparison
- Side-by-side comparison view
- Performance optimization settings
- Comprehensive algorithm reference guide

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Houssem-Ben-Salem/face_detection_testing.git
cd face_detection_testing
```

2. Create and activate a virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows use: myenv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
```
face_detection_testing/
├── app.py                 # Main Gradio application
├── algorithm_reference.py # Algorithm comparison reference
├── requirements.txt      # Project dependencies
├── config/
│   └── algorithms.yaml   # Algorithm configurations
├── algorithms/
│   ├── __init__.py
│   ├── base.py          # Base algorithm class
│   ├── traditional/
│   │   ├── viola_jones.py
│   │   └── hog_svm.py
│   ├── ml_based/
│   │   ├── mediapipe.py
│   │   ├── mtcnn.py
│   │   └── retinaface.py
│   └── object_detection/
│       └── yolo.py
└── utils/
    ├── variations.py     # Pose, lighting, and perturbation variations
    └── __init__.py
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Access interface:
```
http://localhost:7860
```

3. Interface Options:
   - Select input (webcam/video)
   - Choose detection algorithm
   - Apply variations/perturbations
   - Adjust performance settings
   - Capture and compare frames
   - Process videos with progress tracking

## New Features

### Frame Capture
- Capture current frame during detection
- View original and processed versions
- Compare detection results

### Video Processing
- Upload and process video files
- Real-time progress tracking
- Save processed output

### Environmental Perturbations
- Gaussian noise with adjustable intensity
- Snow effect with variable density
- Combine with pose/lighting variations

## Performance Optimization

- Adjustable frame skip rate
- GPU acceleration for supported algorithms
- Memory usage optimization
- Processing speed vs. quality settings

## Algorithm Comparison

Comprehensive reference guide comparing:
- Pose variation handling
- Lighting condition adaptation
- Perturbation resistance
- Processing speed
- Memory usage
- Special features

## Acknowledgments

- OpenCV team (Viola-Jones)
- DLib developers (HOG+SVM)
- Google (MediaPipe)
- InsightFace team (RetinaFace)
- Ultralytics (YOLOv8)
- Open-source contributors

## Citation

```
@software{face_detection_testing,
  author = {Ben Salem, Houssem and Sauvé, Catherine},
  title = {Face Detection Algorithm Testing Suite},
  year = {2025},
  url = {https://github.com/yourusername/face-detection-testing}
}
