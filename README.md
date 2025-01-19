# Face Detection Algorithm Testing Suite

A comprehensive testing platform for evaluating and comparing different face detection algorithms under various conditions.

![image](https://github.com/user-attachments/assets/7fb16a19-2f04-4c38-8859-8fbffb8de0f0)

## Authors
- Ben Salem Houssem
- Sauvé Catherine

## Overview

This project provides a user-friendly interface for testing and comparing various face detection algorithms under different conditions. It includes both traditional and modern deep learning-based approaches, allowing users to evaluate algorithm performance across different scenarios such as pose variations and lighting conditions.

## Features

### Implemented Algorithms
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
- **Pose Variations**
  - Yaw: -90° to +90°
  - Pitch: -45° to +45°
  - Roll: -30° to +30°

- **Lighting Conditions**
  - Illumination: 1-1000 lux
  - Direction: Multi-directional
  - Color temperature: 2000K-7000K

### Key Features
- Real-time webcam testing
- Video file processing
- Side-by-side comparison view
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
├── requirements.txt       # Project dependencies
├── config/
│   └── algorithms.yaml    # Algorithm configurations
├── algorithms/
│   ├── __init__.py
│   ├── base.py           # Base algorithm class
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
    ├── variations.py      # Pose and lighting variations
    └── __init__.py
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:7860
```

3. Select desired options:
   - Choose an algorithm from the dropdown menu
   - Select input method (webcam or video upload)
   - Apply variations if desired
   - Adjust performance settings as needed

## Algorithm Comparison

The application includes a comprehensive reference guide comparing algorithms across multiple dimensions:
- Pose Variations Support
- Lighting Conditions Handling
- Occlusion Support
- Performance Characteristics
- Special Features

Each algorithm is rated on various aspects with detailed explanations of their strengths and limitations.

## Performance Considerations

- Frame skipping can be adjusted to optimize performance
- GPU acceleration is available for supported algorithms
- Memory usage varies significantly between algorithms
- Processing speed depends on the selected algorithm and input resolution

## Acknowledgments

- OpenCV team for Viola-Jones implementation
- DLib developers for HOG+SVM implementation
- Google for MediaPipe framework
- InsightFace team for RetinaFace implementation
- Ultralytics for YOLOv8
- All other open-source contributors

## Citation

If you use this project in your research, please cite:
```
@software{face_detection_testing,
  author = {Ben Salem, Houssem and Sauvé, Catherine},
  title = {Face Detection Algorithm Testing Suite},
  year = {2025},
  url = {https://github.com/yourusername/face-detection-testing}
}
```
