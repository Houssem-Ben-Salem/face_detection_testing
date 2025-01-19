import yaml
from pathlib import Path
from typing import Dict
from .traditional.viola_jones import ViolaJonesFaceDetector
from .traditional.hog_svm import HOGSVMFaceDetector
from .ml_based.mtcnn import MTCNNFaceDetector
from .ml_based.mediapipe_face import MediaPipeFaceDetector
from .ml_based.retinaface import RetinaFaceDetector
from .object_detection.yolo import YOLOFaceDetector

def load_algorithms() -> Dict:
    """Load all available algorithm implementations"""
    # Load configuration
    config_path = Path("config/algorithms.yaml")
    if not config_path.exists():
        raise FileNotFoundError("Algorithm configuration file not found")
        
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Initialize algorithms
    algorithms = {}
    
    # Add Viola-Jones
    if 'viola_jones' in config['algorithms']:
        algorithms['Viola-Jones'] = ViolaJonesFaceDetector(
            config['algorithms']['viola_jones']
        )
    
    # Add HOG + SVM
    if 'hog_svm' in config['algorithms']:
        algorithms['HOG + SVM'] = HOGSVMFaceDetector(
            config['algorithms']['hog_svm']
        )
    
    # Add MTCNN
    if 'mtcnn' in config['algorithms']:
        algorithms['MTCNN'] = MTCNNFaceDetector(
            config['algorithms']['mtcnn']
        )
    
    # Add MediaPipe
    if 'mediapipe' in config['algorithms']:
        algorithms['MediaPipe'] = MediaPipeFaceDetector(
            config['algorithms']['mediapipe']
        )
    
    # Add RetinaFace
    if 'retinaface' in config['algorithms']:
        algorithms['RetinaFace'] = RetinaFaceDetector(
            config['algorithms']['retinaface']
        )
    
    # Add YOLOv8-Face
    if 'yolov8_face' in config['algorithms']:
        algorithms['YOLOv8-Face'] = YOLOFaceDetector(
            config['algorithms']['yolov8_face']
        )
    
    return algorithms