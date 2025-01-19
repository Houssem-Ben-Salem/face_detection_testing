import numpy as np
from facenet_pytorch import MTCNN as FacenetMTCNN
import torch
from ..base import BaseFaceDetector

class MTCNNFaceDetector(BaseFaceDetector):
    def __init__(self, config):
        super().__init__(name="MTCNN")
        self.config = config
        self.detector = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def initialize(self):
        """Initialize the MTCNN face detector"""
        self.detector = FacenetMTCNN(
            keep_all=True,  # Keep all detected faces
            min_face_size=self.config['parameters'].get('min_face_size', 20),
            thresholds=self.config['parameters'].get('thresholds', [0.6, 0.7, 0.7]),
            device=self.device
        )
        self.is_initialized = True
        
    def detect(self, frame: np.ndarray) -> list:
        """
        Detect faces using MTCNN
        
        Args:
            frame: numpy array of shape (H, W, 3) in BGR format
            
        Returns:
            list of detected faces, each face is a dict with:
                - bbox: tuple of (x1, y1, x2, y2)
                - confidence: detection confidence score
                - landmarks: facial landmarks (if available)
        """
        if not self.is_initialized:
            self.initialize()
            
        # Convert BGR to RGB
        rgb_frame = frame[..., ::-1]
        
        # Detect faces
        try:
            # MTCNN expects RGB images
            boxes, probs, landmarks = self.detector.detect(rgb_frame, landmarks=True)
            
            # Handle case where no faces are detected
            if boxes is None:
                return []
            
            # Convert to our standard format
            results = []
            for i in range(len(boxes)):
                box = boxes[i]
                prob = probs[i] if probs is not None else None
                marks = landmarks[i] if landmarks is not None else None
                
                # Convert box coordinates to integers
                x1, y1, x2, y2 = map(int, box.tolist())
                
                result = {
                    'bbox': (x1, y1, x2, y2),
                    'confidence': float(prob) if prob is not None else None
                }
                
                # Add landmarks if available
                if marks is not None:
                    # Convert landmarks from tensor to list of tuples
                    landmarks_list = []
                    for j in range(marks.shape[0]):
                        x, y = marks[j]
                        landmarks_list.append((int(x), int(y)))
                    result['landmarks'] = landmarks_list
                
                results.append(result)
                
            return results
            
        except Exception as e:
            print(f"Error in MTCNN detection: {str(e)}")
            return []

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame before detection"""
        # MTCNN works better with RGB
        return frame[..., ::-1]  # BGR to RGB