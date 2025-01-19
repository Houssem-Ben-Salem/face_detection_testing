import dlib
import numpy as np
from ..base import BaseFaceDetector

class HOGSVMFaceDetector(BaseFaceDetector):
    def __init__(self, config):
        super().__init__(name="HOG + SVM")
        self.config = config
        self.detector = None
        
    def initialize(self):
        """Initialize the HOG + SVM face detector from dlib"""
        self.detector = dlib.get_frontal_face_detector()
        self.is_initialized = True
        
    def detect(self, frame: np.ndarray) -> list:
        """
        Detect faces using HOG + SVM
        
        Args:
            frame: numpy array of shape (H, W, 3) in BGR format
            
        Returns:
            list of detected faces, each face is a dict with:
                - bbox: tuple of (x1, y1, x2, y2)
                - confidence: detection confidence score
        """
        if not self.is_initialized:
            self.initialize()
            
        # Dlib works with RGB
        rgb_frame = frame[..., ::-1]  # BGR to RGB
        
        # Detect faces
        # The second parameter is the number of image pyramid layers to apply
        # More layers can help detect smaller faces but is slower
        detections = self.detector(rgb_frame, self.config.get('upsampling_layers', 1))
        
        # Convert to our standard format
        results = []
        for detection in detections:
            # Get the bounding box
            x1, y1 = detection.left(), detection.top()
            x2, y2 = detection.right(), detection.bottom()
            
            # Get confidence if available
            confidence = detection.confidence() if hasattr(detection, 'confidence') else None
            
            results.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence
            })
            
        return results