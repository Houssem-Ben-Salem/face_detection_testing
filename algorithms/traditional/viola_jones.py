import cv2
import numpy as np
from pathlib import Path
from ..base import BaseFaceDetector

class ViolaJonesFaceDetector(BaseFaceDetector):
    def __init__(self, config):
        super().__init__(name="Viola-Jones")
        self.config = config
        self.face_cascade = None
        
    def initialize(self):
        """Initialize the Haar Cascade classifier"""
        cascade_path = cv2.data.haarcascades + self.config['parameters']['cascade_path']
        if not Path(cascade_path).exists():
            raise FileNotFoundError(f"Cascade file not found: {cascade_path}")
            
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.is_initialized = True
        
    def detect(self, frame: np.ndarray) -> list:
        """
        Detect faces using Viola-Jones algorithm
        
        Args:
            frame: numpy array of shape (H, W, 3) in BGR format
            
        Returns:
            list of detected faces, each face is a dict with:
                - bbox: tuple of (x1, y1, x2, y2)
                - confidence: not available for Viola-Jones
        """
        if not self.is_initialized:
            self.initialize()
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.config['parameters']['scale_factor'],
            minNeighbors=self.config['parameters']['min_neighbors'],
            minSize=tuple(self.config['parameters']['min_size'])
        )
        
        # Convert to our standard format
        detections = []
        for (x, y, w, h) in faces:
            detections.append({
                'bbox': (x, y, x + w, y + h),
                'confidence': None  # Viola-Jones doesn't provide confidence scores
            })
            
        return detections