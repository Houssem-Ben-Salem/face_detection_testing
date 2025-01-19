from abc import ABC, abstractmethod
import numpy as np
import cv2

class BaseFaceDetector(ABC):
    def __init__(self, name: str):
        self.name = name
        self.is_initialized = False

    @abstractmethod
    def initialize(self):
        """Initialize the face detection model"""
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list:
        """
        Detect faces in the given frame
        
        Args:
            frame: numpy array of shape (H, W, 3) in BGR format
            
        Returns:
            list of detected faces, each face is a dict with:
                - bbox: tuple of (x1, y1, x2, y2)
                - confidence: detection confidence
                - landmarks: facial landmarks if available
        """
        pass

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Default preprocessing for input frame"""
        return frame

    def draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Draw detection results on the frame"""
        frame_with_detections = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det.get('confidence', None)
            
            # Draw bounding box
            cv2.rectangle(frame_with_detections, 
                         (int(x1), int(y1)), (int(x2), int(y2)),
                         (0, 255, 0), 2)
            
            # Draw confidence if available
            if conf is not None:
                cv2.putText(frame_with_detections, 
                           f'{conf:.2f}',
                           (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 2)
            
            # Draw landmarks if available
            if 'landmarks' in det:
                for (x, y) in det['landmarks']:
                    cv2.circle(frame_with_detections, 
                             (int(x), int(y)), 
                             2, (255, 0, 0), -1)
        
        return frame_with_detections