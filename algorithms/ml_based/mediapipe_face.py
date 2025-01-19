import mediapipe as mp
import numpy as np
from ..base import BaseFaceDetector

class MediaPipeFaceDetector(BaseFaceDetector):
    def __init__(self, config):
        super().__init__(name="MediaPipe Face")
        self.config = config
        self.face_detector = None
        self.mp_face_detection = mp.solutions.face_detection
        
    def initialize(self):
        """Initialize the MediaPipe face detector"""
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=self.config['parameters'].get('model_selection', 0),
            min_detection_confidence=self.config['parameters'].get('min_detection_confidence', 0.5)
        )
        self.is_initialized = True
        
    def detect(self, frame: np.ndarray) -> list:
        """
        Detect faces using MediaPipe
        
        Args:
            frame: numpy array of shape (H, W, 3) in BGR format
            
        Returns:
            list of detected faces, each face is a dict with:
                - bbox: tuple of (x1, y1, x2, y2)
                - confidence: detection confidence score
                - landmarks: facial landmarks if available
        """
        if not self.is_initialized:
            self.initialize()
            
        # Convert BGR to RGB
        rgb_frame = frame[..., ::-1]
        height, width = frame.shape[:2]
        
        # Detect faces
        results = self.face_detector.process(rgb_frame)
        detections = []
        
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x1 = max(0, int(bbox.xmin * width))
                y1 = max(0, int(bbox.ymin * height))
                x2 = min(width, int((bbox.xmin + bbox.width) * width))
                y2 = min(height, int((bbox.ymin + bbox.height) * height))
                
                # Get keypoints
                landmarks = []
                if hasattr(detection.location_data, 'relative_keypoints'):
                    for keypoint in detection.location_data.relative_keypoints:
                        x = int(keypoint.x * width)
                        y = int(keypoint.y * height)
                        landmarks.append((x, y))
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': detection.score[0],
                    'landmarks': landmarks if landmarks else None
                })
        
        return detections
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if self.face_detector is not None:
            self.face_detector.close()