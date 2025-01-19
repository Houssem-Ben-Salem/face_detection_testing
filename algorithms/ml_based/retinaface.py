import numpy as np
from insightface.app import FaceAnalysis
import cv2
from ..base import BaseFaceDetector

class RetinaFaceDetector(BaseFaceDetector):
    def __init__(self, config):
        super().__init__(name="RetinaFace")
        self.config = config
        self.detector = None
        
    def initialize(self):
        """Initialize the RetinaFace detector"""
        # Initialize with detection threshold
        det_thresh = self.config['parameters'].get('det_thresh', 0.5)
        self.detector = FaceAnalysis(
            name=self.config['parameters'].get('model', 'buffalo_l'),
            providers=['CPUExecutionProvider'],
            allowed_modules=['detection']  # Only load detection module
        )
        self.detector.prepare(ctx_id=0, det_size=(640, 640), det_thresh=det_thresh)
        self.is_initialized = True
        
    def detect(self, frame: np.ndarray) -> list:
        """
        Detect faces using RetinaFace
        
        Args:
            frame: numpy array of shape (H, W, 3) in BGR format
            
        Returns:
            list of detected faces, each face is a dict with:
                - bbox: tuple of (x1, y1, x2, y2)
                - confidence: detection confidence score
                - landmarks: 5 facial landmarks (if available)
        """
        if not self.is_initialized:
            self.initialize()
            
        try:
            # RetinaFace expects BGR format, so we can use frame directly
            min_size = self.config['parameters'].get('min_face_size', 20)
            faces = self.detector.get(frame)
            
            detections = []
            for face in faces:
                bbox = face.bbox.astype(int)
                
                # Skip faces smaller than min_size
                if (bbox[2] - bbox[0]) < min_size or (bbox[3] - bbox[1]) < min_size:
                    continue
                
                # Get landmarks if available
                landmarks = None
                if hasattr(face, 'kps') and face.kps is not None:
                    landmarks = face.kps.astype(int)
                    landmarks = [(x, y) for x, y in landmarks]
                
                detections.append({
                    'bbox': (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                    'confidence': float(face.det_score),
                    'landmarks': landmarks
                })
            
            return detections
            
        except Exception as e:
            print(f"Error in RetinaFace detection: {str(e)}")
            return []
            
    def __del__(self):
        """Cleanup RetinaFace resources"""
        self.detector = None