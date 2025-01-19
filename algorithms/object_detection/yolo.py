from ultralytics import YOLO
import numpy as np
import torch
from pathlib import Path
from ..base import BaseFaceDetector

class YOLOFaceDetector(BaseFaceDetector):
    def __init__(self, config):
        super().__init__(name="YOLOv8-Face")
        self.config = config
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def initialize(self):
        """Initialize the YOLOv8 face detector"""
        model_path = self.config['parameters'].get('model_path', 'yolov8m-face.pt')
        if not Path(model_path).exists():
            raise FileNotFoundError(f"YOLO model not found at: {model_path}")
            
        self.model = YOLO(model_path)
        self.is_initialized = True
        
    def detect(self, frame: np.ndarray) -> list:
        """
        Detect faces using YOLOv8
        
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
            
        try:
            # Run inference
            results = self.model.predict(
                frame,
                conf=self.config['parameters'].get('conf_thresh', 0.25),
                iou=self.config['parameters'].get('iou_thresh', 0.45),
                device=self.device,
                verbose=False
            )
            
            detections = []
            
            # Process detections if any faces were found
            if len(results) > 0 and len(results[0].boxes) > 0:
                result = results[0]  # Get first image result
                boxes = result.boxes
                
                # Convert detections to our standard format
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    detection = {
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(box.conf[0])
                    }
                    
                    # Add landmarks if available (depends on your model)
                    if hasattr(box, 'keypoints') and box.keypoints is not None:
                        landmarks = box.keypoints.cpu().numpy()[0]
                        detection['landmarks'] = [(int(x), int(y)) for x, y in landmarks]
                    
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Error in YOLO detection: {str(e)}")
            return []
    
    def __del__(self):
        """Cleanup YOLO resources"""
        self.model = None
        torch.cuda.empty_cache()