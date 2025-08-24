"""
YOLO-based Object and Person Detection
Uses YOLOv5/YOLOv8 for real-time detection
"""

import torch
import cv2
import numpy as np
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, config):
        """Initialize YOLO detector"""
        self.config = config
        self.model = YOLO(config.get('model_path', 'yolov8n.pt'))
        self.confidence = config.get('confidence', 0.5)
        self.iou_threshold = config.get('iou_threshold', 0.45)
        
        # Classes we're interested in for surveillance
        self.target_classes = {
            0: 'person',
            24: 'backpack',
            25: 'umbrella',
            26: 'handbag',
            27: 'tie',
            28: 'suitcase'
        }
        
    def detect(self, frame):
        """Detect objects in frame"""
        results = self.model(frame, conf=self.confidence, iou=self.iou_threshold)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Get class and confidence
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Only keep target classes
                    if cls in self.target_classes:
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'class': self.target_classes[cls],
                            'class_id': cls,
                            'confidence': conf,
                            'center': [(x1 + x2) // 2, (y1 + y2) // 2],
                            'area': (x2 - x1) * (y2 - y1)
                        }
                        detections.append(detection)
        
        return detections
    
    def get_person_detections(self, detections):
        """Filter for person detections only"""
        return [d for d in detections if d['class'] == 'person']
    
    def get_object_detections(self, detections):
        """Filter for object detections (bags, etc.)"""
        return [d for d in detections if d['class'] != 'person']