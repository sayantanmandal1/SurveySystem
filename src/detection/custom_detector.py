"""
Custom Anomaly Detector using trained model
Integrates custom trained model with YOLO detection
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import sys

# Add training module to path
sys.path.append(str(Path(__file__).parent.parent))
from training.custom_trainer import AnomalyDetectionCNN

class CustomAnomalyDetector:
    """Custom anomaly detector using trained model"""
    
    def __init__(self, model_path, device=None):
        """Initialize custom detector"""
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load trained model"""
        try:
            # Create model
            self.model = AnomalyDetectionCNN(num_classes=1)
            
            # Load weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Custom model loaded from: {self.model_path}")
            
        except Exception as e:
            print(f"❌ Failed to load custom model: {e}")
            self.model = None
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        # Resize to model input size
        frame_resized = cv2.resize(frame, (224, 224))
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        frame_tensor = torch.FloatTensor(frame_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return frame_tensor.to(self.device)
    
    def detect_anomaly(self, frame):
        """Detect anomaly in frame"""
        if self.model is None:
            return 0.0
        
        try:
            # Preprocess frame
            input_tensor = self.preprocess_frame(frame)
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)
                anomaly_score = output.item()
            
            return anomaly_score
            
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            return 0.0
    
    def is_available(self):
        """Check if model is available"""
        return self.model is not None

class HybridDetector:
    """Hybrid detector combining YOLO and custom anomaly detection"""
    
    def __init__(self, yolo_detector, custom_model_path=None):
        """Initialize hybrid detector"""
        self.yolo_detector = yolo_detector
        self.custom_detector = None
        
        if custom_model_path and Path(custom_model_path).exists():
            self.custom_detector = CustomAnomalyDetector(custom_model_path)
    
    def detect(self, frame):
        """Enhanced detection with both YOLO and custom model"""
        # Get YOLO detections
        yolo_detections = self.yolo_detector.detect(frame)
        
        # Get custom anomaly score
        anomaly_score = 0.0
        if self.custom_detector and self.custom_detector.is_available():
            anomaly_score = self.custom_detector.detect_anomaly(frame)
        
        # Enhance detections with anomaly scores
        enhanced_detections = []
        for detection in yolo_detections:
            enhanced_detection = detection.copy()
            enhanced_detection['anomaly_score'] = anomaly_score
            
            # Adjust confidence based on anomaly score
            if anomaly_score > 0.5:  # High anomaly score
                enhanced_detection['confidence'] *= (1 + anomaly_score * 0.5)
                enhanced_detection['is_anomalous'] = True
            else:
                enhanced_detection['is_anomalous'] = False
            
            enhanced_detections.append(enhanced_detection)
        
        return enhanced_detections, anomaly_score
    
    def has_custom_model(self):
        """Check if custom model is available"""
        return self.custom_detector is not None and self.custom_detector.is_available()