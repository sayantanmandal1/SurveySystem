"""
Balanced Anomaly Detector using the trained balanced model
High-performance detection for professional deployment
"""

import numpy as np
import cv2
import joblib
from pathlib import Path
import sys

class BalancedAnomalyDetector:
    """Balanced anomaly detector using trained model"""
    
    def __init__(self, model_path="models/balanced_anomaly_model.pkl"):
        """Initialize balanced detector"""
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.model_info = None
        self.load_model()
    
    def load_model(self):
        """Load trained model"""
        try:
            if not Path(self.model_path).exists():
                print(f"❌ Model not found: {self.model_path}")
                return False
            
            # Load model data
            self.model_info = joblib.load(self.model_path)
            self.model = self.model_info['model']
            self.scaler = self.model_info['scaler']
            
            print(f"✅ Balanced model loaded: {self.model_info['best_name']}")
            print(f"📊 Model score: {self.model_info['best_score']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load balanced model: {e}")
            return False
    
    def extract_comprehensive_features(self, frame):
        """Extract comprehensive features from frame (same as training)"""
        # Resize frame
        frame = cv2.resize(frame, (224, 224))
        
        # Convert to different color spaces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        features = []
        
        # 1. Statistical features
        features.extend([
            np.mean(gray), np.std(gray), np.var(gray),
            np.min(gray), np.max(gray), np.median(gray),
            np.percentile(gray, 25), np.percentile(gray, 75)
        ])
        
        # 2. Histogram features (reduced size)
        hist_gray = cv2.calcHist([gray], [0], None, [16], [0, 256])
        features.extend(hist_gray.flatten())
        
        # 3. Texture features
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        edges = cv2.filter2D(gray, -1, kernel)
        features.extend([np.mean(edges), np.std(edges), np.var(edges)])
        
        # 4. Gradient features
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        features.extend([
            np.mean(sobelx), np.std(sobelx),
            np.mean(sobely), np.std(sobely),
            np.mean(gradient_magnitude), np.std(gradient_magnitude)
        ])
        
        # 5. Color features
        for i in range(3):
            channel = hsv[:,:,i]
            features.extend([np.mean(channel), np.std(channel)])
        
        # 6. Contour features
        contours, _ = cv2.findContours(cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1], 
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            perimeters = [cv2.arcLength(c, True) for c in contours]
            features.extend([
                len(contours), 
                np.mean(areas), np.std(areas) if len(areas) > 1 else 0,
                np.mean(perimeters), np.std(perimeters) if len(perimeters) > 1 else 0
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # 7. Frequency domain features
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        features.extend([
            np.mean(magnitude_spectrum), np.std(magnitude_spectrum),
            np.max(magnitude_spectrum), np.min(magnitude_spectrum)
        ])
        
        # 8. Local Binary Pattern-like features
        lbp_like = np.zeros_like(gray)
        for i in range(1, gray.shape[0]-1):
            for j in range(1, gray.shape[1]-1):
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] > center) << 7
                code |= (gray[i-1, j] > center) << 6
                code |= (gray[i-1, j+1] > center) << 5
                code |= (gray[i, j+1] > center) << 4
                code |= (gray[i+1, j+1] > center) << 3
                code |= (gray[i+1, j] > center) << 2
                code |= (gray[i+1, j-1] > center) << 1
                code |= (gray[i, j-1] > center) << 0
                lbp_like[i, j] = code
        
        features.extend([np.mean(lbp_like), np.std(lbp_like)])
        
        return np.array(features, dtype=np.float32)
    
    def detect_anomaly(self, frame):
        """Detect anomaly in frame"""
        if self.model is None or self.scaler is None:
            return 0.0
        
        try:
            # Extract features
            features = self.extract_comprehensive_features(frame)
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get prediction
            if hasattr(self.model, 'predict_proba'):
                # Standard classifier
                anomaly_score = self.model.predict_proba(features_scaled)[0, 1]
            elif hasattr(self.model, 'decision_function'):
                # Isolation Forest
                decision = self.model.decision_function(features_scaled)[0]
                # Convert to probability (higher is more anomalous)
                anomaly_score = 1 / (1 + np.exp(decision))  # Sigmoid transformation
            else:
                # Fallback
                anomaly_score = float(self.model.predict(features_scaled)[0])
            
            return float(anomaly_score)
            
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            return 0.0
    
    def is_available(self):
        """Check if model is available"""
        return self.model is not None and self.scaler is not None
    
    def get_model_info(self):
        """Get model information"""
        if self.model_info:
            return {
                'name': self.model_info.get('best_name', 'Unknown'),
                'score': self.model_info.get('best_score', 0.0),
                'feature_dim': self.model_info.get('feature_dim', 0)
            }
        return {'name': 'Unknown', 'score': 0.0, 'feature_dim': 0}

class EnhancedHybridDetector:
    """Enhanced hybrid detector combining YOLO and balanced anomaly detection"""
    
    def __init__(self, yolo_detector, balanced_model_path="models/balanced_anomaly_model.pkl"):
        """Initialize enhanced hybrid detector"""
        self.yolo_detector = yolo_detector
        self.balanced_detector = BalancedAnomalyDetector(balanced_model_path)
        
        # Print initialization status
        if self.balanced_detector.is_available():
            info = self.balanced_detector.get_model_info()
            print(f"🏆 Enhanced detector ready with {info['name']} (AUC: {info['score']:.4f})")
        else:
            print("⚠️ Enhanced detector using YOLO only")
    
    def detect(self, frame):
        """Enhanced detection with both YOLO and balanced anomaly detection"""
        # Get YOLO detections
        yolo_detections = self.yolo_detector.detect(frame)
        
        # Get balanced anomaly score
        anomaly_score = 0.0
        if self.balanced_detector.is_available():
            anomaly_score = self.balanced_detector.detect_anomaly(frame)
        
        # Enhance detections with anomaly scores
        enhanced_detections = []
        for detection in yolo_detections:
            enhanced_detection = detection.copy()
            enhanced_detection['anomaly_score'] = anomaly_score
            
            # Adjust confidence based on anomaly score
            if anomaly_score > 0.6:  # High anomaly threshold
                enhanced_detection['confidence'] *= (1 + anomaly_score * 0.8)
                enhanced_detection['is_anomalous'] = True
                enhanced_detection['anomaly_level'] = 'HIGH'
            elif anomaly_score > 0.4:  # Medium anomaly threshold
                enhanced_detection['confidence'] *= (1 + anomaly_score * 0.4)
                enhanced_detection['is_anomalous'] = True
                enhanced_detection['anomaly_level'] = 'MEDIUM'
            else:
                enhanced_detection['is_anomalous'] = False
                enhanced_detection['anomaly_level'] = 'LOW'
            
            enhanced_detections.append(enhanced_detection)
        
        # If no YOLO detections but high anomaly score, create a general anomaly detection
        if not enhanced_detections and anomaly_score > 0.5:
            h, w = frame.shape[:2]
            enhanced_detections.append({
                'class': 'anomaly',
                'confidence': anomaly_score,
                'bbox': [w//4, h//4, 3*w//4, 3*h//4],  # Center region
                'anomaly_score': anomaly_score,
                'is_anomalous': True,
                'anomaly_level': 'HIGH' if anomaly_score > 0.7 else 'MEDIUM'
            })
        
        return enhanced_detections, anomaly_score
    
    def has_balanced_model(self):
        """Check if balanced model is available"""
        return self.balanced_detector.is_available()
    
    def get_model_info(self):
        """Get model information"""
        return self.balanced_detector.get_model_info()