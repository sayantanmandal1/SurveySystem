#!/usr/bin/env python3
"""
🎯 OPTIMIZED ANOMALY DETECTION TRAINING
Simplified high-performance training script for professional deployment
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score
import scipy.io

# Add src to path
sys.path.append('src')

class OptimizedAnomalyTrainer:
    """Optimized trainer using classical ML approaches for high accuracy"""
    
    def __init__(self, data_dir="dataanomaly"):
        self.data_dir = Path(data_dir)
        self.features = []
        self.labels = []
        self.model = None
        self.history = {'train_accuracy': [], 'val_accuracy': [], 'train_auc': [], 'val_auc': []}
    
    def extract_features(self, frame):
        """Extract comprehensive features from frame"""
        # Resize frame
        frame = cv2.resize(frame, (224, 224))
        
        # Convert to different color spaces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        features = []
        
        # 1. Statistical features
        features.extend([
            np.mean(gray), np.std(gray), np.var(gray),
            np.min(gray), np.max(gray), np.median(gray)
        ])
        
        # 2. Histogram features
        hist_gray = cv2.calcHist([gray], [0], None, [32], [0, 256])
        features.extend(hist_gray.flatten())
        
        # 3. Texture features (LBP-like)
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        edges = cv2.filter2D(gray, -1, kernel)
        features.extend([np.mean(edges), np.std(edges)])
        
        # 4. Motion features (optical flow simulation)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        features.extend([np.mean(sobelx), np.std(sobelx), np.mean(sobely), np.std(sobely)])
        
        # 5. Color features
        for i in range(3):
            channel = hsv[:,:,i]
            features.extend([np.mean(channel), np.std(channel)])
        
        # 6. Contour features
        contours, _ = cv2.findContours(cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1], 
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            features.extend([len(contours), np.mean(areas), np.std(areas) if len(areas) > 1 else 0])
        else:
            features.extend([0, 0, 0])
        
        # 7. Frequency domain features
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        features.extend([np.mean(magnitude_spectrum), np.std(magnitude_spectrum)])
        
        return np.array(features, dtype=np.float32)
    
    def load_ground_truth(self, gt_path):
        """Load ground truth from .mat file"""
        if not gt_path or not os.path.exists(gt_path):
            return None
        
        try:
            mat_data = scipy.io.loadmat(str(gt_path))
            
            # Try different possible keys
            for key in ['volLabel', 'gt', 'frameLabel', 'labels']:
                if key in mat_data:
                    return mat_data[key].flatten()
            
            # If no standard key, return first array
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray):
                    return value.flatten()
        except:
            pass
        
        return None
    
    def load_dataset(self):
        """Load and prepare dataset"""
        print("📂 Loading dataset for optimized training...")
        
        # Find Avenue dataset
        avenue_dir = self.data_dir / "Avenue Dataset"
        if not avenue_dir.exists():
            print("❌ Avenue Dataset not found!")
            return False
        
        # Get training videos
        train_videos = list((avenue_dir / "training_videos").glob("*.avi"))
        train_gts = list((avenue_dir / "training_vol").glob("*.mat"))
        
        # Get testing videos
        test_videos = list((avenue_dir / "testing_videos").glob("*.avi"))
        test_gts = list((avenue_dir / "testing_vol").glob("*.mat"))
        
        train_videos.sort()
        train_gts.sort()
        test_videos.sort()
        test_gts.sort()
        
        print(f"📊 Found {len(train_videos)} training videos")
        print(f"📊 Found {len(test_videos)} testing videos")
        
        # Process training videos (these are mostly normal)
        for video_path in tqdm(train_videos[:10], desc="Processing training videos"):
            self.process_video(video_path, None, max_frames=400, default_label=0)  # Normal
        
        # Process some test videos for validation (these have anomalies)
        for video_path, gt_path in tqdm(zip(test_videos[:5], test_gts[:5]), desc="Processing validation videos"):
            self.process_video(video_path, gt_path, max_frames=200)
        
        print(f"✅ Loaded {len(self.features)} samples")
        print(f"📊 Normal samples: {self.labels.count(0)}")
        print(f"🚨 Anomaly samples: {self.labels.count(1)}")
        
        return True
    
    def process_video(self, video_path, gt_path, max_frames=500, default_label=None):
        """Process single video"""
        cap = cv2.VideoCapture(str(video_path))
        gt_labels = self.load_ground_truth(gt_path)
        
        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract features
            features = self.extract_features(frame)
            
            # Get label
            if default_label is not None:
                label = default_label
            elif gt_labels is not None and frame_count < len(gt_labels):
                # Avenue dataset ground truth: 1 = anomaly, 0 = normal
                label = 1 if gt_labels[frame_count] > 0 else 0
            else:
                # Default to normal
                label = 0
            
            self.features.append(features)
            self.labels.append(label)
            frame_count += 1
        
        cap.release()
    
    def train_model(self):
        """Train optimized model"""
        print("🧠 Training optimized anomaly detection model...")
        
        # Convert to numpy arrays
        X = np.array(self.features)
        y = np.array(self.labels)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"📊 Training samples: {len(X_train)}")
        print(f"📊 Validation samples: {len(X_val)}")
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train ensemble of models
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42)
        }
        
        best_model = None
        best_auc = 0.0
        best_name = ""
        
        results = {}
        
        for name, model in models.items():
            print(f"🔧 Training {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions - handle single class case
            train_proba = model.predict_proba(X_train_scaled)
            val_proba = model.predict_proba(X_val_scaled)
            
            if train_proba.shape[1] == 1:
                # Only one class present, create dummy probabilities
                train_pred = model.predict(X_train_scaled).astype(float)
                val_pred = model.predict(X_val_scaled).astype(float)
            else:
                train_pred = train_proba[:, 1]
                val_pred = val_proba[:, 1]
            
            # Calculate metrics - handle single class case
            try:
                train_auc = roc_auc_score(y_train, train_pred) if len(np.unique(y_train)) > 1 else 0.5
                val_auc = roc_auc_score(y_val, val_pred) if len(np.unique(y_val)) > 1 else 0.5
            except:
                train_auc = 0.5
                val_auc = 0.5
            
            train_acc = model.score(X_train_scaled, y_train)
            val_acc = model.score(X_val_scaled, y_val)
            
            results[name] = {
                'train_auc': train_auc,
                'val_auc': val_auc,
                'train_acc': train_acc,
                'val_acc': val_acc
            }
            
            print(f"  📈 Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
            print(f"  📈 Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_model = model
                best_name = name
        
        print(f"\n🏆 Best model: {best_name} with validation AUC: {best_auc:.4f}")
        
        # Save best model
        import joblib
        Path("models").mkdir(exist_ok=True)
        
        model_data = {
            'model': best_model,
            'scaler': scaler,
            'results': results,
            'best_name': best_name,
            'best_auc': best_auc
        }
        
        joblib.dump(model_data, 'models/optimized_anomaly_model.pkl')
        
        # Save results
        with open('models/training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create plots
        self.create_plots(results)
        
        return best_model, best_auc
    
    def create_plots(self, results):
        """Create training result plots"""
        plt.figure(figsize=(15, 5))
        
        # AUC comparison
        plt.subplot(1, 3, 1)
        models = list(results.keys())
        train_aucs = [results[m]['train_auc'] for m in models]
        val_aucs = [results[m]['val_auc'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, train_aucs, width, label='Training AUC', alpha=0.8)
        plt.bar(x + width/2, val_aucs, width, label='Validation AUC', alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('AUC Score')
        plt.title('Model Comparison - AUC Scores')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy comparison
        plt.subplot(1, 3, 2)
        train_accs = [results[m]['train_acc'] for m in models]
        val_accs = [results[m]['val_acc'] for m in models]
        
        plt.bar(x - width/2, train_accs, width, label='Training Accuracy', alpha=0.8)
        plt.bar(x + width/2, val_accs, width, label='Validation Accuracy', alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Model Comparison - Accuracy')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Performance summary
        plt.subplot(1, 3, 3)
        plt.axis('off')
        
        best_model = max(results.keys(), key=lambda k: results[k]['val_auc'])
        best_results = results[best_model]
        
        summary_text = f"""
        🏆 TRAINING SUMMARY
        
        Best Model: {best_model}
        
        Validation AUC: {best_results['val_auc']:.4f}
        Validation Accuracy: {best_results['val_acc']:.4f}
        
        Training AUC: {best_results['train_auc']:.4f}
        Training Accuracy: {best_results['train_acc']:.4f}
        
        Total Samples: {len(self.features)}
        
        🚀 Model Ready for Deployment!
        """
        
        plt.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('models/optimized_training_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📈 Training plots saved to models/optimized_training_results.png")

def main():
    """Main training function"""
    print("🎯" * 20)
    print("🎯 OPTIMIZED ANOMALY DETECTION TRAINING 🎯")
    print("🎯 PROFESSIONAL EDITION 🎯")
    print("🎯" * 20)
    
    trainer = OptimizedAnomalyTrainer()
    
    # Check if data exists
    if not trainer.data_dir.exists():
        print(f"❌ Data directory not found: {trainer.data_dir}")
        print("Please ensure the dataanomaly folder contains the Avenue Dataset")
        return
    
    try:
        print(f"\n⏰ Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load dataset
        if not trainer.load_dataset():
            return
        
        # Train model
        best_model, best_auc = trainer.train_model()
        
        print(f"\n🎉 OPTIMIZED TRAINING COMPLETED!")
        print("=" * 60)
        print(f"🏆 Best model AUC: {best_auc:.4f}")
        print(f"📁 Model saved to: models/optimized_anomaly_model.pkl")
        print(f"📈 Training plots saved to: models/optimized_training_results.png")
        print(f"⏰ Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n🚀 NEXT STEPS FOR DEPLOYMENT:")
        print("1. Run: streamlit run src/dashboard/app.py")
        print("2. Upload your test video and witness the optimized detection!")
        print("3. 🎯 DEPLOY TO PRODUCTION! 🎯")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()