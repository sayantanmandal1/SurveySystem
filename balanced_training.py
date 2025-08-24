#!/usr/bin/env python3
"""
🎯 BALANCED ANOMALY DETECTION TRAINING
High-performance training with balanced dataset for professional deployment
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
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import scipy.io
import joblib

# Add src to path
sys.path.append('src')

class BalancedAnomalyTrainer:
    """Balanced trainer with synthetic anomaly generation"""
    
    def __init__(self, data_dir="dataanomaly"):
        self.data_dir = Path(data_dir)
        self.normal_features = []
        self.anomaly_features = []
        self.model = None
        self.scaler = None
    
    def extract_comprehensive_features(self, frame):
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
    
    def load_normal_data(self):
        """Load normal data from training videos"""
        print("📂 Loading normal data from training videos...")
        
        avenue_dir = self.data_dir / "Avenue Dataset"
        if not avenue_dir.exists():
            print("❌ Avenue Dataset not found!")
            return False
        
        # Get training videos (these are mostly normal)
        train_videos = list((avenue_dir / "training_videos").glob("*.avi"))
        train_videos.sort()
        
        print(f"📊 Found {len(train_videos)} training videos")
        
        # Process training videos
        for video_path in tqdm(train_videos[:12], desc="Processing normal videos"):
            self.process_normal_video(video_path, max_frames=300)
        
        print(f"✅ Loaded {len(self.normal_features)} normal samples")
        return True
    
    def process_normal_video(self, video_path, max_frames=300):
        """Process normal video"""
        cap = cv2.VideoCapture(str(video_path))
        
        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip some frames for diversity
            if frame_count % 3 == 0:
                features = self.extract_comprehensive_features(frame)
                self.normal_features.append(features)
            
            frame_count += 1
        
        cap.release()
    
    def generate_synthetic_anomalies(self):
        """Generate synthetic anomalies from normal data"""
        print("🔧 Generating synthetic anomalies...")
        
        normal_array = np.array(self.normal_features)
        
        # Method 1: Add noise
        noise_anomalies = []
        for i in range(len(self.normal_features) // 4):
            idx = np.random.randint(0, len(self.normal_features))
            noisy = normal_array[idx].copy()
            # Add significant noise
            noise = np.random.normal(0, np.std(noisy) * 0.5, noisy.shape)
            noisy += noise
            noise_anomalies.append(noisy)
        
        # Method 2: Extreme values
        extreme_anomalies = []
        for i in range(len(self.normal_features) // 4):
            idx = np.random.randint(0, len(self.normal_features))
            extreme = normal_array[idx].copy()
            # Make some features extreme
            extreme_indices = np.random.choice(len(extreme), len(extreme)//3, replace=False)
            for ei in extreme_indices:
                if np.random.random() > 0.5:
                    extreme[ei] = np.max(normal_array[:, ei]) * 1.5
                else:
                    extreme[ei] = np.min(normal_array[:, ei]) * 1.5
            extreme_anomalies.append(extreme)
        
        # Method 3: Feature swapping
        swap_anomalies = []
        for i in range(len(self.normal_features) // 4):
            idx1 = np.random.randint(0, len(self.normal_features))
            idx2 = np.random.randint(0, len(self.normal_features))
            swapped = normal_array[idx1].copy()
            # Swap some features
            swap_indices = np.random.choice(len(swapped), len(swapped)//2, replace=False)
            swapped[swap_indices] = normal_array[idx2][swap_indices]
            swap_anomalies.append(swapped)
        
        # Method 4: Outlier generation using statistical methods
        outlier_anomalies = []
        for i in range(len(self.normal_features) // 4):
            outlier = np.random.normal(
                np.mean(normal_array, axis=0),
                np.std(normal_array, axis=0) * 3,  # 3 standard deviations
                normal_array.shape[1]
            )
            outlier_anomalies.append(outlier)
        
        # Combine all synthetic anomalies
        self.anomaly_features = noise_anomalies + extreme_anomalies + swap_anomalies + outlier_anomalies
        
        print(f"✅ Generated {len(self.anomaly_features)} synthetic anomalies")
    
    def train_models(self):
        """Train multiple models and select the best"""
        print("🧠 Training anomaly detection models...")
        
        # Prepare data
        X_normal = np.array(self.normal_features)
        X_anomaly = np.array(self.anomaly_features)
        
        # Create balanced dataset
        X = np.vstack([X_normal, X_anomaly])
        y = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_anomaly))])
        
        print(f"📊 Total samples: {len(X)}")
        print(f"📊 Normal samples: {len(X_normal)} ({len(X_normal)/len(X)*100:.1f}%)")
        print(f"🚨 Anomaly samples: {len(X_anomaly)} ({len(X_anomaly)/len(X)*100:.1f}%)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, 
                max_depth=15, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            ),
            'IsolationForest': IsolationForest(
                n_estimators=200,
                contamination=0.1,
                random_state=42
            )
        }
        
        results = {}
        best_model = None
        best_score = 0.0
        best_name = ""
        
        for name, model in models.items():
            print(f"🔧 Training {name}...")
            
            if name == 'IsolationForest':
                # Isolation Forest works differently
                model.fit(X_train_scaled)
                train_pred = model.decision_function(X_train_scaled)
                test_pred = model.decision_function(X_test_scaled)
                
                # Convert to probabilities (higher is more normal)
                train_pred = (train_pred - train_pred.min()) / (train_pred.max() - train_pred.min())
                test_pred = (test_pred - test_pred.min()) / (test_pred.max() - test_pred.min())
                
                # Invert for anomaly detection (higher should be more anomalous)
                train_pred = 1 - train_pred
                test_pred = 1 - test_pred
                
                # Calculate accuracy using threshold
                train_pred_binary = (train_pred > 0.5).astype(int)
                test_pred_binary = (test_pred > 0.5).astype(int)
                
                train_acc = np.mean(train_pred_binary == y_train)
                test_acc = np.mean(test_pred_binary == y_test)
            else:
                # Standard classifier
                model.fit(X_train_scaled, y_train)
                train_pred = model.predict_proba(X_train_scaled)[:, 1]
                test_pred = model.predict_proba(X_test_scaled)[:, 1]
                
                train_acc = model.score(X_train_scaled, y_train)
                test_acc = model.score(X_test_scaled, y_test)
            
            # Calculate metrics
            try:
                train_auc = roc_auc_score(y_train, train_pred)
                test_auc = roc_auc_score(y_test, test_pred)
                
                # Calculate F1 score
                test_pred_binary = (test_pred > 0.5).astype(int)
                f1 = f1_score(y_test, test_pred_binary)
                
            except Exception as e:
                print(f"Error calculating metrics for {name}: {e}")
                train_auc = test_auc = f1 = 0.0
            
            results[name] = {
                'train_auc': train_auc,
                'test_auc': test_auc,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'f1_score': f1
            }
            
            print(f"  📈 Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}")
            print(f"  📈 Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
            print(f"  📈 F1 Score: {f1:.4f}")
            
            # Select best model based on test AUC
            if test_auc > best_score:
                best_score = test_auc
                best_model = model
                best_name = name
        
        print(f"\n🏆 Best model: {best_name} with test AUC: {best_score:.4f}")
        
        # Save best model
        Path("models").mkdir(exist_ok=True)
        
        model_data = {
            'model': best_model,
            'scaler': self.scaler,
            'results': results,
            'best_name': best_name,
            'best_score': best_score,
            'feature_dim': X_train_scaled.shape[1]
        }
        
        joblib.dump(model_data, 'models/balanced_anomaly_model.pkl')
        
        # Save results
        with open('models/balanced_training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create plots
        self.create_plots(results)
        
        # Print classification report for best model
        if best_name != 'IsolationForest':
            y_pred = best_model.predict(X_test_scaled)
            print(f"\n📊 Classification Report for {best_name}:")
            print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
        
        return best_model, best_score
    
    def create_plots(self, results):
        """Create comprehensive training result plots"""
        plt.figure(figsize=(20, 10))
        
        # AUC comparison
        plt.subplot(2, 4, 1)
        models = list(results.keys())
        train_aucs = [results[m]['train_auc'] for m in models]
        test_aucs = [results[m]['test_auc'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, train_aucs, width, label='Training AUC', alpha=0.8, color='skyblue')
        plt.bar(x + width/2, test_aucs, width, label='Test AUC', alpha=0.8, color='lightcoral')
        
        plt.xlabel('Models')
        plt.ylabel('AUC Score')
        plt.title('Model Comparison - AUC Scores')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy comparison
        plt.subplot(2, 4, 2)
        train_accs = [results[m]['train_acc'] for m in models]
        test_accs = [results[m]['test_acc'] for m in models]
        
        plt.bar(x - width/2, train_accs, width, label='Training Accuracy', alpha=0.8, color='lightgreen')
        plt.bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.8, color='orange')
        
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Model Comparison - Accuracy')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # F1 Score comparison
        plt.subplot(2, 4, 3)
        f1_scores = [results[m]['f1_score'] for m in models]
        
        plt.bar(models, f1_scores, alpha=0.8, color='purple')
        plt.xlabel('Models')
        plt.ylabel('F1 Score')
        plt.title('Model Comparison - F1 Scores')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Combined metrics radar chart
        plt.subplot(2, 4, 4)
        best_model = max(results.keys(), key=lambda k: results[k]['test_auc'])
        best_results = results[best_model]
        
        metrics = ['Train AUC', 'Test AUC', 'Train Acc', 'Test Acc', 'F1 Score']
        values = [
            best_results['train_auc'],
            best_results['test_auc'],
            best_results['train_acc'],
            best_results['test_acc'],
            best_results['f1_score']
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        values += values[:1]  # Complete the circle
        angles = np.concatenate((angles, [angles[0]]))
        
        plt.polar(angles, values, 'o-', linewidth=2, color='red', alpha=0.7)
        plt.fill(angles, values, alpha=0.25, color='red')
        plt.xticks(angles[:-1], metrics)
        plt.title(f'Best Model ({best_model}) - All Metrics')
        plt.ylim(0, 1)
        
        # Data distribution
        plt.subplot(2, 4, 5)
        normal_count = len(self.normal_features)
        anomaly_count = len(self.anomaly_features)
        
        plt.pie([normal_count, anomaly_count], 
                labels=['Normal', 'Anomaly'], 
                autopct='%1.1f%%',
                colors=['lightblue', 'lightcoral'])
        plt.title('Dataset Distribution')
        
        # Feature importance (if available)
        plt.subplot(2, 4, 6)
        best_model_obj = max(results.items(), key=lambda x: x[1]['test_auc'])
        if hasattr(best_model_obj, 'feature_importances_'):
            importances = best_model_obj.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10 features
            
            plt.bar(range(10), importances[indices])
            plt.xlabel('Feature Index')
            plt.ylabel('Importance')
            plt.title('Top 10 Feature Importances')
            plt.xticks(range(10), indices)
        else:
            plt.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Feature Importance')
        
        # Performance summary
        plt.subplot(2, 4, 7)
        plt.axis('off')
        
        summary_text = f"""
        🏆 TRAINING SUMMARY
        
        Best Model: {best_model}
        
        Test AUC: {best_results['test_auc']:.4f}
        Test Accuracy: {best_results['test_acc']:.4f}
        F1 Score: {best_results['f1_score']:.4f}
        
        Normal Samples: {normal_count:,}
        Anomaly Samples: {anomaly_count:,}
        Total Samples: {normal_count + anomaly_count:,}
        
        🚀 Model Ready for Deployment!
        """
        
        plt.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        
        # Model comparison table
        plt.subplot(2, 4, 8)
        plt.axis('off')
        
        table_data = []
        for model_name in models:
            r = results[model_name]
            table_data.append([
                model_name,
                f"{r['test_auc']:.3f}",
                f"{r['test_acc']:.3f}",
                f"{r['f1_score']:.3f}"
            ])
        
        table = plt.table(cellText=table_data,
                         colLabels=['Model', 'AUC', 'Accuracy', 'F1'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        plt.title('Model Performance Summary')
        
        plt.tight_layout()
        plt.savefig('models/balanced_training_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📈 Training plots saved to models/balanced_training_results.png")

def main():
    """Main training function"""
    print("🎯" * 20)
    print("🎯 BALANCED ANOMALY DETECTION TRAINING 🎯")
    print("🎯 PROFESSIONAL EDITION 🎯")
    print("🎯" * 20)
    
    trainer = BalancedAnomalyTrainer()
    
    # Check if data exists
    if not trainer.data_dir.exists():
        print(f"❌ Data directory not found: {trainer.data_dir}")
        print("Please ensure the dataanomaly folder contains the Avenue Dataset")
        return
    
    try:
        print(f"\n⏰ Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load normal data
        if not trainer.load_normal_data():
            return
        
        # Generate synthetic anomalies
        trainer.generate_synthetic_anomalies()
        
        # Train models
        best_model, best_score = trainer.train_models()
        
        print(f"\n🎉 BALANCED TRAINING COMPLETED!")
        print("=" * 60)
        print(f"🏆 Best model score: {best_score:.4f}")
        print(f"📁 Model saved to: models/balanced_anomaly_model.pkl")
        print(f"📈 Training plots saved to: models/balanced_training_results.png")
        print(f"⏰ Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n🚀 NEXT STEPS FOR DEPLOYMENT:")
        print("1. Run: streamlit run src/dashboard/app.py")
        print("2. Upload your test video and witness the balanced detection!")
        print("3. 🎯 DEPLOY TO PRODUCTION! 🎯")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()