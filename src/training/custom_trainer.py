#!/usr/bin/env python3
"""
Custom Anomaly Detection Model Trainer
Trains a specialized model on Avenue and UCSD datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve
import scipy.io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AnomalyDataset(Dataset):
    """Custom dataset for anomaly detection"""
    
    def __init__(self, video_paths, ground_truth_paths, transform=None, max_frames=1000):
        self.video_paths = video_paths
        self.ground_truth_paths = ground_truth_paths
        self.transform = transform
        self.max_frames = max_frames
        
        # Load all data
        self.frames = []
        self.labels = []
        self.load_data()
    
    def load_data(self):
        """Load video frames and labels"""
        print("📥 Loading dataset...")
        
        for video_path, gt_path in tqdm(zip(self.video_paths, self.ground_truth_paths), 
                                       desc="Loading videos"):
            # Load video
            cap = cv2.VideoCapture(video_path)
            
            # Load ground truth
            gt_labels = self.load_ground_truth(gt_path)
            
            frame_count = 0
            while frame_count < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                
                # Get label
                if gt_labels is not None and frame_count < len(gt_labels):
                    label = 1 if gt_labels[frame_count] > 0 else 0
                else:
                    label = 0  # Normal frame
                
                self.frames.append(frame)
                self.labels.append(label)
                frame_count += 1
            
            cap.release()
        
        print(f"✅ Loaded {len(self.frames)} frames")
        print(f"📊 Normal frames: {self.labels.count(0)}")
        print(f"🚨 Anomaly frames: {self.labels.count(1)}")
    
    def load_ground_truth(self, gt_path):
        """Load ground truth from .mat file"""
        if not gt_path or not os.path.exists(gt_path):
            return None
        
        try:
            mat_data = scipy.io.loadmat(gt_path)
            
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
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = torch.FloatTensor(self.frames[idx]).permute(2, 0, 1)
        label = torch.FloatTensor([self.labels[idx]])
        return frame, label

class AnomalyDetectionCNN(nn.Module):
    """Custom CNN for anomaly detection"""
    
    def __init__(self, num_classes=1):
        super(AnomalyDetectionCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CustomAnomalyTrainer:
    """Custom trainer for anomaly detection"""
    
    def __init__(self, data_dir="dataanomaly"):
        self.data_dir = Path(data_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    
    def prepare_data(self):
        """Prepare training and validation data"""
        print("📂 Preparing dataset...")
        
        # Avenue dataset paths
        avenue_train_videos = list((self.data_dir / "Avenue Dataset" / "training_videos").glob("*.avi"))
        avenue_train_gt = list((self.data_dir / "Avenue Dataset" / "training_vol").glob("*.mat"))
        
        avenue_test_videos = list((self.data_dir / "Avenue Dataset" / "testing_videos").glob("*.avi"))
        avenue_test_gt = list((self.data_dir / "Avenue Dataset" / "testing_vol").glob("*.mat"))
        
        # Sort to ensure matching
        avenue_train_videos.sort()
        avenue_train_gt.sort()
        avenue_test_videos.sort()
        avenue_test_gt.sort()
        
        print(f"📊 Found {len(avenue_train_videos)} training videos")
        print(f"📊 Found {len(avenue_test_videos)} testing videos")
        
        # Create datasets
        train_dataset = AnomalyDataset(
            [str(p) for p in avenue_train_videos[:12]],  # Use first 12 for training
            [str(p) for p in avenue_train_gt[:12]],
            max_frames=500  # Limit frames per video
        )
        
        val_dataset = AnomalyDataset(
            [str(p) for p in avenue_train_videos[12:]],  # Use remaining for validation
            [str(p) for p in avenue_train_gt[12:]],
            max_frames=300
        )
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
        
        print(f"✅ Training samples: {len(train_dataset)}")
        print(f"✅ Validation samples: {len(val_dataset)}")
    
    def create_model(self):
        """Create and initialize model"""
        print("🧠 Creating model...")
        self.model = AnomalyDetectionCNN(num_classes=1).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")
    
    def train_epoch(self, optimizer, criterion, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{running_loss/(batch_idx+1):.4f}'
            })
        
        return running_loss / len(self.train_loader)
    
    def validate_epoch(self, criterion, epoch):
        """Validate for one epoch"""
        self.model.eval()
        val_loss = 0.0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} Validation")
            
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                
                all_outputs.extend(output.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
        
        # Calculate AUC
        try:
            auc_score = roc_auc_score(all_targets, all_outputs)
        except:
            auc_score = 0.0
        
        return val_loss / len(self.val_loader), auc_score
    
    def train(self, epochs=50, learning_rate=0.001):
        """Train the model"""
        print(f"🚀 Starting training for {epochs} epochs...")
        
        # Prepare data and model
        self.prepare_data()
        self.create_model()
        
        # Setup training
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        best_auc = 0.0
        best_model_path = "models/custom_anomaly_best.pth"
        
        # Ensure models directory exists
        Path("models").mkdir(exist_ok=True)
        
        print("\n🎯 Training Progress:")
        print("=" * 80)
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(optimizer, criterion, epoch)
            
            # Validate
            val_loss, val_auc = self.validate_epoch(criterion, epoch)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_auc'].append(val_auc)
            
            # Print epoch results
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_auc': best_auc,
                    'history': self.history
                }, best_model_path)
                print(f"💾 New best model saved! AUC: {best_auc:.4f}")
        
        print("\n🎉 Training completed!")
        print(f"🏆 Best validation AUC: {best_auc:.4f}")
        
        # Save final model
        final_model_path = "models/custom_anomaly_final.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'config': {
                'epochs': epochs,
                'learning_rate': learning_rate,
                'best_auc': best_auc
            }
        }, final_model_path)
        
        # Plot training history
        self.plot_training_history()
        
        return best_model_path, final_model_path
    
    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(self.history['train_loss'], label='Training Loss', color='blue')
        plt.plot(self.history['val_loss'], label='Validation Loss', color='red')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # AUC plot
        plt.subplot(1, 3, 2)
        plt.plot(self.history['val_auc'], label='Validation AUC', color='green')
        plt.title('Validation AUC Score')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True)
        
        # Combined plot
        plt.subplot(1, 3, 3)
        plt.plot(self.history['train_loss'], label='Train Loss', alpha=0.7)
        plt.plot(self.history['val_loss'], label='Val Loss', alpha=0.7)
        plt.plot([x * max(self.history['train_loss']) for x in self.history['val_auc']], 
                label='Val AUC (scaled)', alpha=0.7)
        plt.title('Training Overview')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📈 Training plots saved to models/training_history.png")

def main():
    """Main training function"""
    print("🎯 Custom Anomaly Detection Model Training")
    print("=" * 60)
    
    trainer = CustomAnomalyTrainer()
    
    # Check if data exists
    if not trainer.data_dir.exists():
        print(f"❌ Data directory not found: {trainer.data_dir}")
        print("Please ensure the dataanomaly folder contains the Avenue Dataset")
        return
    
    # Start training
    best_model, final_model = trainer.train(epochs=30, learning_rate=0.001)
    
    print(f"\n✅ Training complete!")
    print(f"📁 Best model saved to: {best_model}")
    print(f"📁 Final model saved to: {final_model}")
    print(f"📈 Training plots saved to: models/training_history.png")

if __name__ == "__main__":
    main()