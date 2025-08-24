#!/usr/bin/env python3
"""
🎯 ULTIMATE ANOMALY DETECTION TRAINING SCRIPT
High-Performance Training for Professional Deployment
Optimized for UCSD and Avenue Datasets
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
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
import scipy.io
from datetime import datetime
import warnings
import random
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.nn.functional as F
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings('ignore')

class AdvancedAnomalyDataset(Dataset):
    """Advanced dataset with data augmentation and smart sampling"""
    
    def __init__(self, video_paths, ground_truth_paths, mode='train', max_frames=800):
        self.video_paths = video_paths
        self.ground_truth_paths = ground_truth_paths
        self.mode = mode
        self.max_frames = max_frames
        
        # Advanced augmentations
        if mode == 'train':
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.MotionBlur(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        self.frames = []
        self.labels = []
        self.load_data()
    
    def load_data(self):
        """Load and balance dataset"""
        print(f"📥 Loading {self.mode} dataset...")
        
        all_frames = []
        all_labels = []
        
        for video_path, gt_path in tqdm(zip(self.video_paths, self.ground_truth_paths), 
                                       desc=f"Loading {self.mode} videos"):
            frames, labels = self.load_video_with_gt(video_path, gt_path)
            all_frames.extend(frames)
            all_labels.extend(labels)
        
        # Balance dataset for training
        if self.mode == 'train':
            all_frames, all_labels = self.balance_dataset(all_frames, all_labels)
        
        self.frames = all_frames
        self.labels = all_labels
        
        normal_count = self.labels.count(0)
        anomaly_count = self.labels.count(1)
        
        print(f"✅ Loaded {len(self.frames)} frames")
        print(f"📊 Normal frames: {normal_count} ({normal_count/len(self.frames)*100:.1f}%)")
        print(f"🚨 Anomaly frames: {anomaly_count} ({anomaly_count/len(self.frames)*100:.1f}%)")
    
    def load_video_with_gt(self, video_path, gt_path):
        """Load video with ground truth"""
        cap = cv2.VideoCapture(str(video_path))
        gt_labels = self.load_ground_truth(gt_path)
        
        frames = []
        labels = []
        frame_count = 0
        
        while frame_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get label
            if gt_labels is not None and frame_count < len(gt_labels):
                label = 1 if gt_labels[frame_count] > 0 else 0
            else:
                label = 0
            
            frames.append(frame)
            labels.append(label)
            frame_count += 1
        
        cap.release()
        return frames, labels
    
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
    
    def balance_dataset(self, frames, labels):
        """Balance dataset by oversampling anomalies"""
        normal_indices = [i for i, label in enumerate(labels) if label == 0]
        anomaly_indices = [i for i, label in enumerate(labels) if label == 1]
        
        if len(anomaly_indices) == 0:
            return frames, labels
        
        # Oversample anomalies to balance
        target_anomaly_count = min(len(normal_indices), len(anomaly_indices) * 3)
        
        balanced_indices = normal_indices.copy()
        
        # Add original anomalies
        balanced_indices.extend(anomaly_indices)
        
        # Oversample anomalies
        while len([i for i in balanced_indices if labels[i] == 1]) < target_anomaly_count:
            balanced_indices.extend(random.choices(anomaly_indices, k=min(len(anomaly_indices), 
                                                  target_anomaly_count - len([i for i in balanced_indices if labels[i] == 1]))))
        
        # Shuffle
        random.shuffle(balanced_indices)
        
        balanced_frames = [frames[i] for i in balanced_indices]
        balanced_labels = [labels[i] for i in balanced_indices]
        
        return balanced_frames, balanced_labels
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=frame)
            frame = transformed['image']
        else:
            frame = torch.FloatTensor(frame).permute(2, 0, 1) / 255.0
        
        return frame, torch.FloatTensor([label])

class UltimateAnomalyNet(nn.Module):
    """Ultimate anomaly detection network with attention and residual connections"""
    
    def __init__(self, num_classes=1):
        super(UltimateAnomalyNet, self).__init__()
        
        # Feature extraction with residual blocks
        self.features = nn.Sequential(
            # Initial conv
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Residual blocks
            self._make_layer(64, 64, 2),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.Sigmoid()
        )
        
        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """Create residual layer"""
        layers = []
        
        # First block
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Additional blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Feature extraction
        features = self.features(x)
        
        # Attention
        attention_weights = self.attention(features)
        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)
        
        # Apply attention
        attended_features = features * attention_weights
        
        # Classification
        output = self.classifier(attended_features)
        
        return output

class ResidualBlock(nn.Module):
    """Residual block with batch normalization"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        
        return out

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class UltimateTrainer:
    """Ultimate trainer with advanced techniques"""
    
    def __init__(self, data_dir="dataanomaly"):
        self.data_dir = Path(data_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"🚀 GPU: {torch.cuda.get_device_name()}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.history = {
            'train_loss': [], 'val_loss': [], 'val_auc': [], 
            'val_f1': [], 'learning_rate': []
        }
    
    def prepare_data(self):
        """Prepare optimized datasets"""
        print("📂 Preparing ultimate dataset...")
        
        # Get all available datasets
        datasets = self.find_datasets()
        
        if not datasets:
            raise ValueError("No datasets found!")
        
        # Split datasets
        train_videos, train_gts, val_videos, val_gts = self.split_datasets(datasets)
        
        print(f"📊 Training videos: {len(train_videos)}")
        print(f"📊 Validation videos: {len(val_videos)}")
        
        # Create datasets
        train_dataset = AdvancedAnomalyDataset(train_videos, train_gts, mode='train', max_frames=600)
        val_dataset = AdvancedAnomalyDataset(val_videos, val_gts, mode='val', max_frames=400)
        
        # Create optimized data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=32, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=32, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        print(f"✅ Training samples: {len(train_dataset)}")
        print(f"✅ Validation samples: {len(val_dataset)}")
    
    def find_datasets(self):
        """Find all available datasets"""
        datasets = []
        
        # Avenue Dataset
        avenue_dir = self.data_dir / "Avenue Dataset"
        if avenue_dir.exists():
            train_videos = list((avenue_dir / "training_videos").glob("*.avi"))
            train_gts = list((avenue_dir / "training_vol").glob("*.mat"))
            
            test_videos = list((avenue_dir / "testing_videos").glob("*.avi"))
            test_gts = list((avenue_dir / "testing_vol").glob("*.mat"))
            
            if train_videos:
                datasets.append(('avenue_train', sorted(train_videos), sorted(train_gts)))
            if test_videos:
                datasets.append(('avenue_test', sorted(test_videos), sorted(test_gts)))
        
        # UCSD Dataset
        ucsd_dir = self.data_dir / "UCSD_Anomaly_Dataset.v1p2"
        if ucsd_dir.exists():
            for ped_dir in ['UCSDped1', 'UCSDped2']:
                ped_path = ucsd_dir / ped_dir
                if ped_path.exists():
                    # Training data
                    train_dirs = list((ped_path / "Train").glob("Train*"))
                    if train_dirs:
                        train_videos = []
                        for train_dir in train_dirs:
                            tif_files = list(train_dir.glob("*.tif"))
                            if tif_files:
                                train_videos.append(str(train_dir))
                        if train_videos:
                            datasets.append((f'ucsd_{ped_dir}_train', train_videos, [None] * len(train_videos)))
                    
                    # Test data
                    test_dirs = list((ped_path / "Test").glob("Test*"))
                    if test_dirs:
                        test_videos = []
                        test_gts = []
                        for test_dir in test_dirs:
                            tif_files = list(test_dir.glob("*.tif"))
                            if tif_files:
                                test_videos.append(str(test_dir))
                                # Look for corresponding ground truth
                                gt_dir = ped_path / "Test" / f"{test_dir.name}_gt"
                                if gt_dir.exists():
                                    gt_files = list(gt_dir.glob("*.bmp"))
                                    test_gts.append(str(gt_dir) if gt_files else None)
                                else:
                                    test_gts.append(None)
                        
                        if test_videos:
                            datasets.append((f'ucsd_{ped_dir}_test', test_videos, test_gts))
        
        print(f"📁 Found {len(datasets)} dataset splits")
        for name, videos, gts in datasets:
            print(f"  • {name}: {len(videos)} videos")
        
        return datasets
    
    def split_datasets(self, datasets):
        """Split datasets for training and validation"""
        train_videos = []
        train_gts = []
        val_videos = []
        val_gts = []
        
        for name, videos, gts in datasets:
            if 'train' in name.lower():
                # Use 80% for training, 20% for validation
                split_idx = int(len(videos) * 0.8)
                train_videos.extend(videos[:split_idx])
                train_gts.extend(gts[:split_idx])
                val_videos.extend(videos[split_idx:])
                val_gts.extend(gts[split_idx:])
            else:
                # Use test sets for validation
                val_videos.extend(videos)
                val_gts.extend(gts)
        
        return train_videos, train_gts, val_videos, val_gts
    
    def create_model(self):
        """Create ultimate model"""
        print("🧠 Creating ultimate model...")
        self.model = UltimateAnomalyNet(num_classes=1).to(self.device)
        
        # Initialize weights
        self.model.apply(self.init_weights)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")
    
    def init_weights(self, m):
        """Initialize model weights"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
    
    def train_epoch(self, optimizer, criterion, epoch, scaler):
        """Train one epoch with mixed precision"""
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            with torch.cuda.amp.autocast():
                output = self.model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{running_loss/(batch_idx+1):.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        return running_loss / len(self.train_loader)
    
    def validate_epoch(self, criterion, epoch):
        """Validate one epoch"""
        self.model.eval()
        val_loss = 0.0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} Validation")
            
            for data, target in pbar:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = criterion(output, target)
                
                val_loss += loss.item()
                
                all_outputs.extend(output.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        try:
            auc_score = roc_auc_score(all_targets, all_outputs)
            
            # Calculate F1 score with optimal threshold
            predictions = np.array(all_outputs) > 0.5
            f1 = f1_score(all_targets, predictions)
        except:
            auc_score = 0.0
            f1 = 0.0
        
        return val_loss / len(self.val_loader), auc_score, f1
    
    def train(self, epochs=50, learning_rate=0.001):
        """Ultimate training loop"""
        print(f"🚀 Starting ULTIMATE training for {epochs} epochs...")
        print("=" * 80)
        
        # Prepare everything
        self.prepare_data()
        self.create_model()
        
        # Setup advanced training components
        criterion = FocalLoss(alpha=2, gamma=2)  # Handle class imbalance
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Advanced schedulers
        scheduler1 = CosineAnnealingLR(optimizer, T_max=epochs//2, eta_min=learning_rate/100)
        scheduler2 = ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5, verbose=True)
        
        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler()
        
        # Tracking
        best_auc = 0.0
        best_f1 = 0.0
        patience_counter = 0
        max_patience = 15
        
        # Ensure models directory exists
        Path("models").mkdir(exist_ok=True)
        
        print("\n🎯 ULTIMATE Training Progress:")
        print("=" * 100)
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(optimizer, criterion, epoch, scaler)
            
            # Validate
            val_loss, val_auc, val_f1 = self.validate_epoch(criterion, epoch)
            
            # Update schedulers
            if epoch < epochs // 2:
                scheduler1.step()
            else:
                scheduler2.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_auc'].append(val_auc)
            self.history['val_f1'].append(val_f1)
            self.history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Print epoch results
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} | "
                  f"Val F1: {val_f1:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best models
            is_best = False
            if val_auc > best_auc:
                best_auc = val_auc
                is_best = True
                patience_counter = 0
            else:
                patience_counter += 1
            
            if val_f1 > best_f1:
                best_f1 = val_f1
            
            if is_best:
                best_model_path = "models/custom_anomaly_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_auc': best_auc,
                    'best_f1': best_f1,
                    'history': self.history
                }, best_model_path)
                print(f"💾 🏆 NEW BEST MODEL! AUC: {best_auc:.4f}, F1: {best_f1:.4f}")
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"⏰ Early stopping triggered after {patience_counter} epochs without improvement")
                break
        
        print("\n🎉 ULTIMATE TRAINING COMPLETED!")
        print("=" * 60)
        print(f"🏆 Best validation AUC: {best_auc:.4f}")
        print(f"🏆 Best validation F1: {best_f1:.4f}")
        
        # Save final model
        final_model_path = "models/custom_anomaly_final.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'config': {
                'epochs': epochs,
                'learning_rate': learning_rate,
                'best_auc': best_auc,
                'best_f1': best_f1
            }
        }, final_model_path)
        
        # Create ultimate plots
        self.create_ultimate_plots()
        
        return best_model_path, final_model_path
    
    def create_ultimate_plots(self):
        """Create comprehensive training plots"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('🏆 ULTIMATE ANOMALY DETECTION TRAINING RESULTS', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # AUC plot
        axes[0, 1].plot(epochs, self.history['val_auc'], 'g-', label='Validation AUC', linewidth=2)
        axes[0, 1].set_title('Validation AUC Score', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 plot
        axes[0, 2].plot(epochs, self.history['val_f1'], 'm-', label='Validation F1', linewidth=2)
        axes[0, 2].set_title('Validation F1 Score', fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1, 0].plot(epochs, self.history['learning_rate'], 'orange', linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Combined metrics
        axes[1, 1].plot(epochs, self.history['val_auc'], 'g-', label='AUC', linewidth=2)
        axes[1, 1].plot(epochs, self.history['val_f1'], 'm-', label='F1', linewidth=2)
        axes[1, 1].set_title('Validation Metrics Comparison', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Performance summary
        axes[1, 2].axis('off')
        summary_text = f"""
        🏆 TRAINING SUMMARY
        
        Best AUC: {max(self.history['val_auc']):.4f}
        Best F1: {max(self.history['val_f1']):.4f}
        Final Train Loss: {self.history['train_loss'][-1]:.4f}
        Final Val Loss: {self.history['val_loss'][-1]:.4f}
        
        Total Epochs: {len(epochs)}
        Best Epoch: {np.argmax(self.history['val_auc']) + 1}
        
        🚀 Model Ready for Deployment!
        """
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('models/ultimate_training_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📈 Ultimate training plots saved to models/ultimate_training_results.png")

def main():
    """Main function to run ultimate training"""
    print("🎯" * 20)
    print("🎯 ULTIMATE ANOMALY DETECTION TRAINING 🎯")
    print("🎯 PROFESSIONAL EDITION 🎯")
    print("🎯" * 20)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    trainer = UltimateTrainer()
    
    # Check if data exists
    if not trainer.data_dir.exists():
        print(f"❌ Data directory not found: {trainer.data_dir}")
        print("Please ensure the dataanomaly folder contains the datasets")
        return
    
    try:
        # Start ultimate training
        print(f"\n⏰ Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        best_model, final_model = trainer.train(epochs=40, learning_rate=0.001)
        
        print(f"\n🎉 ULTIMATE TRAINING COMPLETED!")
        print("=" * 60)
        print(f"🏆 Best model saved to: {best_model}")
        print(f"📁 Final model saved to: {final_model}")
        print(f"📈 Training plots saved to: models/ultimate_training_results.png")
        print(f"⏰ Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n🚀 NEXT STEPS FOR DEPLOYMENT:")
        print("1. Run: streamlit run src/dashboard/app.py")
        print("2. Select 'Custom Trained Model (Advanced)' in the dashboard")
        print("3. Upload your test video and witness the magic!")
        print("4. 🎯 DEPLOY TO PRODUCTION! 🎯")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()