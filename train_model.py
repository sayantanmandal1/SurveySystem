#!/usr/bin/env python3
"""
Custom Model Training Launcher
Train a custom anomaly detection model on your dataset
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

def main():
    """Main training launcher"""
    print("🎯 Custom Anomaly Detection Model Training")
    print("=" * 60)
    
    # Check if dataset exists
    data_dir = Path("dataanomaly")
    if not data_dir.exists():
        print("❌ Dataset directory 'dataanomaly' not found!")
        print("Please ensure you have the Avenue Dataset in the dataanomaly folder")
        return
    
    avenue_dir = data_dir / "Avenue Dataset"
    if not avenue_dir.exists():
        print("❌ Avenue Dataset not found in dataanomaly folder!")
        return
    
    print("✅ Dataset found!")
    print(f"📁 Training videos: {len(list((avenue_dir / 'training_videos').glob('*.avi')))}")
    print(f"📁 Testing videos: {len(list((avenue_dir / 'testing_videos').glob('*.avi')))}")
    
    # Import and run training
    try:
        from training.custom_trainer import CustomAnomalyTrainer
        
        print("\n🚀 Starting training...")
        trainer = CustomAnomalyTrainer(data_dir=str(data_dir))
        
        # Train with optimal settings for production
        best_model, final_model = trainer.train(epochs=30, learning_rate=0.001)
        
        print("\n🎉 Training Complete!")
        print("=" * 40)
        print(f"✅ Best model: {best_model}")
        print(f"✅ Final model: {final_model}")
        print(f"📈 Training plots: models/training_history.png")
        
        print("\n🏆 Next Steps:")
        print("1. Run the ultimate dashboard: streamlit run src/dashboard/app.py")
        print("2. Select 'Custom Trained Model' in the dashboard")
        print("3. Upload your video and see the enhanced detection!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have PyTorch installed: pip install torch torchvision")
        print("2. Check that the dataset is properly structured")
        print("3. Make sure you have enough disk space and memory")

if __name__ == "__main__":
    main()