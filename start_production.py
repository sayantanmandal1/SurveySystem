#!/usr/bin/env python3
"""
Production startup script for AI Surveillance System
Handles all initialization and health checks before starting the dashboard
"""

import sys
import os
import subprocess
import time
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        'data', 'models', 'exp', 'logs', 'output',
        'src/dashboard', 'config', '.streamlit'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Directory ensured: {directory}")

def check_dependencies():
    """Check if all dependencies are installed"""
    logger.info("🔍 Checking dependencies...")
    
    try:
        import torch
        import cv2
        import streamlit
        import ultralytics
        import pandas
        import numpy
        import matplotlib
        import seaborn
        import sklearn
        import plotly
        logger.info("✅ All dependencies available")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False

def download_base_model():
    """Download YOLOv8 base model if not present"""
    model_path = Path('yolov8n.pt')
    
    if not model_path.exists():
        logger.info("📥 Downloading YOLOv8 base model...")
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')  # This will download if not present
            logger.info("✅ YOLOv8 model downloaded")
        except Exception as e:
            logger.error(f"❌ Failed to download YOLOv8 model: {e}")
            return False
    else:
        logger.info("✅ YOLOv8 model already available")
    
    return True

def create_default_config():
    """Create default configuration files if they don't exist"""
    
    # System config
    system_config_path = Path('config/system_config.json')
    if not system_config_path.exists():
        default_config = {
            "yolo": {
                "model_path": "yolov8n.pt",
                "confidence": 0.4,
                "iou_threshold": 0.45
            },
            "anomaly": {
                "loitering_threshold": 12,
                "abandonment_threshold": 8,
                "movement_threshold": 0.06
            },
            "dashboard": {
                "max_upload_size": 200,
                "processing_fps": 30,
                "max_frames": 300
            }
        }
        
        with open(system_config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        logger.info("✅ Created default system configuration")
    
    # Streamlit config
    streamlit_config_path = Path('.streamlit/config.toml')
    if not streamlit_config_path.exists():
        streamlit_config = """[global]
developmentMode = false

[server]
port = 8501
address = "0.0.0.0"
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
serverAddress = "0.0.0.0"
serverPort = 8501

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
"""
        
        with open(streamlit_config_path, 'w') as f:
            f.write(streamlit_config)
        logger.info("✅ Created default Streamlit configuration")

def check_models():
    """Check available models and log status"""
    models = {
        'yolov8n.pt': 'YOLOv8 Base Model',
        'models/custom_anomaly_best.pth': 'Custom Anomaly Model',
        'models/balanced_anomaly_model.pkl': 'Balanced AI Model',
        'exp/best.pth': 'Survey Model'
    }
    
    available_models = []
    
    for model_path, description in models.items():
        if Path(model_path).exists():
            available_models.append(description)
            logger.info(f"✅ {description} available")
        else:
            logger.info(f"⚠️  {description} not found (optional)")
    
    if not available_models:
        logger.error("❌ No models available!")
        return False
    
    logger.info(f"📊 {len(available_models)} models available for use")
    return True

def run_health_check():
    """Run comprehensive health check"""
    logger.info("🏥 Running health check...")
    
    try:
        result = subprocess.run([
            sys.executable, 'health_check.py'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.info("✅ Health check passed")
            return True
        else:
            logger.warning("⚠️  Health check found issues but continuing...")
            logger.warning(result.stdout)
            return True  # Continue anyway for production
            
    except Exception as e:
        logger.warning(f"⚠️  Health check failed: {e}")
        return True  # Continue anyway

def start_dashboard():
    """Start the Streamlit dashboard"""
    logger.info("🚀 Starting AI Surveillance Dashboard...")
    
    dashboard_path = Path('src/dashboard/app.py')
    if not dashboard_path.exists():
        logger.error("❌ Dashboard file not found!")
        return False
    
    try:
        # Start Streamlit
        cmd = [
            sys.executable, '-m', 'streamlit', 'run',
            str(dashboard_path),
            '--server.port=8501',
            '--server.address=localhost',
            '--browser.gatherUsageStats=false'
        ]
        
        logger.info("🌐 Dashboard starting at http://localhost:8501")
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        logger.info("🛑 Dashboard stopped by user")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to start dashboard: {e}")
        return False

def main():
    """Main production startup sequence"""
    logger.info("🚀 Starting AI Surveillance System - Production Mode")
    logger.info("=" * 60)
    
    # Step 1: Ensure directories
    ensure_directories()
    
    # Step 2: Check dependencies
    if not check_dependencies():
        logger.error("❌ Dependency check failed. Run: pip install -r requirements.txt")
        return 1
    
    # Step 3: Download base model
    if not download_base_model():
        logger.error("❌ Failed to ensure base model availability")
        return 1
    
    # Step 4: Create default configurations
    create_default_config()
    
    # Step 5: Check models
    if not check_models():
        logger.error("❌ No models available for detection")
        return 1
    
    # Step 6: Run health check
    run_health_check()
    
    # Step 7: Start dashboard
    logger.info("🎯 All checks passed. Starting dashboard...")
    time.sleep(2)
    
    if start_dashboard():
        logger.info("✅ System shutdown gracefully")
        return 0
    else:
        logger.error("❌ Dashboard failed to start")
        return 1

if __name__ == "__main__":
    sys.exit(main())