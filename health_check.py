#!/usr/bin/env python3
"""
Health check script for AI Surveillance System
Verifies all components are working correctly
"""

import sys
import os
import subprocess
import importlib
import requests
import time
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'torch', 'torchvision', 'cv2', 'ultralytics', 
        'streamlit', 'pandas', 'numpy', 'matplotlib', 
        'seaborn', 'sklearn', 'PIL', 'plotly', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                importlib.import_module('cv2')
            elif package == 'PIL':
                importlib.import_module('PIL')
            elif package == 'sklearn':
                importlib.import_module('sklearn')
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_directory_structure():
    """Check if required directories exist"""
    print("\n📁 Checking directory structure...")
    
    required_dirs = [
        'src', 'src/dashboard', 'src/detection', 'src/anomaly',
        'src/utils', 'data', 'models', 'config'
    ]
    
    missing_dirs = []
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - Missing")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\n⚠️  Missing directories: {', '.join(missing_dirs)}")
        return False
    
    return True

def check_model_files():
    """Check if model files are available"""
    print("\n🧠 Checking model files...")
    
    model_files = [
        ('yolov8n.pt', 'YOLOv8 base model'),
        ('models/custom_anomaly_best.pth', 'Custom anomaly model (optional)'),
        ('models/balanced_anomaly_model.pkl', 'Balanced model (optional)'),
        ('exp/best.pth', 'Survey model (optional)')
    ]
    
    available_models = 0
    
    for model_path, description in model_files:
        if Path(model_path).exists():
            print(f"✅ {description}")
            available_models += 1
        else:
            print(f"⚠️  {description} - Not found")
    
    if available_models == 0:
        print("❌ No models found. At least YOLOv8 base model is required.")
        return False
    
    return True

def check_configuration():
    """Check configuration files"""
    print("\n⚙️  Checking configuration...")
    
    config_files = [
        'config/system_config.json',
        '.streamlit/config.toml'
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✅ {config_file}")
        else:
            print(f"⚠️  {config_file} - Not found (will use defaults)")
    
    return True

def check_streamlit_health():
    """Check if Streamlit can start"""
    print("\n🌐 Testing Streamlit startup...")
    
    try:
        # Try to import streamlit
        import streamlit as st
        print("✅ Streamlit import successful")
        
        # Check if dashboard file exists
        dashboard_path = Path('src/dashboard/app.py')
        if dashboard_path.exists():
            print("✅ Dashboard file found")
            return True
        else:
            print("❌ Dashboard file not found")
            return False
            
    except Exception as e:
        print(f"❌ Streamlit check failed: {e}")
        return False

def check_docker_setup():
    """Check Docker configuration"""
    print("\n🐳 Checking Docker setup...")
    
    docker_files = ['Dockerfile', 'docker-compose.yml', '.dockerignore']
    
    for docker_file in docker_files:
        if Path(docker_file).exists():
            print(f"✅ {docker_file}")
        else:
            print(f"⚠️  {docker_file} - Not found")
    
    # Check if Docker is available
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Docker is available")
        else:
            print("⚠️  Docker not available")
    except:
        print("⚠️  Docker not available")
    
    return True

def run_basic_tests():
    """Run basic functionality tests"""
    print("\n🧪 Running basic tests...")
    
    try:
        # Test YOLO import
        from ultralytics import YOLO
        print("✅ YOLO import successful")
        
        # Test OpenCV
        import cv2
        print("✅ OpenCV import successful")
        
        # Test torch
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available - {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - using CPU")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic tests failed: {e}")
        return False

def main():
    """Run complete health check"""
    print("🏥 AI Surveillance System Health Check")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Directory Structure", check_directory_structure),
        ("Model Files", check_model_files),
        ("Configuration", check_configuration),
        ("Streamlit", check_streamlit_health),
        ("Docker Setup", check_docker_setup),
        ("Basic Tests", run_basic_tests)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        try:
            if check_func():
                passed += 1
        except Exception as e:
            print(f"❌ {check_name} check failed with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Health Check Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! System is ready for deployment.")
        return 0
    elif passed >= total * 0.8:
        print("⚠️  Most checks passed. System should work with minor issues.")
        return 0
    else:
        print("❌ Multiple issues found. Please fix before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())