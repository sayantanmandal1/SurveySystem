#!/usr/bin/env python3
"""
Demo Launcher - Easy way to run different demo modes
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'opencv-python', 'ultralytics', 
        'numpy', 'requests', 'scipy'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\n📦 Install with: pip install -r requirements.txt")
        return False
    
    return True

def setup_directories():
    """Create necessary directories"""
    dirs = ['data', 'output', 'models']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)

def main():
    """Main launcher"""
    print("🚀 AI Surveillance System - Demo Launcher")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Setup directories
    setup_directories()
    
    print("\n🎯 Available Options:")
    print("1. 🎬 Quick Demo (Create and process demo video)")
    print("2. 🌐 Web Demo (Streamlit app with video URL support)")
    print("3. 🎯 Ultimate Dashboard (Professional edition)")
    print("4. 🧠 Train Custom Model (Train on your dataset)")
    print("5. 🔧 Setup System (Download models and prepare)")
    print("6. 📈 Benchmark Evaluation (Test on datasets)")
    
    choice = input("\nSelect option (1-6): ").strip()
    
    if choice == "1":
        print("\n🎬 Starting Quick Demo...")
        subprocess.run([sys.executable, "quick_demo.py"])
    
    elif choice == "2":
        print("\n🌐 Starting Web Demo...")
        print("Opening browser at: http://localhost:8501")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/web_demo.py"])
    
    elif choice == "3":
        print("\n🎯 Starting Ultimate Dashboard...")
        print("🚀 Professional Edition with Custom Model Support!")
        print("Opening browser at: http://localhost:8501")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py"])
    
    elif choice == "4":
        print("\n🧠 Starting Custom Model Training...")
        subprocess.run([sys.executable, "train_model.py"])
    
    elif choice == "5":
        print("\n🔧 Setting up system...")
        subprocess.run([sys.executable, "setup.py"])
        print("\n✅ Setup complete! You can now run other options.")
    
    elif choice == "6":
        print("\n📈 Starting Benchmark Evaluation...")
        subprocess.run([sys.executable, "benchmark_demo.py"])
    
    else:
        print("❌ Invalid choice. Please select 1-6.")

if __name__ == "__main__":
    main()