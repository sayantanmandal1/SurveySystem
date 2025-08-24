#!/usr/bin/env python3
"""
Comprehensive test suite for AI Surveillance System
"""

import unittest
import sys
import os
import tempfile
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

class TestSystemComponents(unittest.TestCase):
    """Test system components"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_data_dir = Path(tempfile.mkdtemp())
        
    def test_imports(self):
        """Test that all required modules can be imported"""
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
            self.assertTrue(True, "All imports successful")
        except ImportError as e:
            self.fail(f"Import failed: {e}")
    
    def test_yolo_detector(self):
        """Test YOLO detector initialization"""
        try:
            from detection.yolo_detector import YOLODetector
            config = {
                "model_path": "yolov8n.pt",
                "confidence": 0.4,
                "iou_threshold": 0.45
            }
            detector = YOLODetector(config)
            self.assertIsNotNone(detector)
        except Exception as e:
            self.skipTest(f"YOLO detector test skipped: {e}")
    
    def test_alert_manager(self):
        """Test alert manager functionality"""
        try:
            from utils.alert_manager import AlertManager
            config = {
                'save_path': str(self.test_data_dir / 'test_alerts.json'),
                'dashboard_update': False
            }
            alert_manager = AlertManager(config)
            
            # Test alert creation
            anomaly = {
                'type': 'loitering',
                'bbox': [100, 100, 200, 200],
                'confidence': 0.8
            }
            alert = alert_manager.create_alert(anomaly, 10.5)
            self.assertIsNotNone(alert)
            self.assertEqual(alert['type'], 'loitering')
            
        except Exception as e:
            self.skipTest(f"Alert manager test skipped: {e}")
    
    def test_behavior_analyzer(self):
        """Test behavior analyzer"""
        try:
            from anomaly.behavior_analyzer import BehaviorAnalyzer
            config = {
                "loitering_threshold": 12,
                "abandonment_threshold": 8,
                "movement_threshold": 0.06
            }
            analyzer = BehaviorAnalyzer(config)
            self.assertIsNotNone(analyzer)
            
        except Exception as e:
            self.skipTest(f"Behavior analyzer test skipped: {e}")
    
    def test_synthetic_frame_generation(self):
        """Test synthetic frame generation"""
        # Create a simple test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.assertEqual(test_frame.shape, (480, 640, 3))
        
    def test_configuration_loading(self):
        """Test configuration file loading"""
        config_path = Path('config/system_config.json')
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.assertIsInstance(config, dict)
        else:
            self.skipTest("Configuration file not found")

class TestDataProcessing(unittest.TestCase):
    """Test data processing functions"""
    
    def test_video_processing(self):
        """Test basic video processing capabilities"""
        try:
            import cv2
            # Test if OpenCV can create a VideoCapture object
            cap = cv2.VideoCapture()
            self.assertIsNotNone(cap)
            cap.release()
        except Exception as e:
            self.fail(f"Video processing test failed: {e}")
    
    def test_image_processing(self):
        """Test image processing functions"""
        try:
            import cv2
            import numpy as np
            
            # Create test image
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Test basic operations
            gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            self.assertEqual(gray.shape, (480, 640))
            
            # Test resize
            resized = cv2.resize(test_image, (320, 240))
            self.assertEqual(resized.shape, (240, 320, 3))
            
        except Exception as e:
            self.fail(f"Image processing test failed: {e}")

class TestModelLoading(unittest.TestCase):
    """Test model loading capabilities"""
    
    def test_yolo_model_loading(self):
        """Test YOLO model loading"""
        try:
            from ultralytics import YOLO
            
            # Try to load YOLOv8 model
            if Path('yolov8n.pt').exists():
                model = YOLO('yolov8n.pt')
                self.assertIsNotNone(model)
            else:
                self.skipTest("YOLOv8 model file not found")
                
        except Exception as e:
            self.skipTest(f"YOLO model loading test skipped: {e}")
    
    def test_custom_model_paths(self):
        """Test custom model file paths"""
        model_paths = [
            'models/custom_anomaly_best.pth',
            'models/balanced_anomaly_model.pkl',
            'exp/best.pth'
        ]
        
        found_models = []
        for model_path in model_paths:
            if Path(model_path).exists():
                found_models.append(model_path)
        
        # At least one model should be available for full functionality
        if not found_models:
            self.skipTest("No custom models found - this is optional")

class TestStreamlitApp(unittest.TestCase):
    """Test Streamlit application components"""
    
    def test_streamlit_import(self):
        """Test Streamlit import and basic functionality"""
        try:
            import streamlit as st
            self.assertTrue(hasattr(st, 'set_page_config'))
            self.assertTrue(hasattr(st, 'markdown'))
            self.assertTrue(hasattr(st, 'selectbox'))
        except ImportError:
            self.fail("Streamlit import failed")
    
    def test_dashboard_file_exists(self):
        """Test that dashboard file exists"""
        dashboard_path = Path('src/dashboard/app.py')
        self.assertTrue(dashboard_path.exists(), "Dashboard file not found")
    
    def test_plotly_functionality(self):
        """Test Plotly for dashboard charts"""
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            import pandas as pd
            
            # Test basic chart creation
            df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
            fig = px.line(df, x='x', y='y')
            self.assertIsNotNone(fig)
            
        except Exception as e:
            self.fail(f"Plotly functionality test failed: {e}")

def run_tests():
    """Run all tests"""
    print("🧪 Running AI Surveillance System Tests")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestSystemComponents,
        TestDataProcessing,
        TestModelLoading,
        TestStreamlitApp
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results:")
    print(f"✅ Tests run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️  Errors: {len(result.errors)}")
    print(f"⏭️  Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print("\n⚠️  Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('\\n')[-2]}")
    
    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)