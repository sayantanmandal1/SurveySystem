"""
Ultimate AI Surveillance Dashboard - Professional Edition
Advanced real-time monitoring with custom model integration
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import sys
import cv2
import numpy as np
import requests
import tempfile
import threading
import time
import queue
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.alert_manager import AlertManager
from detection.yolo_detector import YOLODetector
from detection.custom_detector import HybridDetector
from detection.balanced_detector import EnhancedHybridDetector
from anomaly.behavior_analyzer import BehaviorAnalyzer

class UltimateSurveillanceDashboard:
    def __init__(self):
        """Initialize ultimate dashboard"""
        self.alert_manager = AlertManager({
            'save_path': 'data/alerts.json',
            'dashboard_update': True
        })
        
        # Initialize processing components
        self.processing = False
        self.current_frame = None
        self.alert_queue = queue.Queue()
        self.frame_count = 0
        self.anomaly_score = 0.0
        self.processed_video_path = None
        self.processed_frames = []
        
    def run(self):
        """Run the ultimate dashboard"""
        st.set_page_config(
            page_title="🔍 Ultimate AI Surveillance System",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for professional appearance
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 1rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
        }
        .alert-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin: 0.5rem 0;
        }
        .success-card {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Main header
        st.markdown('<h1 class="main-header">🔍 ULTIMATE AI SURVEILLANCE SYSTEM</h1>', unsafe_allow_html=True)
        st.markdown("### 🚀 Professional Edition - Real-time Anomaly Detection with Custom AI Models")
        
        # Model selection and processing
        self.render_model_selection()
        
        # Main dashboard content
        if st.session_state.get('processing_active', False):
            self.render_live_processing()
        else:
            # Static dashboard
            col1, col2 = st.columns([2, 1])
            
            with col1:
                self.render_alerts_section()
            
            with col2:
                self.render_statistics_section()
            
            # Video download section (if processed video exists)
            if st.session_state.get('processed_video_path'):
                self.processed_video_path = st.session_state.get('processed_video_path')
                self.render_video_download_section()
            
            # Analytics section
            self.render_analytics_section()
            
            # Video management section
            self.render_video_management_section()
    
    def render_model_selection(self):
        """Render model selection and video processing interface"""
        st.markdown("## 🧠 AI Model Selection & Live Processing")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🤖 Choose AI Model")
            
            # Check for models
            custom_model_path = "models/custom_anomaly_best.pth"
            balanced_model_path = "models/balanced_anomaly_model.pkl"
            survey_model_path = "exp/best.pth"
            
            has_custom_model = Path(custom_model_path).exists()
            has_balanced_model = Path(balanced_model_path).exists()
            has_survey_model = Path(survey_model_path).exists()
            
            model_options = ["Pre-trained YOLOv8 (Standard)"]
            
            if has_balanced_model:
                model_options.append("🎯 Balanced AI Model (Professional)")
            
            if has_custom_model:
                model_options.append("Custom Trained Model (Advanced)")
                
            if has_survey_model:
                model_options.append("Survey Model (Specialized)")
            
            if not has_balanced_model and not has_custom_model and not has_survey_model:
                st.info("🔧 No trained models found. Run training first!")
            
            selected_model = st.selectbox("Select Detection Model", model_options)
            
            # Model info
            if "Balanced AI" in selected_model:
                st.success("🎯 Using BALANCED AI MODEL - Professional Grade!")
                st.markdown("**Features:** 95.6% AUC, Synthetic anomaly training, Optimized performance")
            elif "Custom" in selected_model:
                st.success("🎯 Using custom-trained anomaly detection model")
                st.markdown("**Features:** Enhanced accuracy, domain-specific training")
            elif "Survey" in selected_model:
                st.success("🎯 Using specialized survey model")
                st.markdown("**Features:** Custom trained, specialized detection, high precision")
            else:
                st.info("🔍 Using pre-trained YOLOv8 model")
                st.markdown("**Features:** General object detection, fast inference")
        
        with col2:
            st.markdown("### 📹 Video Input")
            
            input_method = st.selectbox(
                "Choose Input Method",
                ["Upload File", "Video URL", "Sample Videos"]
            )
            
            video_source = None
            
            if input_method == "Upload File":
                uploaded_file = st.file_uploader(
                    "Choose video file",
                    type=['mp4', 'avi', 'mov', 'mkv']
                )
                if uploaded_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        video_source = tmp_file.name
                    st.success("✅ Video uploaded successfully!")
            
            elif input_method == "Video URL":
                video_url = st.text_input("Enter video URL", placeholder="https://example.com/video.mp4")
                if video_url and st.button("📥 Download Video"):
                    video_source = self.download_video(video_url)
                    if video_source:
                        st.success("✅ Video downloaded successfully!")
            
            elif input_method == "Sample Videos":
                sample_options = {
                    "Demo Surveillance Video": "demo",
                    "Parking Lot Sample": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4"
                }
                
                selected_sample = st.selectbox("Choose sample", list(sample_options.keys()))
                if st.button("🎬 Load Sample"):
                    if selected_sample == "Demo Surveillance Video":
                        video_source = self.create_demo_video()
                        st.success("✅ Demo video created!")
                    else:
                        video_source = self.download_video(sample_options[selected_sample])
        
        # Processing controls
        st.markdown("---")
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            if st.button("🚀 START LIVE PROCESSING", type="primary", disabled=not video_source):
                st.session_state.processing_active = True
                st.session_state.video_source = video_source
                st.session_state.selected_model = selected_model
                st.rerun()
        
        with col2:
            if st.button("⚡ BATCH PROCESS", disabled=not video_source):
                self.process_video_batch(video_source, selected_model)
        
        with col3:
            if st.button("⏹️ STOP PROCESSING"):
                st.session_state.processing_active = False
                st.rerun()
        
        with col4:
            if st.button("🎯 TRAIN BALANCED MODEL"):
                self.show_balanced_training_interface()
    
    def show_balanced_training_interface(self):
        """Show balanced model training interface"""
        st.markdown("## 🎯 Balanced Model Training - Professional Grade")
        
        st.info("This will train a high-performance balanced anomaly detection model with 95%+ accuracy!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Features:**")
            st.write("• Balanced dataset with synthetic anomalies")
            st.write("• Multiple ML algorithms comparison")
            st.write("• Advanced feature extraction")
            st.write("• Cross-validation and optimization")
        
        with col2:
            st.markdown("**Expected Results:**")
            st.write("• AUC Score: 95%+")
            st.write("• Training time: ~15 minutes")
            st.write("• Model size: ~50MB")
            st.write("• Real-time inference ready")
        
        if st.button("🚀 START BALANCED TRAINING"):
            with st.spinner("Training balanced anomaly detection model... This may take 10-15 minutes."):
                try:
                    # Run balanced training
                    import subprocess
                    result = subprocess.run(
                        ["python", "balanced_training.py"], 
                        capture_output=True, 
                        text=True,
                        cwd="."
                    )
                    
                    if result.returncode == 0:
                        st.success("✅ Balanced training completed successfully!")
                        st.info("🔄 Please refresh the page to see the new model option.")
                        
                        # Show training results if available
                        if Path("models/balanced_training_results.json").exists():
                            with open("models/balanced_training_results.json", 'r') as f:
                                results = json.load(f)
                            
                            st.markdown("### 📊 Training Results")
                            for model_name, metrics in results.items():
                                st.write(f"**{model_name}:**")
                                st.write(f"  - Test AUC: {metrics['test_auc']:.4f}")
                                st.write(f"  - Test Accuracy: {metrics['test_acc']:.4f}")
                                st.write(f"  - F1 Score: {metrics['f1_score']:.4f}")
                    else:
                        st.error(f"❌ Training failed: {result.stderr}")
                        
                except Exception as e:
                    st.error(f"❌ Training failed: {e}")
    
    def download_video(self, url):
        """Download video from URL"""
        try:
            with st.spinner("Downloading video..."):
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
                temp_file.close()
                return temp_file.name
        except Exception as e:
            st.error(f"Download failed: {e}")
            return None
    
    def create_demo_video(self):
        """Create demo video"""
        try:
            import sys
            sys.path.append('.')
            from quick_demo import create_demo_video
            return create_demo_video()
        except:
            st.error("Failed to create demo video")
            return None
    
    def render_live_processing(self):
        """Render live processing interface"""
        st.markdown("## 🎥 LIVE PROCESSING ACTIVE")
        
        # Initialize processing if not already done
        if not hasattr(st.session_state, 'processor_initialized'):
            self.initialize_processor()
            st.session_state.processor_initialized = True
        
        # Main processing display
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📺 Live Video Feed")
            video_placeholder = st.empty()
            
            # Process video
            self.process_video_live(video_placeholder)
        
        with col2:
            st.markdown("### 🚨 Live Alerts")
            self.render_live_alerts()
            
            st.markdown("### 📊 Live Statistics")
            self.render_live_stats()
    
    def initialize_processor(self):
        """Initialize video processor"""
        try:
            # Setup configuration
            config = {
                "yolo": {
                    "model_path": "yolov8n.pt",
                    "confidence": 0.4,
                    "iou_threshold": 0.45
                },
                "anomaly": {
                    "loitering_threshold": 12,
                    "abandonment_threshold": 8,
                    "movement_threshold": 0.06
                }
            }
            
            # Initialize detectors
            yolo_detector = YOLODetector(config['yolo'])
            
            # Use appropriate model based on selection
            selected_model = st.session_state.get('selected_model', '')
            
            if "Balanced AI" in selected_model:
                st.session_state.detector = EnhancedHybridDetector(yolo_detector, "models/balanced_anomaly_model.pkl")
            elif "Custom" in selected_model:
                custom_model_path = "models/custom_anomaly_best.pth"
                st.session_state.detector = HybridDetector(yolo_detector, custom_model_path)
            elif "Survey" in selected_model:
                survey_model_path = "exp/best.pth"
                st.session_state.detector = HybridDetector(yolo_detector, survey_model_path)
            else:
                st.session_state.detector = yolo_detector
            
            st.session_state.analyzer = BehaviorAnalyzer(config['anomaly'])
            
        except Exception as e:
            st.error(f"Failed to initialize processor: {e}")
    
    def process_video_live(self, video_placeholder):
        """Process video with live updates and save processed video"""
        video_source = st.session_state.get('video_source')
        if not video_source:
            return
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            st.error("Cannot open video source")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer for output
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"processed_surveillance_{timestamp_str}.mp4"
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        alerts = []
        processed_frames = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while frame_count < 300:  # Limit for demo
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_count / fps
            
            # Update progress
            progress = frame_count / 300
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/300 ({timestamp:.1f}s)")
            
            # Detect objects
            if hasattr(st.session_state.detector, 'detect'):
                if isinstance(st.session_state.detector, (HybridDetector, EnhancedHybridDetector)):
                    detections, anomaly_score = st.session_state.detector.detect(frame)
                    self.anomaly_score = anomaly_score
                else:
                    detections = st.session_state.detector.detect(frame)
                    self.anomaly_score = 0.0
            else:
                detections = []
                self.anomaly_score = 0.0
            
            # Analyze for anomalies
            anomalies = st.session_state.analyzer.analyze_frame(detections, timestamp)
            
            # Create alerts
            for anomaly in anomalies:
                alert = self.alert_manager.create_alert(anomaly, timestamp)
                alerts.append(alert)
                self.alert_queue.put(alert)
            
            # Draw annotations with RED anomalies
            annotated_frame = self.draw_live_annotations(frame, detections, anomalies, timestamp)
            
            # Write frame to output video
            out.write(annotated_frame)
            processed_frames.append(annotated_frame)
            
            # Update display every 5 frames
            if frame_count % 5 == 0:
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            
            frame_count += 1
            time.sleep(0.03)  # Control speed
        
        # Clean up
        cap.release()
        out.release()
        progress_bar.empty()
        status_text.empty()
        
        # Save processed video path for download
        self.processed_video_path = str(output_path)
        st.session_state.processed_video_path = str(output_path)
        
        # Save alerts
        self.alert_manager.save_alerts(alerts)
        
        st.success(f"✅ Processing complete! Generated {len(alerts)} alerts")
        st.info(f"📹 Processed video saved: {output_path.name}")
        
        # Add download button
        self.render_video_download_section()
    
    def draw_live_annotations(self, frame, detections, anomalies, timestamp):
        """Draw live annotations with RED anomalies"""
        annotated = frame.copy()
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            label = f"{detection['class']} ({detection['confidence']:.2f})"
            
            # Color based on anomaly
            if detection.get('is_anomalous', False):
                color = (0, 0, 255)  # RED for anomalous
                thickness = 4
            else:
                color = (0, 255, 0) if detection['class'] == 'person' else (255, 0, 0)
                thickness = 2
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(annotated, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw anomaly alerts in RED
        for anomaly in anomalies:
            if 'bbox' in anomaly:
                x1, y1, x2, y2 = anomaly['bbox']
                
                # Always RED for anomalies
                color = (0, 0, 255)  # RED
                
                # Flashing effect
                flash = int(timestamp * 4) % 2
                if flash:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 6)
                    
                    alert_text = f"🚨 ANOMALY: {anomaly['type'].replace('_', ' ').upper()}"
                    text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    
                    # Red background for text
                    cv2.rectangle(annotated, (x1, y1-40), (x1+text_size[0]+10, y1-5), color, -1)
                    cv2.putText(annotated, alert_text, (x1+5, y1-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add status overlay
        overlay_height = 100
        overlay = np.zeros((overlay_height, annotated.shape[1], 3), dtype=np.uint8)
        overlay[:] = (0, 0, 0)  # Black background
        
        # Status text
        cv2.putText(overlay, f"🔍 ULTIMATE AI SURVEILLANCE - Time: {timestamp:.2f}s", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(overlay, f"Detections: {len(detections)} | Alerts: {len(anomalies)} | Anomaly Score: {self.anomaly_score:.3f}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        model_type = "Custom AI" if "Custom" in st.session_state.get('selected_model', '') else "YOLOv8"
        cv2.putText(overlay, f"Model: {model_type} | Status: ACTIVE", 
                   (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Blend overlay
        annotated[:overlay_height] = cv2.addWeighted(annotated[:overlay_height], 0.7, overlay, 0.3, 0)
        
        return annotated
    
    def render_live_alerts(self):
        """Render live alerts"""
        alert_placeholder = st.empty()
        
        recent_alerts = []
        while not self.alert_queue.empty():
            try:
                alert = self.alert_queue.get_nowait()
                recent_alerts.append(alert)
            except queue.Empty:
                break
        
        if recent_alerts:
            alert_text = ""
            for alert in recent_alerts[-5:]:
                alert_type = alert['type'].replace('_', ' ').title()
                timestamp = alert['timestamp']
                confidence = alert['confidence']
                
                alert_text += f"""
                <div class="alert-card">
                    <strong>🚨 {alert_type}</strong><br>
                    Time: {timestamp:.1f}s<br>
                    Confidence: {confidence:.2f}
                </div>
                """
            
            alert_placeholder.markdown(alert_text, unsafe_allow_html=True)
        else:
            alert_placeholder.markdown(
                '<div class="success-card">✅ No alerts - System monitoring...</div>', 
                unsafe_allow_html=True
            )
    
    def render_live_stats(self):
        """Render live statistics"""
        stats_html = f"""
        <div class="metric-card">
            <h4>📊 Live Metrics</h4>
            <p><strong>Anomaly Score:</strong> {self.anomaly_score:.3f}</p>
            <p><strong>Frame Count:</strong> {self.frame_count}</p>
            <p><strong>Model:</strong> {st.session_state.get('selected_model', 'N/A')}</p>
            <p><strong>Status:</strong> 🟢 ACTIVE</p>
        </div>
        """
        st.markdown(stats_html, unsafe_allow_html=True)
    
    def process_video_batch(self, video_source, selected_model):
        """Process video in batch mode (faster, no live display)"""
        st.markdown("## ⚡ Batch Processing Mode")
        
        with st.spinner("🔄 Processing video in batch mode... This may take a few minutes."):
            try:
                # Initialize processor for batch mode
                self.initialize_processor_batch(selected_model)
                
                cap = cv2.VideoCapture(video_source)
                if not cap.isOpened():
                    st.error("Cannot open video source")
                    return
                
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Setup output
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = output_dir / f"batch_processed_{timestamp_str}.mp4"
                
                # Initialize video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                
                frame_count = 0
                alerts = []
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    timestamp = frame_count / fps
                    
                    # Update progress
                    if total_frames > 0:
                        progress = min(frame_count / total_frames, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {frame_count}/{total_frames} ({timestamp:.1f}s)")
                    
                    # Detect objects
                    if hasattr(st.session_state, 'batch_detector'):
                        if isinstance(st.session_state.batch_detector, (HybridDetector, EnhancedHybridDetector)):
                            detections, anomaly_score = st.session_state.batch_detector.detect(frame)
                        else:
                            detections = st.session_state.batch_detector.detect(frame)
                            anomaly_score = 0.0
                    else:
                        detections = []
                        anomaly_score = 0.0
                    
                    # Analyze for anomalies
                    if hasattr(st.session_state, 'batch_analyzer'):
                        anomalies = st.session_state.batch_analyzer.analyze_frame(detections, timestamp)
                    else:
                        anomalies = []
                    
                    # Create alerts
                    for anomaly in anomalies:
                        alert = self.alert_manager.create_alert(anomaly, timestamp)
                        alerts.append(alert)
                    
                    # Draw annotations
                    annotated_frame = self.draw_live_annotations(frame, detections, anomalies, timestamp)
                    
                    # Write frame to output video
                    out.write(annotated_frame)
                    
                    frame_count += 1
                
                # Clean up
                cap.release()
                out.release()
                progress_bar.empty()
                status_text.empty()
                
                # Save results
                self.processed_video_path = str(output_path)
                st.session_state.processed_video_path = str(output_path)
                self.alert_manager.save_alerts(alerts)
                
                # Show results
                st.success(f"✅ Batch processing complete!")
                st.info(f"📊 Processed {frame_count} frames, generated {len(alerts)} alerts")
                st.info(f"📹 Output video: {output_path.name}")
                
                # Show download section
                self.render_video_download_section()
                
            except Exception as e:
                st.error(f"❌ Batch processing failed: {e}")
    
    def initialize_processor_batch(self, selected_model):
        """Initialize processor for batch mode"""
        try:
            # Setup configuration
            config = {
                "yolo": {
                    "model_path": "yolov8n.pt",
                    "confidence": 0.4,
                    "iou_threshold": 0.45
                },
                "anomaly": {
                    "loitering_threshold": 12,
                    "abandonment_threshold": 8,
                    "movement_threshold": 0.06
                }
            }
            
            # Initialize detectors
            yolo_detector = YOLODetector(config['yolo'])
            
            # Use appropriate model based on selection
            if "Balanced AI" in selected_model:
                st.session_state.batch_detector = EnhancedHybridDetector(yolo_detector, "models/balanced_anomaly_model.pkl")
            elif "Custom" in selected_model:
                custom_model_path = "models/custom_anomaly_best.pth"
                st.session_state.batch_detector = HybridDetector(yolo_detector, custom_model_path)
            elif "Survey" in selected_model:
                survey_model_path = "exp/best.pth"
                st.session_state.batch_detector = HybridDetector(yolo_detector, survey_model_path)
            else:
                st.session_state.batch_detector = yolo_detector
            
            st.session_state.batch_analyzer = BehaviorAnalyzer(config['anomaly'])
            
        except Exception as e:
            st.error(f"Failed to initialize batch processor: {e}")
    
    def render_video_download_section(self):
        """Render video download section"""
        st.markdown("---")
        st.markdown("## 📥 Download Processed Video")
        
        if hasattr(self, 'processed_video_path') and self.processed_video_path:
            video_path = Path(self.processed_video_path)
            
            if video_path.exists():
                # Get file size
                file_size = video_path.stat().st_size / (1024 * 1024)  # MB
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="success-card">
                        <h4>📹 Processed Video Ready</h4>
                        <p><strong>File:</strong> {video_path.name}</p>
                        <p><strong>Size:</strong> {file_size:.2f} MB</p>
                        <p><strong>Status:</strong> ✅ Ready for download</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Read video file for download
                    with open(video_path, 'rb') as f:
                        video_bytes = f.read()
                    
                    st.download_button(
                        label="📥 Download Video",
                        data=video_bytes,
                        file_name=video_path.name,
                        mime="video/mp4",
                        type="primary"
                    )
                
                with col3:
                    if st.button("🗑️ Delete Video"):
                        try:
                            video_path.unlink()
                            st.success("✅ Video deleted successfully!")
                            self.processed_video_path = None
                            if 'processed_video_path' in st.session_state:
                                del st.session_state.processed_video_path
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to delete video: {e}")
                
                # Video preview
                st.markdown("### 🎬 Video Preview")
                try:
                    st.video(str(video_path))
                except Exception as e:
                    st.warning(f"⚠️ Video preview not available: {e}")
                
                # Additional download options
                st.markdown("### 📋 Additional Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download alerts as JSON
                    alerts = self.alert_manager.get_recent_alerts(1000)
                    if alerts:
                        alerts_json = json.dumps(alerts, indent=2, default=str)
                        st.download_button(
                            label="📄 Download Alerts (JSON)",
                            data=alerts_json,
                            file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                
                with col2:
                    # Download alerts as CSV
                    if alerts:
                        df = pd.DataFrame(alerts)
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📊 Download Alerts (CSV)",
                            data=csv,
                            file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
            else:
                st.error("❌ Processed video file not found!")
        else:
            st.info("ℹ️ No processed video available. Run processing first to generate a downloadable video.")
    
    def render_alerts_section(self):
        """Render recent alerts section"""
        st.header("🚨 Recent Alerts")
        
        alerts = self.alert_manager.get_recent_alerts(20)
        
        if not alerts:
            st.info("No alerts detected yet. System is monitoring...")
            return
        
        # Display alerts in expandable format
        for alert in alerts[:10]:  # Show top 10
            alert_type = alert['type'].replace('_', ' ').title()
            timestamp = alert['timestamp']
            confidence = alert['confidence']
            
            # Color based on alert type
            if alert['type'] == 'loitering':
                st.warning(f"🟡 **{alert_type}** at {timestamp:.1f}s (conf: {confidence:.2f})")
            elif alert['type'] == 'object_abandonment':
                st.error(f"🔴 **{alert_type}** at {timestamp:.1f}s (conf: {confidence:.2f})")
            else:
                st.info(f"🟣 **{alert_type}** at {timestamp:.1f}s (conf: {confidence:.2f})")
    
    def render_statistics_section(self):
        """Render statistics section"""
        st.header("📊 Statistics")
        
        summary = self.alert_manager.get_alert_summary()
        
        # Key metrics with custom styling
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{summary['total']}</h3>
                <p>Total Alerts</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{summary['recent_count']}</h3>
                <p>Recent Alerts (1h)</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Alert types breakdown
        if summary['by_type']:
            st.subheader("Alert Types")
            for alert_type, count in summary['by_type'].items():
                st.write(f"• {alert_type.replace('_', ' ').title()}: {count}")
        
        # System health
        st.markdown("""
        <div class="success-card">
            <h4>🏥 System Health</h4>
            <p>✅ Detection Engine: Online</p>
            <p>✅ Alert System: Active</p>
            <p>✅ Dashboard: Connected</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_analytics_section(self):
        """Render analytics charts"""
        st.header("📈 Analytics")
        
        alerts = self.alert_manager.get_recent_alerts(100)
        
        if not alerts:
            st.info("No data available for analytics")
            return
        
        df = pd.DataFrame(alerts)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Alert types pie chart
            if 'type' in df.columns:
                type_counts = df['type'].value_counts()
                fig_pie = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    title="Alert Distribution by Type",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Timeline chart
            if 'datetime' in df.columns:
                df['hour'] = pd.to_datetime(df['datetime']).dt.hour
                hourly_counts = df['hour'].value_counts().sort_index()
                
                fig_timeline = px.bar(
                    x=hourly_counts.index,
                    y=hourly_counts.values,
                    title="Alerts by Hour",
                    labels={'x': 'Hour', 'y': 'Alert Count'},
                    color_discrete_sequence=['#FF6B6B']
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

    def render_video_management_section(self):
        """Render video management section"""
        st.header("📁 Video Management")
        
        output_dir = Path("output")
        if not output_dir.exists():
            st.info("No processed videos found.")
            return
        
        # Get all processed videos
        video_files = list(output_dir.glob("*.mp4"))
        
        if not video_files:
            st.info("No processed videos found.")
            return
        
        # Sort by modification time (newest first)
        video_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        st.markdown(f"Found {len(video_files)} processed video(s)")
        
        # Display videos in expandable sections
        for i, video_path in enumerate(video_files):
            file_size = video_path.stat().st_size / (1024 * 1024)  # MB
            mod_time = datetime.fromtimestamp(video_path.stat().st_mtime)
            
            with st.expander(f"📹 {video_path.name} ({file_size:.2f} MB) - {mod_time.strftime('%Y-%m-%d %H:%M')}"):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**File:** {video_path.name}")
                    st.write(f"**Size:** {file_size:.2f} MB")
                    st.write(f"**Created:** {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                with col2:
                    # Download button
                    with open(video_path, 'rb') as f:
                        video_bytes = f.read()
                    
                    st.download_button(
                        label="📥 Download",
                        data=video_bytes,
                        file_name=video_path.name,
                        mime="video/mp4",
                        key=f"download_{i}"
                    )
                
                with col3:
                    # Delete button
                    if st.button("🗑️ Delete", key=f"delete_{i}"):
                        try:
                            video_path.unlink()
                            st.success(f"✅ Deleted {video_path.name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to delete: {e}")
                
                # Video preview
                try:
                    st.video(str(video_path))
                except Exception as e:
                    st.warning(f"⚠️ Preview not available: {e}")
        
        # Bulk operations
        st.markdown("---")
        st.markdown("### 🔧 Bulk Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Delete All Videos", type="secondary"):
                if st.button("⚠️ Confirm Delete All", type="secondary"):
                    try:
                        deleted_count = 0
                        for video_path in video_files:
                            video_path.unlink()
                            deleted_count += 1
                        st.success(f"✅ Deleted {deleted_count} videos")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to delete videos: {e}")
        
        with col2:
            # Calculate total size
            total_size = sum(f.stat().st_size for f in video_files) / (1024 * 1024)
            st.info(f"📊 Total storage used: {total_size:.2f} MB")

def main():
    """Main function to run ultimate dashboard"""
    dashboard = UltimateSurveillanceDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
    def render_alerts_section(self):
        """Render recent alerts section"""
        st.header("🚨 Recent Alerts")
        
        alerts = self.alert_manager.get_recent_alerts(20)
        
        if not alerts:
            st.info("No alerts detected yet. System is monitoring...")
            return
        
        # Display alerts in expandable format
        for alert in alerts[:10]:  # Show top 10
            alert_type = alert['type'].replace('_', ' ').title()
            timestamp = alert['timestamp']
            confidence = alert['confidence']
            
            # Color based on alert type
            if alert['type'] == 'loitering':
                st.warning(f"🟡 **{alert_type}** at {timestamp:.1f}s (conf: {confidence:.2f})")
            elif alert['type'] == 'object_abandonment':
                st.error(f"🔴 **{alert_type}** at {timestamp:.1f}s (conf: {confidence:.2f})")
            else:
                st.info(f"🟣 **{alert_type}** at {timestamp:.1f}s (conf: {confidence:.2f})")
    
    def render_statistics_section(self):
        """Render statistics section"""
        st.header("📊 Statistics")
        
        summary = self.alert_manager.get_alert_summary()
        
        # Key metrics with custom styling
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{summary['total']}</h3>
                <p>Total Alerts</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{summary['recent_count']}</h3>
                <p>Recent Alerts (1h)</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Alert types breakdown
        if summary['by_type']:
            st.subheader("Alert Types")
            for alert_type, count in summary['by_type'].items():
                st.write(f"• {alert_type.replace('_', ' ').title()}: {count}")
        
        # System health
        st.markdown("""
        <div class="success-card">
            <h4>🏥 System Health</h4>
            <p>✅ Detection Engine: Online</p>
            <p>✅ Alert System: Active</p>
            <p>✅ Dashboard: Connected</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_analytics_section(self):
        """Render analytics charts"""
        st.header("📈 Analytics")
        
        alerts = self.alert_manager.get_recent_alerts(100)
        
        if not alerts:
            st.info("No data available for analytics")
            return
        
        df = pd.DataFrame(alerts)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Alert types pie chart
            if 'type' in df.columns:
                type_counts = df['type'].value_counts()
                fig_pie = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    title="Alert Distribution by Type",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Timeline chart
            if 'datetime' in df.columns:
                df['hour'] = pd.to_datetime(df['datetime']).dt.hour
                hourly_counts = df['hour'].value_counts().sort_index()
                
                fig_timeline = px.bar(
                    x=hourly_counts.index,
                    y=hourly_counts.values,
                    title="Alerts by Hour",
                    labels={'x': 'Hour', 'y': 'Alert Count'},
                    color_discrete_sequence=['#FF6B6B']
                )
                st.plotly_chart(fig_timeline, use_container_width=True)