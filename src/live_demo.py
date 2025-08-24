#!/usr/bin/env python3
"""
Live Demo System for AI Surveillance
Real-time video processing with live dashboard updates
"""

import cv2
import numpy as np
import streamlit as st
import threading
import time
import queue
import requests
import tempfile
from pathlib import Path
import sys
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from detection.yolo_detector import YOLODetector
from anomaly.behavior_analyzer import BehaviorAnalyzer
from utils.alert_manager import AlertManager

class LiveSurveillanceDemo:
    def __init__(self):
        """Initialize live demo system"""
        self.config = {
            "yolo": {
                "model_path": "yolov8n.pt",
                "confidence": 0.4,
                "iou_threshold": 0.45
            },
            "anomaly": {
                "loitering_threshold": 15,  # Reduced for demo
                "abandonment_threshold": 8,
                "movement_threshold": 0.08
            },
            "alerts": {
                "save_path": "data/live_alerts.json",
                "dashboard_update": True
            }
        }
        
        self.detector = YOLODetector(self.config['yolo'])
        self.analyzer = BehaviorAnalyzer(self.config['anomaly'])
        self.alert_manager = AlertManager(self.config['alerts'])
        
        self.processing = False
        self.current_frame = None
        self.alert_queue = queue.Queue()
        self.frame_queue = queue.Queue(maxsize=10)
        
    def download_video(self, url, output_path):
        """Download video from URL"""
        try:
            print(f"📥 Downloading video from: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Video downloaded to: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def process_video_stream(self, video_source):
        """Process video stream in real-time"""
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"❌ Cannot open video source: {video_source}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = 0
        
        print(f"🎥 Starting video processing (FPS: {fps})")
        
        while self.processing and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_count / fps
            
            # Detect objects
            detections = self.detector.detect(frame)
            
            # Analyze for anomalies
            anomalies = self.analyzer.analyze_frame(detections, timestamp)
            
            # Create annotated frame
            annotated_frame = self.draw_annotations(frame, detections, anomalies)
            
            # Store current frame
            self.current_frame = annotated_frame.copy()
            
            # Add to frame queue for display
            if not self.frame_queue.full():
                self.frame_queue.put(annotated_frame)
            
            # Process alerts
            for anomaly in anomalies:
                alert = self.alert_manager.create_alert(anomaly, timestamp)
                self.alert_queue.put(alert)
                print(f"🚨 LIVE ALERT: {alert['type']} at {timestamp:.2f}s")
            
            frame_count += 1
            
            # Control processing speed
            time.sleep(1/30)  # 30 FPS max
        
        cap.release()
        print("🛑 Video processing stopped")
    
    def draw_annotations(self, frame, detections, anomalies):
        """Draw real-time annotations on frame"""
        annotated = frame.copy()
        
        # Draw object detections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            label = f"{detection['class']} ({detection['confidence']:.2f})"
            
            # Color based on class
            color = (0, 255, 0) if detection['class'] == 'person' else (255, 0, 0)
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw anomaly alerts with flashing effect
        current_time = time.time()
        flash = int(current_time * 4) % 2  # Flash every 0.25 seconds
        
        for anomaly in anomalies:
            if 'bbox' in anomaly:
                x1, y1, x2, y2 = anomaly['bbox']
                
                # Alert colors
                alert_colors = {
                    'loitering': (0, 165, 255),      # Orange
                    'object_abandonment': (0, 0, 255),  # Red
                    'unusual_movement': (255, 0, 255)   # Magenta
                }
                
                color = alert_colors.get(anomaly['type'], (0, 0, 255))
                
                if flash:  # Flashing effect
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)
                    
                    # Alert text
                    alert_text = f"ALERT: {anomaly['type'].replace('_', ' ').upper()}"
                    text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    
                    # Background for text
                    cv2.rectangle(annotated, (x1, y1-35), (x1+text_size[0]+10, y1-5), color, -1)
                    cv2.putText(annotated, alert_text, (x1+5, y1-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add timestamp and stats
        timestamp_text = f"Time: {time.strftime('%H:%M:%S')}"
        cv2.putText(annotated, timestamp_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        detection_count = len([d for d in detections if d['class'] == 'person'])
        stats_text = f"Persons: {detection_count} | Alerts: {len(anomalies)}"
        cv2.putText(annotated, stats_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated
    
    def start_processing(self, video_source):
        """Start video processing in background thread"""
        self.processing = True
        self.processing_thread = threading.Thread(
            target=self.process_video_stream, 
            args=(video_source,)
        )
        self.processing_thread.start()
    
    def stop_processing(self):
        """Stop video processing"""
        self.processing = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()

def create_streamlit_live_demo():
    """Create Streamlit live demo interface"""
    st.set_page_config(
        page_title="Live AI Surveillance Demo",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 Live AI Surveillance System Demo")
    st.markdown("Real-time anomaly detection with live video processing")
    
    # Initialize demo system
    if 'demo_system' not in st.session_state:
        st.session_state.demo_system = LiveSurveillanceDemo()
    
    demo = st.session_state.demo_system
    
    # Sidebar controls
    st.sidebar.header("🎮 Demo Controls")
    
    # Video source options
    video_option = st.sidebar.selectbox(
        "Select Video Source",
        ["Upload File", "Sample Videos", "Webcam (if available)"]
    )
    
    video_source = None
    
    if video_option == "Upload File":
        uploaded_file = st.sidebar.file_uploader(
            "Choose video file", 
            type=['mp4', 'avi', 'mov', 'mkv']
        )
        if uploaded_file:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_source = tmp_file.name
    
    elif video_option == "Sample Videos":
        sample_videos = {
            "Parking Lot Surveillance": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",
            "Pedestrian Area": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
        }
        
        selected_sample = st.sidebar.selectbox("Choose sample", list(sample_videos.keys()))
        
        if st.sidebar.button("Load Sample Video"):
            url = sample_videos[selected_sample]
            temp_path = f"temp_sample_{selected_sample.replace(' ', '_')}.mp4"
            
            if demo.download_video(url, temp_path):
                video_source = temp_path
                st.sidebar.success("Sample video loaded!")
    
    elif video_option == "Webcam (if available)":
        if st.sidebar.button("Use Webcam"):
            video_source = 0  # Default webcam
    
    # Processing controls
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("▶️ Start Demo", disabled=video_source is None):
            demo.start_processing(video_source)
            st.success("Demo started!")
    
    with col2:
        if st.button("⏹️ Stop Demo"):
            demo.stop_processing()
            st.info("Demo stopped!")
    
    # Main display area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎥 Live Video Feed")
        video_placeholder = st.empty()
        
        # Display current frame
        if demo.current_frame is not None:
            # Convert BGR to RGB for Streamlit
            rgb_frame = cv2.cvtColor(demo.current_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
        else:
            video_placeholder.info("No video feed. Start demo to begin processing.")
    
    with col2:
        st.header("🚨 Live Alerts")
        alert_placeholder = st.empty()
        
        # Display recent alerts
        recent_alerts = []
        while not demo.alert_queue.empty():
            try:
                alert = demo.alert_queue.get_nowait()
                recent_alerts.append(alert)
            except queue.Empty:
                break
        
        if recent_alerts:
            for alert in recent_alerts[-5:]:  # Show last 5 alerts
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
        else:
            alert_placeholder.info("No alerts yet. Monitoring for anomalies...")
    
    # Statistics section
    st.header("📊 Real-time Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Processing Status", "Active" if demo.processing else "Stopped")
    
    with col2:
        alert_count = len(demo.alert_manager.alerts)
        st.metric("Total Alerts", alert_count)
    
    with col3:
        if demo.current_frame is not None:
            st.metric("Video Status", "Live")
        else:
            st.metric("Video Status", "No Feed")
    
    with col4:
        st.metric("Detection Model", "YOLOv8")
    
    # Auto-refresh
    time.sleep(0.1)  # Small delay
    st.rerun()

def main():
    """Main function for live demo"""
    create_streamlit_live_demo()

if __name__ == "__main__":
    main()