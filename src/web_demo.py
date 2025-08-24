#!/usr/bin/env python3
"""
Web-based Live Demo for AI Surveillance System
Streamlit app with video URL processing and real-time alerts
"""

import streamlit as st
import cv2
import numpy as np
import requests
import tempfile
import threading
import time
import queue
import json
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from detection.yolo_detector import YOLODetector
from anomaly.behavior_analyzer import BehaviorAnalyzer
from utils.alert_manager import AlertManager

class WebDemo:
    def __init__(self):
        """Initialize web demo"""
        self.config = {
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
            "alerts": {
                "save_path": "data/web_demo_alerts.json",
                "dashboard_update": True
            }
        }
        
        try:
            self.detector = YOLODetector(self.config['yolo'])
            self.analyzer = BehaviorAnalyzer(self.config['anomaly'])
            self.alert_manager = AlertManager(self.config['alerts'])
            self.initialized = True
        except Exception as e:
            st.error(f"Failed to initialize AI models: {e}")
            self.initialized = False
        
        self.processing = False
        self.current_frame = None
        self.alerts = []
        self.frame_count = 0
    
    def download_video(self, url):
        """Download video from URL"""
        try:
            with st.spinner(f"Downloading video from URL..."):
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                progress_bar = st.progress(0)
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = downloaded / total_size
                            progress_bar.progress(progress)
                
                temp_file.close()
                progress_bar.empty()
                
                return temp_file.name
        except Exception as e:
            st.error(f"Failed to download video: {e}")
            return None
    
    def process_video(self, video_path, max_frames=900):  # ~30 seconds at 30fps
        """Process video and generate alerts"""
        if not self.initialized:
            st.error("AI models not initialized")
            return
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error(f"Cannot open video: {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create placeholders for live updates
        col1, col2 = st.columns([2, 1])
        
        with col1:
            frame_placeholder = st.empty()
        
        with col2:
            alert_placeholder = st.empty()
            stats_placeholder = st.empty()
        
        frame_count = 0
        alerts = []
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_count / fps
            
            # Update progress
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames} ({timestamp:.1f}s)")
            
            # Detect objects
            detections = self.detector.detect(frame)
            
            # Analyze for anomalies
            anomalies = self.analyzer.analyze_frame(detections, timestamp)
            
            # Create alerts
            for anomaly in anomalies:
                alert = self.alert_manager.create_alert(anomaly, timestamp)
                alerts.append(alert)
            
            # Draw annotations
            annotated_frame = self.draw_annotations(frame, detections, anomalies, timestamp)
            
            # Update display every 10 frames for performance
            if frame_count % 10 == 0:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                
                # Update alerts display
                recent_alerts = alerts[-5:] if len(alerts) > 5 else alerts
                alert_text = "🚨 **Recent Alerts:**\n\n"
                
                if recent_alerts:
                    for alert in recent_alerts:
                        alert_type = alert['type'].replace('_', ' ').title()
                        alert_text += f"• **{alert_type}** at {alert['timestamp']:.1f}s\n"
                else:
                    alert_text += "No alerts detected yet..."
                
                alert_placeholder.markdown(alert_text)
                
                # Update stats
                person_count = len([d for d in detections if d['class'] == 'person'])
                stats_text = f"""
                **📊 Live Stats:**
                - Frame: {frame_count}/{total_frames}
                - Time: {timestamp:.1f}s
                - Persons: {person_count}
                - Total Alerts: {len(alerts)}
                """
                stats_placeholder.markdown(stats_text)
            
            frame_count += 1
            
            # Small delay to make it feel more "live"
            time.sleep(0.01)
        
        cap.release()
        progress_bar.empty()
        status_text.empty()
        
        # Save alerts
        self.alert_manager.save_alerts(alerts)
        
        return alerts, annotated_frame
    
    def draw_annotations(self, frame, detections, anomalies, timestamp):
        """Draw annotations on frame"""
        annotated = frame.copy()
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            label = f"{detection['class']} ({detection['confidence']:.2f})"
            color = (0, 255, 0) if detection['class'] == 'person' else (255, 0, 0)
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw anomaly alerts
        for anomaly in anomalies:
            if 'bbox' in anomaly:
                x1, y1, x2, y2 = anomaly['bbox']
                
                colors = {
                    'loitering': (0, 165, 255),
                    'object_abandonment': (0, 0, 255),
                    'unusual_movement': (255, 0, 255)
                }
                color = colors.get(anomaly['type'], (0, 0, 255))
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)
                
                alert_text = f"ALERT: {anomaly['type'].replace('_', ' ').upper()}"
                cv2.putText(annotated, alert_text, (x1, y1-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add timestamp
        cv2.putText(annotated, f"Time: {timestamp:.2f}s", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="AI Surveillance Live Demo",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 AI-Powered Surveillance System - Live Demo")
    st.markdown("Real-time anomaly detection with live video processing")
    
    # Initialize demo
    if 'demo' not in st.session_state:
        st.session_state.demo = WebDemo()
    
    demo = st.session_state.demo
    
    if not demo.initialized:
        st.error("⚠️ AI models failed to initialize. Please check your setup.")
        st.stop()
    
    # Sidebar
    st.sidebar.header("🎮 Demo Controls")
    
    # Video input options
    input_method = st.sidebar.selectbox(
        "Choose Input Method",
        ["Video URL", "Upload File", "Sample Videos"]
    )
    
    video_source = None
    
    if input_method == "Video URL":
        video_url = st.sidebar.text_input(
            "Enter video URL",
            placeholder="https://example.com/video.mp4"
        )
        
        if video_url and st.sidebar.button("Load Video from URL"):
            video_source = demo.download_video(video_url)
            if video_source:
                st.sidebar.success("✅ Video downloaded successfully!")
    
    elif input_method == "Upload File":
        uploaded_file = st.sidebar.file_uploader(
            "Choose video file",
            type=['mp4', 'avi', 'mov', 'mkv']
        )
        
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_source = tmp_file.name
            st.sidebar.success("✅ Video uploaded successfully!")
    
    elif input_method == "Sample Videos":
        sample_options = {
            "Create Demo Video": "demo",
            "Parking Lot Sample": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4"
        }
        
        selected_sample = st.sidebar.selectbox("Choose sample", list(sample_options.keys()))
        
        if st.sidebar.button("Load Sample"):
            if selected_sample == "Create Demo Video":
                # Create a quick demo video
                st.sidebar.info("Creating demo video...")
                # Use the quick_demo function
                import sys
                sys.path.append('.')
                from quick_demo import create_demo_video
                video_source = create_demo_video()
                st.sidebar.success("✅ Demo video created!")
            else:
                url = sample_options[selected_sample]
                video_source = demo.download_video(url)
                if video_source:
                    st.sidebar.success("✅ Sample video loaded!")
    
    # Processing controls
    st.sidebar.markdown("---")
    
    # Configuration
    st.sidebar.subheader("⚙️ Detection Settings")
    loitering_threshold = st.sidebar.slider("Loitering Threshold (s)", 5, 30, 12)
    abandonment_threshold = st.sidebar.slider("Abandonment Threshold (s)", 3, 20, 8)
    
    # Update config
    demo.config['anomaly']['loitering_threshold'] = loitering_threshold
    demo.config['anomaly']['abandonment_threshold'] = abandonment_threshold
    demo.analyzer = BehaviorAnalyzer(demo.config['anomaly'])
    
    # Main processing
    if video_source and st.button("🚀 Start Live Processing", type="primary"):
        st.markdown("## 🎬 Live Processing Results")
        
        with st.spinner("Initializing AI models..."):
            time.sleep(1)  # Brief pause for effect
        
        # Process video
        alerts, final_frame = demo.process_video(video_source)
        
        # Show final results
        st.success(f"✅ Processing complete! Generated {len(alerts)} alerts")
        
        # Results summary
        if alerts:
            st.markdown("## 📊 Final Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Alerts", len(alerts))
            
            with col2:
                alert_types = {}
                for alert in alerts:
                    alert_type = alert['type']
                    alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
                
                most_common = max(alert_types.items(), key=lambda x: x[1]) if alert_types else ("None", 0)
                st.metric("Most Common Alert", most_common[0].replace('_', ' ').title())
            
            with col3:
                avg_confidence = sum(alert['confidence'] for alert in alerts) / len(alerts)
                st.metric("Avg Confidence", f"{avg_confidence:.2f}")
            
            # Alert breakdown
            st.markdown("### 🚨 Alert Breakdown")
            for alert_type, count in alert_types.items():
                st.write(f"• **{alert_type.replace('_', ' ').title()}**: {count} alerts")
            
            # Timeline
            st.markdown("### ⏰ Alert Timeline")
            for alert in alerts[-10:]:  # Show last 10
                alert_type = alert['type'].replace('_', ' ').title()
                timestamp = alert['timestamp']
                confidence = alert['confidence']
                st.write(f"🔸 **{alert_type}** at {timestamp:.1f}s (confidence: {confidence:.2f})")
        
        else:
            st.info("No anomalies detected in this video.")
    
    elif not video_source:
        st.info("👆 Please select a video source from the sidebar to begin the demo.")
    
    # Information section
    st.markdown("---")
    st.markdown("## ℹ️ About This Demo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔍 Detection Capabilities:**
        - Person and object detection using YOLOv8
        - Loitering behavior analysis
        - Abandoned object detection
        - Unusual movement patterns
        """)
    
    with col2:
        st.markdown("""
        **⚡ Real-time Features:**
        - Live video processing
        - Instant alert generation
        - Confidence scoring
        - Configurable thresholds
        """)

if __name__ == "__main__":
    main()