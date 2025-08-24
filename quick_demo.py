#!/usr/bin/env python3
"""
Quick Demo Script - Process sample videos and show results
"""

import cv2
import numpy as np
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append('src')

from detection.yolo_detector import YOLODetector
from anomaly.behavior_analyzer import BehaviorAnalyzer
from utils.alert_manager import AlertManager

def create_demo_video():
    """Create a demo video with simulated surveillance scenario"""
    print("🎬 Creating demo surveillance video...")
    
    # Video parameters
    width, height = 640, 480
    fps = 30
    duration = 30  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('demo_surveillance.mp4', fourcc, fps, (width, height))
    
    for frame_num in range(total_frames):
        # Create background (parking lot style)
        frame = np.ones((height, width, 3), dtype=np.uint8) * 100
        
        # Add parking lines
        for i in range(0, width, 80):
            cv2.line(frame, (i, 0), (i, height), (255, 255, 255), 2)
        for i in range(0, height, 60):
            cv2.line(frame, (0, i), (width, i), (255, 255, 255), 1)
        
        # Simulate person movement
        time_progress = frame_num / total_frames
        
        # Person 1: Normal walking
        person1_x = int(50 + time_progress * 400)
        person1_y = 200
        cv2.rectangle(frame, (person1_x-15, person1_y-30), (person1_x+15, person1_y+30), (0, 255, 0), -1)
        cv2.putText(frame, "Person", (person1_x-20, person1_y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Person 2: Loitering (stays in one area)
        if time_progress > 0.3:
            person2_x = 300 + int(np.sin(frame_num * 0.1) * 10)  # Small movement
            person2_y = 150 + int(np.cos(frame_num * 0.1) * 5)
            cv2.rectangle(frame, (person2_x-15, person2_y-30), (person2_x+15, person2_y+30), (0, 255, 255), -1)
            cv2.putText(frame, "Loitering", (person2_x-25, person2_y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Abandoned object (bag)
        if time_progress > 0.5:
            cv2.rectangle(frame, (400, 300), (420, 320), (0, 0, 255), -1)
            cv2.putText(frame, "Bag", (395, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Person 3: Fast movement (running)
        if time_progress > 0.7:
            person3_x = int(100 + (time_progress - 0.7) * 1000)  # Fast movement
            person3_y = 350
            if person3_x < width:
                cv2.rectangle(frame, (person3_x-15, person3_y-30), (person3_x+15, person3_y+30), (255, 0, 255), -1)
                cv2.putText(frame, "Running", (person3_x-25, person3_y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add timestamp
        timestamp = frame_num / fps
        cv2.putText(frame, f"Time: {timestamp:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "AI Surveillance Demo", (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print("✅ Demo video created: demo_surveillance.mp4")
    return "demo_surveillance.mp4"

def process_demo_video(video_path):
    """Process demo video and show real-time results"""
    print(f"🔍 Processing demo video: {video_path}")
    
    # Initialize system
    config = {
        "yolo": {
            "model_path": "yolov8n.pt",
            "confidence": 0.3,
            "iou_threshold": 0.45
        },
        "anomaly": {
            "loitering_threshold": 10,  # Reduced for demo
            "abandonment_threshold": 5,
            "movement_threshold": 0.05
        },
        "alerts": {
            "save_path": "data/demo_alerts.json",
            "dashboard_update": True
        }
    }
    
    detector = YOLODetector(config['yolo'])
    analyzer = BehaviorAnalyzer(config['anomaly'])
    alert_manager = AlertManager(config['alerts'])
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # Create output video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('demo_output.mp4', fourcc, fps, (width, height))
    
    frame_count = 0
    alerts = []
    
    print("🎥 Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = frame_count / fps
        
        # Detect objects
        detections = detector.detect(frame)
        
        # Analyze for anomalies
        anomalies = analyzer.analyze_frame(detections, timestamp)
        
        # Create alerts
        for anomaly in anomalies:
            alert = alert_manager.create_alert(anomaly, timestamp)
            alerts.append(alert)
            print(f"🚨 ALERT: {alert['type']} detected at {timestamp:.2f}s (confidence: {alert['confidence']:.2f})")
        
        # Draw annotations
        annotated_frame = draw_demo_annotations(frame, detections, anomalies, timestamp)
        
        # Write frame
        out.write(annotated_frame)
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count} frames ({timestamp:.1f}s)")
    
    cap.release()
    out.release()
    
    # Save alerts
    alert_manager.save_alerts(alerts)
    
    print(f"✅ Processing complete!")
    print(f"📊 Results:")
    print(f"  - Total frames processed: {frame_count}")
    print(f"  - Total alerts generated: {len(alerts)}")
    print(f"  - Output video: demo_output.mp4")
    print(f"  - Alerts saved to: {config['alerts']['save_path']}")
    
    # Alert summary
    alert_types = {}
    for alert in alerts:
        alert_type = alert['type']
        alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
    
    print(f"📈 Alert breakdown:")
    for alert_type, count in alert_types.items():
        print(f"  - {alert_type.replace('_', ' ').title()}: {count}")
    
    return alerts

def draw_demo_annotations(frame, detections, anomalies, timestamp):
    """Draw annotations on demo frame"""
    annotated = frame.copy()
    
    # Draw detections
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        label = f"{detection['class']} ({detection['confidence']:.2f})"
        color = (0, 255, 0) if detection['class'] == 'person' else (255, 0, 0)
        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw anomaly alerts
    for anomaly in anomalies:
        if 'bbox' in anomaly:
            x1, y1, x2, y2 = anomaly['bbox']
            
            # Alert colors
            colors = {
                'loitering': (0, 165, 255),
                'object_abandonment': (0, 0, 255),
                'unusual_movement': (255, 0, 255)
            }
            color = colors.get(anomaly['type'], (0, 0, 255))
            
            # Flashing effect
            flash = int(timestamp * 4) % 2
            if flash:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)
                
                alert_text = f"ALERT: {anomaly['type'].replace('_', ' ').upper()}"
                cv2.putText(annotated, alert_text, (x1, y1-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Add status overlay
    overlay_height = 80
    overlay = np.zeros((overlay_height, annotated.shape[1], 3), dtype=np.uint8)
    overlay[:] = (0, 0, 0)  # Black background
    
    # Add text to overlay
    cv2.putText(overlay, f"AI Surveillance System - Time: {timestamp:.2f}s", 
               (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, f"Detections: {len(detections)} | Alerts: {len(anomalies)}", 
               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Blend overlay with frame
    annotated[:overlay_height] = cv2.addWeighted(annotated[:overlay_height], 0.7, overlay, 0.3, 0)
    
    return annotated

def main():
    """Main demo function"""
    print("🎯 AI Surveillance System - Quick Demo")
    print("=" * 50)
    
    # Create demo video
    demo_video = create_demo_video()
    
    # Process the demo video
    alerts = process_demo_video(demo_video)
    
    print("\n🎉 DEMO COMPLETE!")
    print("=" * 30)
    print("📁 Generated files:")
    print("  - demo_surveillance.mp4 (input)")
    print("  - demo_output.mp4 (processed with alerts)")
    print("  - data/demo_alerts.json (alert data)")
    
    print("\n🎬 Next steps:")
    print("1. Play demo_output.mp4 to see the processed video")
    print("2. Run: streamlit run src/dashboard/app.py")
    print("3. View alerts in the dashboard")
    
    print("\n💡 For live demo:")
    print("Run: streamlit run src/live_demo.py")

if __name__ == "__main__":
    main()