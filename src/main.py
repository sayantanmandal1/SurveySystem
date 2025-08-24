#!/usr/bin/env python3
"""
AI-Powered Surveillance System - Main Entry Point
Processes video feeds and detects behavioral anomalies
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from detection.yolo_detector import YOLODetector
from anomaly.behavior_analyzer import BehaviorAnalyzer
from utils.alert_manager import AlertManager
from data.dataset_loader import DatasetLoader

class SurveillanceSystem:
    def __init__(self, config_path="config/system_config.json"):
        """Initialize the surveillance system"""
        self.config = self.load_config(config_path)
        self.detector = YOLODetector(self.config['yolo'])
        self.analyzer = BehaviorAnalyzer(self.config['anomaly'])
        self.alert_manager = AlertManager(self.config['alerts'])
        
    def load_config(self, config_path):
        """Load system configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.get_default_config()
    
    def get_default_config(self):
        """Default configuration"""
        return {
            "yolo": {
                "model_path": "models/yolov5s.pt",
                "confidence": 0.5,
                "iou_threshold": 0.45
            },
            "anomaly": {
                "loitering_threshold": 30,  # seconds
                "abandonment_threshold": 15,  # seconds
                "movement_threshold": 0.1
            },
            "alerts": {
                "save_path": "data/alerts.json",
                "dashboard_update": True
            }
        }
    
    def process_video(self, video_path, output_path=None):
        """Process video and detect anomalies"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        alerts = []
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_count / fps
            
            # Detect objects and people
            detections = self.detector.detect(frame)
            
            # Analyze behavior for anomalies
            anomalies = self.analyzer.analyze_frame(detections, timestamp)
            
            # Process any detected anomalies
            for anomaly in anomalies:
                alert = self.alert_manager.create_alert(anomaly, timestamp)
                alerts.append(alert)
                print(f"ALERT: {alert['type']} detected at {alert['timestamp']:.2f}s")
            
            # Draw detections and alerts on frame
            annotated_frame = self.draw_annotations(frame, detections, anomalies)
            
            if writer:
                writer.write(annotated_frame)
            
            frame_count += 1
            
            # Progress update every 100 frames
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames ({timestamp:.1f}s)")
        
        cap.release()
        if writer:
            writer.release()
        
        # Save alerts
        self.alert_manager.save_alerts(alerts)
        
        print(f"Processing complete. {len(alerts)} alerts detected.")
        return alerts
    
    def draw_annotations(self, frame, detections, anomalies):
        """Draw bounding boxes and alerts on frame"""
        annotated = frame.copy()
        
        # Draw object detections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            label = f"{detection['class']} ({detection['confidence']:.2f})"
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw anomaly alerts
        for anomaly in anomalies:
            if 'bbox' in anomaly:
                x1, y1, x2, y2 = anomaly['bbox']
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(annotated, f"ALERT: {anomaly['type']}", 
                           (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return annotated
    
    def process_benchmark_dataset(self, dataset_name, output_dir=None):
        """Process benchmark dataset (Avenue or UCSD)"""
        dataset_loader = DatasetLoader()
        all_alerts = []
        
        if dataset_name == 'avenue':
            videos, ground_truths = dataset_loader.load_avenue_dataset()
            if not videos:
                print("Avenue dataset not found. Please download it first.")
                return []
            
            print(f"Processing {len(videos)} Avenue videos...")
            for i, video_path in enumerate(videos[:3]):  # Process first 3 for demo
                print(f"Processing video {i+1}/{min(3, len(videos))}: {Path(video_path).name}")
                
                output_path = None
                if output_dir:
                    output_path = f"{output_dir}/avenue_processed_{i+1}.mp4"
                
                alerts = self.process_video(video_path, output_path)
                all_alerts.extend(alerts)
        
        elif dataset_name == 'ucsd':
            videos, ground_truths = dataset_loader.load_ucsd_dataset()
            if not videos:
                print("UCSD dataset not found. Please download it first.")
                return []
            
            print(f"Processing {len(videos)} UCSD sequences...")
            for i, video_info in enumerate(videos[:2]):  # Process first 2 for demo
                print(f"Processing sequence {i+1}/{min(2, len(videos))}: {video_info['video_name']}")
                
                # Convert frames to video first
                temp_video = f"temp_ucsd_{i}.mp4"
                if dataset_loader.create_video_from_frames(video_info['frames'], temp_video):
                    output_path = None
                    if output_dir:
                        output_path = f"{output_dir}/ucsd_processed_{i+1}.mp4"
                    
                    alerts = self.process_video(temp_video, output_path)
                    all_alerts.extend(alerts)
                    
                    # Clean up temp file
                    Path(temp_video).unlink(missing_ok=True)
        
        return all_alerts

def main():
    parser = argparse.ArgumentParser(description='AI-Powered Surveillance System')
    parser.add_argument('--video', help='Input video path')
    parser.add_argument('--dataset', choices=['avenue', 'ucsd'], help='Use benchmark dataset')
    parser.add_argument('--output', help='Output video path (optional)')
    parser.add_argument('--config', default='config/system_config.json', 
                       help='Configuration file path')
    parser.add_argument('--evaluate', action='store_true', 
                       help='Run evaluation on benchmark datasets')
    
    args = parser.parse_args()
    
    if args.evaluate:
        # Run benchmark evaluation
        from evaluation.benchmark_evaluator import BenchmarkEvaluator
        evaluator = BenchmarkEvaluator(args.config)
        evaluator.evaluate_avenue_dataset()
        evaluator.evaluate_ucsd_dataset()
        evaluator.generate_evaluation_report()
        evaluator.plot_results()
        return
    
    # Initialize system
    system = SurveillanceSystem(args.config)
    
    if args.dataset:
        # Process benchmark dataset
        alerts = system.process_benchmark_dataset(args.dataset, args.output)
    elif args.video:
        # Process single video
        alerts = system.process_video(args.video, args.output)
    else:
        print("Please specify either --video or --dataset")
        return
    
    print(f"\nSummary:")
    print(f"Total alerts: {len(alerts)}")
    
    # Group alerts by type
    alert_types = {}
    for alert in alerts:
        alert_type = alert['type']
        alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
    
    for alert_type, count in alert_types.items():
        print(f"- {alert_type}: {count}")

if __name__ == "__main__":
    main()