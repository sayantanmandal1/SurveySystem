"""
Benchmark Evaluator for Avenue and UCSD Datasets
Evaluates anomaly detection performance using standard metrics
"""

import numpy as np
import cv2
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset_loader import DatasetLoader
from detection.yolo_detector import YOLODetector
from anomaly.behavior_analyzer import BehaviorAnalyzer

class BenchmarkEvaluator:
    def __init__(self, config_path="config/system_config.json"):
        """Initialize benchmark evaluator"""
        self.dataset_loader = DatasetLoader()
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except:
            config = self.get_default_config()
        
        self.detector = YOLODetector(config['yolo'])
        self.analyzer = BehaviorAnalyzer(config['anomaly'])
        
        self.results = {}
    
    def get_default_config(self):
        """Default configuration for evaluation"""
        return {
            "yolo": {
                "model_path": "yolov8n.pt",
                "confidence": 0.3,  # Lower threshold for evaluation
                "iou_threshold": 0.45
            },
            "anomaly": {
                "loitering_threshold": 20,  # Adjusted for datasets
                "abandonment_threshold": 10,
                "movement_threshold": 0.05
            }
        }
    
    def evaluate_avenue_dataset(self):
        """Evaluate on Avenue dataset"""
        print("🔍 Evaluating on Avenue Dataset...")
        
        videos, ground_truths = self.dataset_loader.load_avenue_dataset()
        
        if not videos:
            print("❌ Avenue dataset not available")
            return None
        
        all_predictions = []
        all_ground_truth = []
        
        for i, (video_path, gt_path) in enumerate(zip(videos[:5], ground_truths[:5])):  # Limit for demo
            print(f"Processing video {i+1}/{min(5, len(videos))}: {Path(video_path).name}")
            
            # Load ground truth
            gt = self.dataset_loader.load_ground_truth(gt_path)
            
            # Process video
            predictions = self.process_video_for_evaluation(video_path)
            
            if gt is not None and predictions is not None:
                # Align lengths
                min_len = min(len(gt), len(predictions))
                all_ground_truth.extend(gt[:min_len])
                all_predictions.extend(predictions[:min_len])
        
        if all_predictions and all_ground_truth:
            metrics = self.calculate_metrics(all_predictions, all_ground_truth)
            self.results['avenue'] = metrics
            print(f"✅ Avenue Results - AUC: {metrics['auc']:.3f}, Precision: {metrics['precision']:.3f}")
            return metrics
        
        return None
    
    def evaluate_ucsd_dataset(self):
        """Evaluate on UCSD dataset"""
        print("🔍 Evaluating on UCSD Dataset...")
        
        videos, ground_truths = self.dataset_loader.load_ucsd_dataset()
        
        if not videos:
            print("❌ UCSD dataset not available")
            return None
        
        all_predictions = []
        all_ground_truth = []
        
        for i, (video_info, gt_path) in enumerate(zip(videos[:3], ground_truths[:3])):  # Limit for demo
            print(f"Processing sequence {i+1}/{min(3, len(videos))}: {video_info['video_name']}")
            
            # Load ground truth
            gt = self.dataset_loader.load_ground_truth(gt_path)
            
            # Process frame sequence
            predictions = self.process_frame_sequence_for_evaluation(video_info)
            
            if gt is not None and predictions is not None:
                # Align lengths
                min_len = min(len(gt), len(predictions))
                all_ground_truth.extend(gt[:min_len])
                all_predictions.extend(predictions[:min_len])
        
        if all_predictions and all_ground_truth:
            metrics = self.calculate_metrics(all_predictions, all_ground_truth)
            self.results['ucsd'] = metrics
            print(f"✅ UCSD Results - AUC: {metrics['auc']:.3f}, Precision: {metrics['precision']:.3f}")
            return metrics
        
        return None
    
    def process_video_for_evaluation(self, video_path):
        """Process single video and return anomaly scores per frame"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_scores = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_count / fps
            
            # Detect objects
            detections = self.detector.detect(frame)
            
            # Analyze for anomalies
            anomalies = self.analyzer.analyze_frame(detections, timestamp)
            
            # Calculate frame-level anomaly score
            frame_score = self.calculate_frame_anomaly_score(anomalies)
            frame_scores.append(frame_score)
            
            frame_count += 1
            
            # Progress update
            if frame_count % 100 == 0:
                print(f"  Processed {frame_count} frames")
        
        cap.release()
        return frame_scores
    
    def process_frame_sequence_for_evaluation(self, video_info):
        """Process UCSD frame sequence and return anomaly scores"""
        frame_paths = video_info['frames']
        frame_scores = []
        
        fps = 30  # Assume 30 FPS for UCSD
        
        for i, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            
            timestamp = i / fps
            
            # Detect objects
            detections = self.detector.detect(frame)
            
            # Analyze for anomalies
            anomalies = self.analyzer.analyze_frame(detections, timestamp)
            
            # Calculate frame-level anomaly score
            frame_score = self.calculate_frame_anomaly_score(anomalies)
            frame_scores.append(frame_score)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(frame_paths)} frames")
        
        return frame_scores
    
    def calculate_frame_anomaly_score(self, anomalies):
        """Calculate anomaly score for a single frame"""
        if not anomalies:
            return 0.0
        
        # Combine anomaly scores
        total_score = 0.0
        for anomaly in anomalies:
            confidence = anomaly.get('confidence', 1.0)
            
            # Weight different anomaly types
            if anomaly['type'] == 'loitering':
                total_score += confidence * 0.8
            elif anomaly['type'] == 'object_abandonment':
                total_score += confidence * 1.0
            elif anomaly['type'] == 'unusual_movement':
                total_score += confidence * 0.6
        
        # Normalize to [0, 1]
        return min(total_score, 1.0)
    
    def calculate_metrics(self, predictions, ground_truth):
        """Calculate evaluation metrics"""
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # Ensure binary ground truth
        ground_truth = (ground_truth > 0).astype(int)
        
        # Calculate AUC
        try:
            auc_score = roc_auc_score(ground_truth, predictions)
        except:
            auc_score = 0.0
        
        # Calculate precision-recall AUC
        try:
            precision, recall, _ = precision_recall_curve(ground_truth, predictions)
            pr_auc = auc(recall, precision)
        except:
            pr_auc = 0.0
        
        # Calculate precision at different thresholds
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        precision_at_threshold = {}
        
        for threshold in thresholds:
            binary_pred = (predictions >= threshold).astype(int)
            tp = np.sum((binary_pred == 1) & (ground_truth == 1))
            fp = np.sum((binary_pred == 1) & (ground_truth == 0))
            
            if tp + fp > 0:
                precision_at_threshold[threshold] = tp / (tp + fp)
            else:
                precision_at_threshold[threshold] = 0.0
        
        return {
            'auc': auc_score,
            'pr_auc': pr_auc,
            'precision': precision_at_threshold[0.5],
            'precision_at_threshold': precision_at_threshold,
            'total_frames': len(predictions),
            'anomaly_frames': np.sum(ground_truth),
            'detected_frames': np.sum(predictions > 0.5)
        }
    
    def generate_evaluation_report(self, output_path="evaluation_report.json"):
        """Generate comprehensive evaluation report"""
        report = {
            'evaluation_timestamp': str(np.datetime64('now')),
            'datasets_evaluated': list(self.results.keys()),
            'results': self.results,
            'system_config': {
                'detection_model': 'YOLOv8',
                'anomaly_types': ['loitering', 'object_abandonment', 'unusual_movement']
            }
        }
        
        # Calculate overall performance
        if self.results:
            all_aucs = [result['auc'] for result in self.results.values()]
            report['overall_performance'] = {
                'mean_auc': np.mean(all_aucs),
                'std_auc': np.std(all_aucs)
            }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Evaluation report saved to {output_path}")
        return report
    
    def plot_results(self, output_dir="evaluation_plots"):
        """Generate evaluation plots"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        if not self.results:
            print("No results to plot")
            return
        
        # AUC comparison plot
        datasets = list(self.results.keys())
        aucs = [self.results[dataset]['auc'] for dataset in datasets]
        
        plt.figure(figsize=(10, 6))
        plt.bar(datasets, aucs, color=['skyblue', 'lightcoral'])
        plt.title('Anomaly Detection Performance (AUC Score)')
        plt.ylabel('AUC Score')
        plt.ylim(0, 1)
        
        for i, auc in enumerate(aucs):
            plt.text(i, auc + 0.02, f'{auc:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'auc_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Precision at threshold plot
        plt.figure(figsize=(12, 8))
        
        for i, (dataset, results) in enumerate(self.results.items()):
            thresholds = list(results['precision_at_threshold'].keys())
            precisions = list(results['precision_at_threshold'].values())
            
            plt.subplot(2, 1, i + 1)
            plt.plot(thresholds, precisions, marker='o', linewidth=2, markersize=6)
            plt.title(f'{dataset.upper()} Dataset - Precision vs Threshold')
            plt.xlabel('Threshold')
            plt.ylabel('Precision')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'precision_threshold.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Plots saved to {output_dir}")

def main():
    """Main evaluation function"""
    print("🎯 Benchmark Evaluation on Avenue and UCSD Datasets")
    print("=" * 60)
    
    evaluator = BenchmarkEvaluator()
    
    # Evaluate on both datasets
    avenue_results = evaluator.evaluate_avenue_dataset()
    ucsd_results = evaluator.evaluate_ucsd_dataset()
    
    # Generate report and plots
    if avenue_results or ucsd_results:
        report = evaluator.generate_evaluation_report()
        evaluator.plot_results()
        
        print("\n📊 EVALUATION SUMMARY")
        print("=" * 30)
        
        for dataset, results in evaluator.results.items():
            print(f"\n{dataset.upper()} Dataset:")
            print(f"  AUC Score: {results['auc']:.3f}")
            print(f"  Precision@0.5: {results['precision']:.3f}")
            print(f"  Total Frames: {results['total_frames']}")
            print(f"  Anomaly Frames: {results['anomaly_frames']}")
            print(f"  Detected Frames: {results['detected_frames']}")
        
        if 'overall_performance' in report:
            print(f"\nOverall Mean AUC: {report['overall_performance']['mean_auc']:.3f}")
    
    else:
        print("❌ No datasets available for evaluation")
        print("Please download datasets manually:")
        print("Avenue: http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html")
        print("UCSD: http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm")

if __name__ == "__main__":
    main()