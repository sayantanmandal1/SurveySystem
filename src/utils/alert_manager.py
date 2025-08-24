"""
Alert Management System
Handles alert creation, storage, and dashboard updates
"""

import json
import os
from datetime import datetime
from pathlib import Path

class AlertManager:
    def __init__(self, config):
        """Initialize alert manager"""
        self.config = config
        self.save_path = config.get('save_path', 'data/alerts.json')
        self.dashboard_update = config.get('dashboard_update', True)
        
        # Ensure data directory exists
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing alerts
        self.alerts = self.load_alerts()
    
    def create_alert(self, anomaly, timestamp):
        """Create alert from anomaly detection"""
        alert = {
            'id': len(self.alerts) + 1,
            'timestamp': timestamp,
            'datetime': datetime.now().isoformat(),
            'type': anomaly['type'],
            'confidence': anomaly.get('confidence', 1.0),
            'details': self.extract_details(anomaly),
            'status': 'active',
            'bbox': anomaly.get('bbox', [])
        }
        
        return alert
    
    def extract_details(self, anomaly):
        """Extract relevant details from anomaly"""
        details = {}
        
        if anomaly['type'] == 'loitering':
            details['duration'] = f"{anomaly.get('duration', 0):.1f} seconds"
            details['description'] = f"Person loitering for {details['duration']}"
        
        elif anomaly['type'] == 'object_abandonment':
            details['duration'] = f"{anomaly.get('duration', 0):.1f} seconds"
            details['description'] = f"Object abandoned for {details['duration']}"
        
        elif anomaly['type'] == 'unusual_movement':
            speed = anomaly.get('speed', 0)
            details['speed'] = f"{speed:.1f} pixels/second"
            details['description'] = f"Unusual fast movement detected ({details['speed']})"
        
        details['track_id'] = anomaly.get('track_id', 'unknown')
        
        return details
    
    def save_alerts(self, new_alerts):
        """Save alerts to file"""
        self.alerts.extend(new_alerts)
        
        try:
            # Convert numpy types to native Python types
            serializable_alerts = []
            for alert in self.alerts:
                serializable_alert = {}
                for key, value in alert.items():
                    if hasattr(value, 'item'):  # numpy scalar
                        serializable_alert[key] = value.item()
                    elif isinstance(value, dict):
                        # Handle nested dictionaries
                        serializable_dict = {}
                        for k, v in value.items():
                            if hasattr(v, 'item'):
                                serializable_dict[k] = v.item()
                            else:
                                serializable_dict[k] = v
                        serializable_alert[key] = serializable_dict
                    else:
                        serializable_alert[key] = value
                serializable_alerts.append(serializable_alert)
            
            with open(self.save_path, 'w') as f:
                json.dump(serializable_alerts, f, indent=2)
        except Exception as e:
            print(f"Error saving alerts: {e}")
    
    def load_alerts(self):
        """Load existing alerts from file"""
        try:
            if os.path.exists(self.save_path):
                with open(self.save_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading alerts: {e}")
        
        return []
    
    def get_recent_alerts(self, limit=50):
        """Get recent alerts for dashboard"""
        return sorted(self.alerts, key=lambda x: x['timestamp'], reverse=True)[:limit]
    
    def get_alert_summary(self):
        """Get summary statistics of alerts"""
        if not self.alerts:
            return {
                'total': 0,
                'by_type': {},
                'recent_count': 0
            }
        
        # Count by type
        by_type = {}
        recent_count = 0
        current_time = datetime.now()
        
        for alert in self.alerts:
            alert_type = alert['type']
            by_type[alert_type] = by_type.get(alert_type, 0) + 1
            
            # Count recent alerts (last hour)
            try:
                alert_time = datetime.fromisoformat(alert['datetime'])
                if (current_time - alert_time).total_seconds() < 3600:
                    recent_count += 1
            except:
                pass
        
        return {
            'total': len(self.alerts),
            'by_type': by_type,
            'recent_count': recent_count
        }
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts = []
        try:
            with open(self.save_path, 'w') as f:
                json.dump([], f)
        except Exception as e:
            print(f"Error clearing alerts: {e}")