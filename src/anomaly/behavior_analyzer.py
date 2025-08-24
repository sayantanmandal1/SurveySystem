"""
Behavioral Anomaly Detection
Detects loitering, object abandonment, and unusual movements
"""

import numpy as np
from collections import defaultdict, deque
import math

class BehaviorAnalyzer:
    def __init__(self, config):
        """Initialize behavior analyzer"""
        self.config = config
        self.loitering_threshold = config.get('loitering_threshold', 30)  # seconds
        self.abandonment_threshold = config.get('abandonment_threshold', 15)  # seconds
        self.movement_threshold = config.get('movement_threshold', 0.1)
        
        # Tracking data
        self.person_tracks = defaultdict(lambda: {
            'positions': deque(maxlen=100),
            'timestamps': deque(maxlen=100),
            'last_seen': 0
        })
        
        self.object_tracks = defaultdict(lambda: {
            'position': None,
            'first_seen': None,
            'last_person_nearby': None,
            'abandoned': False
        })
        
        self.track_id_counter = 0
        
    def analyze_frame(self, detections, timestamp):
        """Analyze frame for behavioral anomalies"""
        anomalies = []
        
        # Separate persons and objects
        persons = [d for d in detections if d['class'] == 'person']
        objects = [d for d in detections if d['class'] != 'person']
        
        # Update tracking
        self.update_person_tracking(persons, timestamp)
        self.update_object_tracking(objects, persons, timestamp)
        
        # Detect anomalies
        anomalies.extend(self.detect_loitering(timestamp))
        anomalies.extend(self.detect_object_abandonment(timestamp))
        anomalies.extend(self.detect_unusual_movement(timestamp))
        
        return anomalies
    
    def update_person_tracking(self, persons, timestamp):
        """Update person tracking data"""
        # Simple tracking based on position proximity
        matched_tracks = set()
        
        for person in persons:
            center = person['center']
            best_match = None
            best_distance = float('inf')
            
            # Find closest existing track
            for track_id, track_data in self.person_tracks.items():
                if track_data['positions'] and track_id not in matched_tracks:
                    last_pos = track_data['positions'][-1]
                    distance = self.euclidean_distance(center, last_pos)
                    
                    if distance < 100 and distance < best_distance:  # 100 pixel threshold
                        best_distance = distance
                        best_match = track_id
            
            # Update existing track or create new one
            if best_match is not None:
                track_id = best_match
                matched_tracks.add(track_id)
            else:
                track_id = self.track_id_counter
                self.track_id_counter += 1
            
            # Update track data
            self.person_tracks[track_id]['positions'].append(center)
            self.person_tracks[track_id]['timestamps'].append(timestamp)
            self.person_tracks[track_id]['last_seen'] = timestamp
            
            # Add track_id to person detection
            person['track_id'] = track_id
        
        # Clean up old tracks
        self.cleanup_old_tracks(timestamp)
    
    def update_object_tracking(self, objects, persons, timestamp):
        """Update object tracking for abandonment detection"""
        for obj in objects:
            obj_center = obj['center']
            
            # Find if this object matches an existing track
            matched_track = None
            for track_id, track_data in self.object_tracks.items():
                if track_data['position'] is not None:
                    distance = self.euclidean_distance(obj_center, track_data['position'])
                    if distance < 50:  # 50 pixel threshold for objects
                        matched_track = track_id
                        break
            
            if matched_track is None:
                # New object
                track_id = f"obj_{self.track_id_counter}"
                self.track_id_counter += 1
                self.object_tracks[track_id] = {
                    'position': obj_center,
                    'first_seen': timestamp,
                    'last_person_nearby': timestamp,
                    'abandoned': False,
                    'bbox': obj['bbox']
                }
            else:
                track_id = matched_track
            
            # Check if person is nearby
            person_nearby = False
            for person in persons:
                distance = self.euclidean_distance(obj_center, person['center'])
                if distance < 150:  # 150 pixel threshold for "nearby"
                    person_nearby = True
                    self.object_tracks[track_id]['last_person_nearby'] = timestamp
                    break
            
            obj['track_id'] = track_id
    
    def detect_loitering(self, current_timestamp):
        """Detect loitering behavior"""
        anomalies = []
        
        for track_id, track_data in self.person_tracks.items():
            if len(track_data['positions']) < 10:  # Need enough data points
                continue
            
            # Calculate time spent in area
            time_in_area = current_timestamp - track_data['timestamps'][0]
            
            if time_in_area > self.loitering_threshold:
                # Check if person has been relatively stationary
                positions = list(track_data['positions'])
                movement = self.calculate_total_movement(positions)
                
                if movement < self.movement_threshold * len(positions):
                    anomaly = {
                        'type': 'loitering',
                        'track_id': track_id,
                        'duration': time_in_area,
                        'bbox': self.get_person_bbox(track_id),
                        'confidence': min(time_in_area / self.loitering_threshold, 2.0)
                    }
                    anomalies.append(anomaly)
        
        return anomalies
    
    def detect_object_abandonment(self, current_timestamp):
        """Detect abandoned objects"""
        anomalies = []
        
        for track_id, track_data in self.object_tracks.items():
            if track_data['abandoned']:
                continue
            
            time_since_person = current_timestamp - track_data['last_person_nearby']
            
            if time_since_person > self.abandonment_threshold:
                self.object_tracks[track_id]['abandoned'] = True
                
                anomaly = {
                    'type': 'object_abandonment',
                    'track_id': track_id,
                    'duration': time_since_person,
                    'bbox': track_data['bbox'],
                    'confidence': min(time_since_person / self.abandonment_threshold, 2.0)
                }
                anomalies.append(anomaly)
        
        return anomalies
    
    def detect_unusual_movement(self, current_timestamp):
        """Detect unusual movement patterns"""
        anomalies = []
        
        for track_id, track_data in self.person_tracks.items():
            if len(track_data['positions']) < 5:
                continue
            
            # Calculate recent movement speed
            recent_positions = list(track_data['positions'])[-5:]
            recent_timestamps = list(track_data['timestamps'])[-5:]
            
            if len(recent_positions) >= 2:
                total_distance = 0
                for i in range(1, len(recent_positions)):
                    distance = self.euclidean_distance(recent_positions[i-1], recent_positions[i])
                    total_distance += distance
                
                time_diff = recent_timestamps[-1] - recent_timestamps[0]
                if time_diff > 0:
                    speed = total_distance / time_diff  # pixels per second
                    
                    # Detect unusually fast movement (running)
                    if speed > 200:  # Threshold for unusual speed
                        anomaly = {
                            'type': 'unusual_movement',
                            'subtype': 'fast_movement',
                            'track_id': track_id,
                            'speed': speed,
                            'bbox': self.get_person_bbox(track_id),
                            'confidence': min(speed / 200, 2.0)
                        }
                        anomalies.append(anomaly)
        
        return anomalies
    
    def euclidean_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def calculate_total_movement(self, positions):
        """Calculate total movement distance"""
        if len(positions) < 2:
            return 0
        
        total = 0
        for i in range(1, len(positions)):
            total += self.euclidean_distance(positions[i-1], positions[i])
        
        return total
    
    def get_person_bbox(self, track_id):
        """Get approximate bounding box for person track"""
        if track_id in self.person_tracks and self.person_tracks[track_id]['positions']:
            center = self.person_tracks[track_id]['positions'][-1]
            # Approximate person bounding box
            return [center[0]-30, center[1]-60, center[0]+30, center[1]+60]
        return [0, 0, 100, 100]
    
    def cleanup_old_tracks(self, current_timestamp, max_age=10):
        """Remove old tracks that haven't been seen recently"""
        to_remove = []
        
        for track_id, track_data in self.person_tracks.items():
            if current_timestamp - track_data['last_seen'] > max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.person_tracks[track_id]