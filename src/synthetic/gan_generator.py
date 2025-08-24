"""
GAN-based Synthetic Video Generation for Edge Cases
Generates synthetic surveillance scenarios for training
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path

class SimpleVideoGAN:
    """Simple GAN for generating synthetic surveillance scenarios"""
    
    def __init__(self, config=None):
        """Initialize GAN generator"""
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = None
        
    def generate_synthetic_scenario(self, scenario_type, duration=10, fps=30):
        """Generate synthetic video scenario"""
        frames = []
        
        if scenario_type == "loitering":
            frames = self.generate_loitering_scenario(duration, fps)
        elif scenario_type == "abandonment":
            frames = self.generate_abandonment_scenario(duration, fps)
        elif scenario_type == "unusual_movement":
            frames = self.generate_unusual_movement_scenario(duration, fps)
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
        
        return frames
    
    def generate_loitering_scenario(self, duration, fps):
        """Generate synthetic loitering scenario"""
        frames = []
        total_frames = duration * fps
        
        # Create background
        background = self.create_background()
        
        # Person position (stays mostly static)
        center_x, center_y = 400, 300
        
        for frame_idx in range(total_frames):
            frame = background.copy()
            
            # Add slight random movement to simulate loitering
            noise_x = np.random.randint(-10, 10)
            noise_y = np.random.randint(-5, 5)
            
            person_x = center_x + noise_x
            person_y = center_y + noise_y
            
            # Draw person (simple rectangle for demo)
            cv2.rectangle(frame, 
                         (person_x-20, person_y-40), 
                         (person_x+20, person_y+40), 
                         (0, 255, 0), -1)
            
            # Add timestamp
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            frames.append(frame)
        
        return frames
    
    def generate_abandonment_scenario(self, duration, fps):
        """Generate synthetic object abandonment scenario"""
        frames = []
        total_frames = duration * fps
        
        background = self.create_background()
        
        # Person walks in, leaves object, walks away
        abandon_frame = total_frames // 3
        
        for frame_idx in range(total_frames):
            frame = background.copy()
            
            if frame_idx < abandon_frame:
                # Person walking with object
                person_x = 100 + (frame_idx * 5)
                person_y = 300
                
                # Draw person
                cv2.rectangle(frame, 
                             (person_x-20, person_y-40), 
                             (person_x+20, person_y+40), 
                             (0, 255, 0), -1)
                
                # Draw object (bag)
                cv2.rectangle(frame, 
                             (person_x+25, person_y-10), 
                             (person_x+40, person_y+10), 
                             (0, 0, 255), -1)
            
            elif frame_idx == abandon_frame:
                # Person leaves object
                object_x = 100 + (abandon_frame * 5) + 25
                object_y = 300
                
                # Only draw abandoned object
                cv2.rectangle(frame, 
                             (object_x, object_y-10), 
                             (object_x+15, object_y+10), 
                             (0, 0, 255), -1)
            
            else:
                # Object remains abandoned
                object_x = 100 + (abandon_frame * 5) + 25
                object_y = 300
                
                cv2.rectangle(frame, 
                             (object_x, object_y-10), 
                             (object_x+15, object_y+10), 
                             (0, 0, 255), -1)
            
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            frames.append(frame)
        
        return frames
    
    def generate_unusual_movement_scenario(self, duration, fps):
        """Generate synthetic unusual movement scenario"""
        frames = []
        total_frames = duration * fps
        
        background = self.create_background()
        
        for frame_idx in range(total_frames):
            frame = background.copy()
            
            # Erratic movement pattern
            if frame_idx < total_frames // 2:
                # Normal walking
                person_x = 50 + (frame_idx * 3)
                person_y = 300
            else:
                # Sudden fast/erratic movement
                person_x = 50 + (total_frames // 2 * 3) + ((frame_idx - total_frames // 2) * 15)
                person_y = 300 + np.sin(frame_idx * 0.5) * 50
            
            # Ensure person stays in frame
            person_x = max(20, min(person_x, 760))
            person_y = max(40, min(person_y, 520))
            
            # Draw person
            cv2.rectangle(frame, 
                         (int(person_x)-20, int(person_y)-40), 
                         (int(person_x)+20, int(person_y)+40), 
                         (0, 255, 0), -1)
            
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            frames.append(frame)
        
        return frames
    
    def create_background(self, width=800, height=600):
        """Create simple background for synthetic scenarios"""
        # Create gray background
        background = np.ones((height, width, 3), dtype=np.uint8) * 128
        
        # Add some simple features (lines for parking lot, etc.)
        cv2.line(background, (0, height//2), (width, height//2), (255, 255, 255), 2)
        cv2.line(background, (width//2, 0), (width//2, height), (255, 255, 255), 2)
        
        return background
    
    def save_video(self, frames, output_path, fps=30):
        """Save frames as video file"""
        if not frames:
            raise ValueError("No frames to save")
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        print(f"Synthetic video saved: {output_path}")

def generate_all_scenarios():
    """Generate all synthetic scenarios"""
    generator = SimpleVideoGAN()
    
    scenarios = [
        ("loitering", "synthetic_loitering.mp4"),
        ("abandonment", "synthetic_abandonment.mp4"),
        ("unusual_movement", "synthetic_unusual_movement.mp4")
    ]
    
    output_dir = Path("data/synthetic")
    output_dir.mkdir(exist_ok=True)
    
    for scenario_type, filename in scenarios:
        print(f"Generating {scenario_type} scenario...")
        frames = generator.generate_synthetic_scenario(scenario_type, duration=15)
        output_path = str(output_dir / filename)
        generator.save_video(frames, output_path)
    
    print("All synthetic scenarios generated!")

if __name__ == "__main__":
    generate_all_scenarios()