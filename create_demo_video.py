#!/usr/bin/env python3
"""
Create a simple demo video for testing the anomaly detection system
"""

import cv2
import numpy as np
import os

def create_demo_video(output_path="demo_video.mp4", duration=30, fps=30):
    """Create a demo video with moving objects"""
    
    # Video properties
    width, height = 640, 480
    total_frames = duration * fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating demo video: {output_path}")
    print(f"Duration: {duration}s, FPS: {fps}, Total frames: {total_frames}")
    
    for frame_num in range(total_frames):
        # Create background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50  # Dark gray background
        
        # Add some texture
        noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # Add moving objects (normal behavior)
        time_factor = frame_num / fps
        
        # Moving circle (person)
        circle_x = int(50 + (time_factor * 20) % (width - 100))
        circle_y = int(height // 2 + 50 * np.sin(time_factor * 0.5))
        cv2.circle(frame, (circle_x, circle_y), 20, (0, 255, 0), -1)
        
        # Moving rectangle (vehicle)
        rect_x = int(width - 100 - (time_factor * 30) % (width - 100))
        rect_y = int(height // 3)
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 60, rect_y + 30), (255, 0, 0), -1)
        
        # Add some "anomalous" behavior after 15 seconds
        if time_factor > 15:
            # Stationary object (abandoned object)
            cv2.rectangle(frame, (300, 350), (350, 400), (0, 0, 255), -1)
            
            # Erratic movement (suspicious behavior)
            erratic_x = int(400 + 100 * np.sin(time_factor * 5))
            erratic_y = int(200 + 50 * np.cos(time_factor * 7))
            cv2.circle(frame, (erratic_x, erratic_y), 15, (255, 255, 0), -1)
        
        # Add timestamp
        cv2.putText(frame, f"Time: {time_factor:.1f}s", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add frame info
        cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
        
        # Progress indicator
        if frame_num % (total_frames // 10) == 0:
            progress = (frame_num / total_frames) * 100
            print(f"Progress: {progress:.1f}%")
    
    out.release()
    print(f"✅ Demo video created: {output_path}")
    return output_path

if __name__ == "__main__":
    create_demo_video()