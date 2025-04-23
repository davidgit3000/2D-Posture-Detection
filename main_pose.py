"""Main script for real-time posture analysis using MediaPipe Pose"""
import cv2
import numpy as np
import os
from posture_analysis.pose_detector import PoseDetector
from posture_analysis.posture_analyzer import analyze_posture

def process_webcam():
    """Process webcam feed for posture analysis."""
    # Initialize pose detector
    detector = PoseDetector()
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    print("Starting webcam posture analysis...")
    print("Press 'q' to quit")

    frame_count = 0
    save_dir = "captured_frames"
    os.makedirs(save_dir, exist_ok=True)

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame")
                break

            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect pose
            keypoints, found = detector.find_pose(frame)
            
            if found:
                # Analyze posture
                angles, issues = analyze_posture(keypoints)
                
                # Draw visualization
                frame = detector.draw_pose(frame, keypoints)
                
                # Draw status
                y_pos = 30
                # Show angles
                for name, value in angles.items():
                    text = f"{name}: {value:.1f}"
                    cv2.putText(frame, text, (10, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_pos += 20
                
                # Show issues in red
                for issue in issues:
                    cv2.putText(frame, issue, (10, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    y_pos += 20
                
                # Save frames periodically
                if frame_count % 60 == 0:  # Save every 60 frames
                    frame_path = os.path.join(save_dir, f"frame_{frame_count//60:04d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    print(f"Saved frame to {frame_path}")

            # Display the frame
            cv2.imshow('Posture Analysis', frame)
            frame_count += 1

            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    process_webcam()
