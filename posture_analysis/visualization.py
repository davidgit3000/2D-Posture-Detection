"""Visualization functions for posture analysis"""
import cv2
import numpy as np

def draw_landmarks(frame, keypoints):
    """Draw facial landmarks and connections on the frame."""
    # Draw points
    for name, point in keypoints.items():
        x, y = int(point[0]), int(point[1])
        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    # Draw connections
    connections = [
        ('left_eye', 'right_eye'),
        ('left_eye', 'nose'),
        ('right_eye', 'nose'),
        ('left_ear', 'left_eye'),
        ('right_ear', 'right_eye'),
        ('left_ear', 'left_shoulder'),
        ('right_ear', 'right_shoulder'),
        ('left_shoulder', 'right_shoulder')
    ]
    
    for start_point, end_point in connections:
        if start_point in keypoints and end_point in keypoints:
            pt1 = (int(keypoints[start_point][0]), int(keypoints[start_point][1]))
            pt2 = (int(keypoints[end_point][0]), int(keypoints[end_point][1]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 1)

def draw_status(frame, angles, issues):
    """Draw angle measurements and posture warnings on the frame."""
    # Display angles with descriptions
    y_pos = 30
    angle_descriptions = {
        'forward_tilt': 'Head tilt % (- down, + up)',
        'left_neck': 'Left ear-shoulder vertical angle',
        'right_neck': 'Right ear-shoulder vertical angle',
        'eye_level': 'Eye horizontal tilt',
        'visible_side': 'Dominant side',
        'side_tilt': 'Side tilt angle (- left, + right)'
    }
    
    for joint, angle in angles.items():
        description = angle_descriptions.get(joint, joint)
        if joint == 'visible_side':
            text = f"{description}: {angle}"
        else:
            text = f"{description}: {angle:.1f}°"
        cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_pos += 20
    
    # Display posture issues in red
    for issue in issues:
        cv2.putText(frame, issue, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        y_pos += 20
