"""Visualization functions for posture analysis"""
import cv2
import numpy as np

def draw_landmarks(frame, keypoints):
    """Draw landmarks and connections on the frame."""
    # Define face and body connections
    face_connections = [
        ('nose', 'left_eye'),
        ('nose', 'right_eye'),
        ('left_eye', 'left_ear'),
        ('right_eye', 'right_ear')
    ]
    
    body_connections = [
        ('left_shoulder', 'right_shoulder'),
        ('left_hip', 'right_hip'),
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'),
        ('left_hip', 'left_knee'),
        ('right_hip', 'right_knee'),
        ('left_knee', 'left_ankle'),
        ('right_knee', 'right_ankle')
    ]
    
    # Draw landmarks
    for name, point in keypoints.items():
        x, y = int(point[0]), int(point[1])
        # Draw larger circles for better visibility
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)  # White fill
        cv2.circle(frame, (x, y), 4, (0, 255, 0), 1)      # Green outline
        # Draw labels with better visibility
        # cv2.putText(frame, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black outline
        cv2.putText(frame, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # White text
    
    # Draw face connections in blue
    for start, end in face_connections:
        if start in keypoints and end in keypoints:
            start_point = tuple(map(int, keypoints[start][:2]))
            end_point = tuple(map(int, keypoints[end][:2]))
            cv2.line(frame, start_point, end_point, (255, 0, 0), 1)
    
    # Draw body connections in green with thickness
    for start, end in body_connections:
        if start in keypoints and end in keypoints:
            start_point = tuple(map(int, keypoints[start][:2]))
            end_point = tuple(map(int, keypoints[end][:2]))
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

def draw_status(frame, angles, issues):
    """Draw angle measurements and posture warnings on the frame."""
    # Create a wider frame with black padding on the left for text
    padding_width = 400  # Width of the black padding area
    h, w = frame.shape[:2]
    padded_frame = np.zeros((h, w + padding_width, 3), dtype=np.uint8)
    padded_frame[:, padding_width:] = frame  # Place original frame on the right
    
    # Display angles with descriptions
    y_pos = 30
    x_margin = 10
    line_height = 25
    max_width = padding_width - 20  # Text area width
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    
    # Helper function to wrap text
    def get_wrapped_text(text, font, font_scale, thickness, max_width):
        words = text.split(' ')
        lines = []
        current_line = []
        
        for word in words:
            # Try adding the word to current line
            test_line = current_line + [word]
            text_size = cv2.getTextSize(' '.join(test_line), font, font_scale, thickness)[0]
            
            if text_size[0] <= max_width:
                # Word fits, add it to current line
                current_line = test_line
            else:
                # Word doesn't fit
                if current_line:
                    # Save current line if it exists
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # If current line is empty, word is too long - split it
                    while word:
                        for i in range(len(word), 0, -1):
                            text_size = cv2.getTextSize(word[:i], font, font_scale, thickness)[0]
                            if text_size[0] <= max_width:
                                lines.append(word[:i])
                                word = word[i:]
                                break
                        if i == 1:  # Prevent infinite loop for very long words
                            lines.append(word[:1])
                            word = word[1:]
        
        # Add remaining line if it exists
        if current_line:
            lines.append(' '.join(current_line))
        return lines
    
    # Display angles
    for joint, value in angles.items():
        text = f"{joint}: {value:.1f}"
        cv2.putText(padded_frame, text, (x_margin, y_pos), font, font_scale, (0, 255, 0), thickness)
        y_pos += line_height
    
    # Display posture issues in red
    for issue in issues:
        wrapped_lines = get_wrapped_text(issue, font, font_scale, thickness, max_width)
        for line in wrapped_lines:
            cv2.putText(padded_frame, line, (x_margin, y_pos), font, font_scale, (0, 0, 255), thickness)
            y_pos += line_height
    
    return padded_frame
