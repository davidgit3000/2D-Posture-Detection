"""Angle calculation functions"""
import numpy as np

# Global smoothing variables
angle_history = {}
SMOOTHING_WINDOW = 5

def smooth_angle(joint_name, angle):
    """Smooth angle values over time using moving average."""
    if joint_name not in angle_history:
        angle_history[joint_name] = []
    
    angle_history[joint_name].append(angle)
    if len(angle_history[joint_name]) > SMOOTHING_WINDOW:
        angle_history[joint_name].pop(0)
    
    return np.mean(angle_history[joint_name])

def calculate_relative_depth(point1, point2):
    """Calculate depth difference between two points."""
    if isinstance(point1, dict):
        z1 = point1['z']
    else:
        z1 = point1[2]
        
    if isinstance(point2, dict):
        z2 = point2['z']
    else:
        z2 = point2[2]
    return z1 - z2

def angle_with_vertical(point1, point2):
    """Calculate angle between a line and vertical axis."""
    # Get vector from point1 to point2
    vector = np.array([point2[1] - point1[1], point2[0] - point1[0]])  # Use x,y coordinates
    
    # Calculate angle with vertical (y-axis)
    if np.linalg.norm(vector) > 0:
        # Angle between vector and vertical (0, -1)
        cos_angle = -vector[0] / np.linalg.norm(vector)  # Negative because y-axis points down
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        return angle
    return 0

def calculate_angles(keypoints):
    """Calculate all relevant angles from keypoints."""
    angles = {}
    
    # Calculate forward/backward tilt using nose and ears
    if 'nose' in keypoints and 'left_ear' in keypoints and 'right_ear' in keypoints:
        nose = keypoints['nose']
        left_ear = keypoints['left_ear']
        right_ear = keypoints['right_ear']
        
        # Calculate eye positions
        if 'left_eye' in keypoints and 'right_eye' in keypoints:
            left_eye = keypoints['left_eye']
            right_eye = keypoints['right_eye']
            eye_mid_y = (left_eye[1] + right_eye[1]) / 2
            ear_mid_y = (left_ear[1] + right_ear[1]) / 2
            
            # Calculate tilt using eye position relative to ears
            # Negative: eyes below ears (looking down)
            # Positive: eyes above ears (looking up)
            tilt = eye_mid_y - ear_mid_y
            
            # Normalize by face size (distance between ears)
            face_width = abs(keypoints['right_ear'][0] - keypoints['left_ear'][0])
            if face_width > 0:
                tilt = (tilt / face_width) * 100  # Convert to percentage
                
            angles['forward_tilt'] = smooth_angle('forward_tilt', tilt)
    
    # Calculate neck angles using ear-shoulder relationship
    for side in ['left', 'right']:
        if f'{side}_ear' in keypoints and f'{side}_shoulder' in keypoints:
            # Calculate neck angle using y,z coordinates (vertical plane)
            raw_angle = angle_with_vertical(
                keypoints[f'{side}_shoulder'],
                keypoints[f'{side}_ear']
            )
            if raw_angle is not None:
                # Normalize angle to be between 0 and 90 degrees
                raw_angle = abs(raw_angle)
                if raw_angle > 90:
                    raw_angle = 180 - raw_angle
                
                angles[f'{side}_neck'] = smooth_angle(f'{side}_neck', raw_angle)
    
    # Calculate eye level angle
    if 'left_eye' in keypoints and 'right_eye' in keypoints:
        left_eye = np.array(keypoints['left_eye'][:2])
        right_eye = np.array(keypoints['right_eye'][:2])
        eye_vector = right_eye - left_eye
        if np.linalg.norm(eye_vector) > 0:
            angle = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))
            angles['eye_level'] = smooth_angle('eye_level', angle)
    
    # Determine which side is more visible
    if 'nose' in keypoints and 'left_ear' in keypoints and 'right_ear' in keypoints:
        left_depth = calculate_relative_depth(keypoints['nose'], keypoints['left_ear'])
        right_depth = calculate_relative_depth(keypoints['nose'], keypoints['right_ear'])
        angles['visible_side'] = 'left' if left_depth < right_depth else 'right'
    
    return angles
