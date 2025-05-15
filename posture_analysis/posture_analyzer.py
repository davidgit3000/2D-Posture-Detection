import numpy as np

def calculate_angles(keypoints):
    """Calculate relevant angles for posture analysis."""
    angles = {}
    
    # Calculate basic angles between key points
    if all(k in keypoints for k in ['left_shoulder', 'right_shoulder']):
        left = np.array(keypoints['left_shoulder'][:2])
        right = np.array(keypoints['right_shoulder'][:2])
        angles['shoulder_width'] = np.linalg.norm(right - left)
    
    if all(k in keypoints for k in ['left_eye', 'right_eye']):
        left = np.array(keypoints['left_eye'][:2])
        right = np.array(keypoints['right_eye'][:2])
        angles['eye_width'] = np.linalg.norm(right - left)
    
    return angles

def detect_sitting_position(keypoints):
    if all(k in keypoints for k in ['left_knee', 'right_knee', 'left_hip', 'right_hip', 'left_ankle', 'right_ankle']):
        left_knee = np.array(keypoints['left_knee'][:2])
        right_knee = np.array(keypoints['right_knee'][:2])
        left_hip = np.array(keypoints['left_hip'][:2])
        right_hip = np.array(keypoints['right_hip'][:2])
        left_ankle = np.array(keypoints['left_ankle'][:2])
        right_ankle = np.array(keypoints['right_ankle'][:2])
        
        # Calculate average positions
        knee_y = (left_knee[1] + right_knee[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2
        ankle_y = (left_ankle[1] + right_ankle[1]) / 2
        
        # Calculate vertical distances
        knee_hip_diff = knee_y - hip_y
        knee_ankle_diff = knee_y - ankle_y
        hip_ankle_diff = hip_y - ankle_y
        
        # When standing:
        # - Knees should be below hips (knee_hip_diff > 50)
        # - Ankles should be below knees (knee_ankle_diff > 50)
        # - Total leg length (hip to ankle) should be significant (hip_ankle_diff > 100)
        
        is_standing = (knee_hip_diff > 50 and knee_ankle_diff < -50 and hip_ankle_diff < -100)
        print(f"Debug - knee_hip_diff: {knee_hip_diff}, knee_ankle_diff: {knee_ankle_diff}, hip_ankle_diff: {hip_ankle_diff}")
        return not is_standing
    return False

# def detect_facing_direction(keypoints):
#     left_shoulder = np.array(keypoints['left_shoulder'][:2])
#     right_shoulder = np.array(keypoints['right_shoulder'][:2])
#     return right_shoulder[0] < left_shoulder[0]  # Right side is closer

def analyze_head_position(keypoints, angles, position_label, is_sitting):
    issues = []
    
    left_eye = np.array(keypoints['left_eye'][:2])
    right_eye = np.array(keypoints['right_eye'][:2])
    left_ear = np.array(keypoints['left_ear'][:2])
    right_ear = np.array(keypoints['right_ear'][:2])
    eye_mid = (left_eye + right_eye) / 2
    ear_mid = (left_ear + right_ear) / 2
    eye_vector = right_eye - left_eye
    head_tilt = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))

    if head_tilt > 90:
        head_tilt = 180 - head_tilt
    elif head_tilt < -90:
        head_tilt = -180 - head_tilt
    angles['head_tilt'] = head_tilt

    if all(k in keypoints for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
        shoulder_mid = (np.array(keypoints['left_shoulder'][:2]) + np.array(keypoints['right_shoulder'][:2])) / 2
        hip_mid = (np.array(keypoints['left_hip'][:2]) + np.array(keypoints['right_hip'][:2])) / 2
        torso_vector = shoulder_mid - hip_mid
        torso_angle = np.degrees(np.arctan2(torso_vector[0], -torso_vector[1]))
        eye_ear_y_diff = eye_mid[1] - ear_mid[1]
        angles['head_forward'] = eye_ear_y_diff

        ear_shoulder_x_diff = ear_mid[0] - shoulder_mid[0]
        adjusted_x_diff = ear_shoulder_x_diff + (torso_angle * 2)
        angles['head_side'] = adjusted_x_diff
        shoulder_width = np.linalg.norm(np.array(keypoints['right_shoulder'][:2]) - np.array(keypoints['left_shoulder'][:2]))
        angles['shoulder_width'] = shoulder_width
        is_side_view = shoulder_width < 150

        if not is_side_view and abs(torso_angle) < 20:
            if abs(head_tilt) > 10:
                tilt_direction = 'right' if head_tilt > 0 else 'left'
                issues.append(f'{position_label} FRONT VIEW: Head tilted to the {tilt_direction} - Level your head')

            forward_threshold = 8 if is_sitting else -5
            backward_threshold = -15 if is_sitting else -25
            if eye_ear_y_diff > forward_threshold:
                issues.append(f'{position_label} FRONT VIEW: Head tilted forward - Raise your chin')
            elif eye_ear_y_diff < backward_threshold:
                issues.append(f'{position_label} FRONT VIEW: Head tilted back - Lower your chin')

        if is_side_view:
            hip_forward = hip_mid[0] - shoulder_mid[0]

            if abs(torso_angle) < 30:
                if adjusted_x_diff < -30:
                    issues.append(f'{position_label} SIDE VIEW: Head too far backward - Lower your chin')
                elif adjusted_x_diff > 40:
                    issues.append(f'{position_label} SIDE VIEW: Forward head posture - Head too far forward - Pull head back and down slightly')

            angles['hip_shoulder_offset'] = hip_forward

            if is_sitting:
                # Detect slouching based on shoulder-hip alignment
                if hip_forward < -10:  # More sensitive threshold
                    issues.append(f'{position_label} SIDE VIEW: Slouching in chair - Sit up straight and pull shoulders back')
                
                # Check torso angle
                if abs(torso_angle) > 15:  # More sensitive threshold
                    issues.append(f'{position_label} SIDE VIEW: Torso leaning - Align your back with the chair')
            else:
                if hip_forward > 30:
                    issues.append(f'{position_label} SIDE VIEW: Sway back posture - Tuck hips under and engage core')
                elif hip_forward < -30:  # More sensitive threshold
                    issues.append(f'{position_label} SIDE VIEW: Slouching forward - Pull shoulders back')
    return angles, issues

def analyze_additional_metrics(keypoints, angles, issues):
    if all(k in keypoints for k in ['left_ear', 'right_ear', 'left_shoulder', 'right_shoulder']):
        ear_mid = (np.array(keypoints['left_ear'][:2]) + np.array(keypoints['right_ear'][:2])) / 2
        shoulder_mid = (np.array(keypoints['left_shoulder'][:2]) + np.array(keypoints['right_shoulder'][:2])) / 2
        head_offset = ear_mid - shoulder_mid
        angles['forward_head'] = head_offset[0]
        angles['vertical_head'] = head_offset[1]
        angles['head_angle'] = np.degrees(np.arctan2(-head_offset[1], head_offset[0]))

    if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
        shoulder_vector = np.array(keypoints['right_shoulder'][:2]) - np.array(keypoints['left_shoulder'][:2])
        shoulder_width = np.linalg.norm(shoulder_vector)
        shoulder_angle = np.degrees(np.arctan2(shoulder_vector[1], shoulder_vector[0]))
        angles['shoulder_tilt'] = shoulder_angle
        if shoulder_width >= 150 and abs(shoulder_angle) > 8:
            direction = 'right' if shoulder_angle > 0 else 'left'
            issues.append(f'Shoulders tilted {direction} - Level your shoulders')

    if all(k in keypoints for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
        shoulder_mid = (np.array(keypoints['left_shoulder'][:2]) + np.array(keypoints['right_shoulder'][:2])) / 2
        hip_mid = (np.array(keypoints['left_hip'][:2]) + np.array(keypoints['right_hip'][:2])) / 2
        torso_vector = shoulder_mid - hip_mid
        angles['torso_tilt'] = np.degrees(np.arctan2(torso_vector[0], -torso_vector[1]))
        angles['shoulder_hip_angle'] = np.degrees(np.arctan2(shoulder_mid[1] - hip_mid[1], shoulder_mid[0] - hip_mid[0]))
    
    return angles, issues

def analyze_posture(keypoints):
    """Main posture analysis function."""
    angles = calculate_angles(keypoints)
    issues = []
    is_sitting = detect_sitting_position(keypoints)
    position_label = 'SITTING' if is_sitting else 'STANDING'
    # right_side_facing = detect_facing_direction(keypoints)

    if all(k in keypoints for k in ['left_eye', 'right_eye', 'left_ear', 'right_ear']):
        angles, head_issues = analyze_head_position(keypoints, angles, position_label, is_sitting)
        issues.extend(head_issues)
    
    angles, issues = analyze_additional_metrics(keypoints, angles, issues)
    
    return angles, issues
