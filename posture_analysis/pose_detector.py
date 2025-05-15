"""Pose detection and analysis using MediaPipe Pose"""
import cv2
import mediapipe as mp
import numpy as np

class PoseDetector:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,  # More accurate model
            smooth_landmarks=True,
            enable_segmentation=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            refine_face_landmarks=True  # Get more detailed face landmarks
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def find_pose(self, frame):
        """Detect pose and face landmarks in frame for video."""
        results = self.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks and results.face_landmarks:
            # Extract key points for posture analysis
            keypoints = {}
            pose_landmarks = results.pose_landmarks.landmark
            face_landmarks = results.face_landmarks.landmark
            
            # Map important landmarks (swapped left/right for mirrored view)
            landmark_map = {
                # Face landmarks from face detection (more accurate)
                'nose': face_landmarks[1],  # Nose tip
                'left_eye': face_landmarks[33],  # Right eye outer corner
                'right_eye': face_landmarks[263],  # Left eye outer corner
                'left_ear': face_landmarks[234],  # Right ear
                'right_ear': face_landmarks[454],  # Left ear
                
                # Upper body landmarks from pose detection
                'right_shoulder': pose_landmarks[11],  # Swapped with left
                'left_shoulder': pose_landmarks[12],   # Swapped with right
                'right_hip': pose_landmarks[23],  # Swapped with left
                'left_hip': pose_landmarks[24],   # Swapped with right
                
                # Lower body landmarks from pose detection
                'right_knee': pose_landmarks[25],  # Swapped with left
                'left_knee': pose_landmarks[26],   # Swapped with right
                'right_ankle': pose_landmarks[27],  # Swapped with left
                'left_ankle': pose_landmarks[28],   # Swapped with right
            }
            
            h, w, _ = frame.shape
            for name, point in landmark_map.items():
                x, y, z = int(point.x * w), int(point.y * h), point.z
                keypoints[name] = np.array([x, y, z])
            
            return keypoints, True
        
        return None, False
    
    def find_pose_image(self, image):
        """Detect pose and face landmarks in a single image."""
        # Create a temporary Holistic object with static_image_mode=True
        with self.mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.7,
            refine_face_landmarks=True
        ) as holistic:
            # Process the image
            results = holistic.process(image)
            
            if not results.pose_landmarks:
                return None
            
            # Get image dimensions
            h, w = image.shape[:2]
            
            # Convert normalized coordinates to pixel coordinates
            keypoints = {}
            landmark_map = {
                'nose': (results.pose_landmarks.landmark[0], True),  # True for face landmarks
                'left_eye': (results.pose_landmarks.landmark[2], True),
                'right_eye': (results.pose_landmarks.landmark[5], True),
                'left_ear': (results.pose_landmarks.landmark[7], True),
                'right_ear': (results.pose_landmarks.landmark[8], True),
                'left_shoulder': (results.pose_landmarks.landmark[11], False),
                'right_shoulder': (results.pose_landmarks.landmark[12], False),
                'left_hip': (results.pose_landmarks.landmark[23], False),
                'right_hip': (results.pose_landmarks.landmark[24], False),
                'left_knee': (results.pose_landmarks.landmark[25], False),
                'right_knee': (results.pose_landmarks.landmark[26], False),
                'left_ankle': (results.pose_landmarks.landmark[27], False),
                'right_ankle': (results.pose_landmarks.landmark[28], False)
            }
            
            for name, (landmark, is_face) in landmark_map.items():
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                z = landmark.z
                keypoints[name] = np.array([x, y, z])
            
            return keypoints