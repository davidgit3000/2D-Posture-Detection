# 2D Pose Estimation for Posture Analysis

A real-time posture analysis system using MediaPipe for pose estimation and OpenCV for visualization. The system can analyze posture from both webcam input and static images.

## Features

- Real-time pose detection and analysis
- Static image posture analysis
- Detection of common posture issues:
  - Forward head posture
  - Slouching
  - Head tilt
  - Shoulder alignment
  - Torso alignment
- Support for both sitting and standing posture analysis
- Visual feedback with landmarks and measurements

## Setup

1. Install dependencies:
```bash
pip install mediapipe opencv-python numpy
```

2. Clone the repository:
```bash
git clone <repository-url>
cd 2d_pose_estimation
```

## Usage

### Real-time Analysis
Run the main script to analyze posture in real-time using your webcam:
```bash
python main_pose.py
```

### Static Image Analysis
Analyze posture from a static image:
```bash
python process_image.py path/to/image.jpg
```

Optional: Save the annotated output:
```bash
python process_image.py path/to/image.jpg --output result.jpg
```

## Project Structure

### Core Files
- `main_pose.py`: Main script for real-time webcam analysis
- `process_image.py`: Script for analyzing static images
- `requirements.txt`: List of Python dependencies

### Posture Analysis Package
Located in `posture_analysis/`:
- `__init__.py`: Package initialization
- `pose_detector.py`: MediaPipe pose detection wrapper
  - Handles pose landmark detection
  - Converts normalized coordinates to pixel space
  - Supports both video and image input
- `posture_analyzer.py`: Core posture analysis logic
  - Calculates angles and measurements
  - Detects posture issues
  - Provides feedback and recommendations
- `visualization.py`: Visualization utilities
  - Draws landmarks and connections
  - Displays measurements and warnings

### Input/Output Directories
- `images/`: Sample images for testing
- `captured_frames/`: Saved frames from webcam analysis
- `res/`: Directory for saving analysis results

## Key Components

### PoseDetector Class
- Wraps MediaPipe Holistic model
- Handles pose and face landmark detection
- Provides normalized and pixel-space coordinates
- Supports both real-time and static image processing

### Posture Analysis
The system analyzes several aspects of posture:
- Head position (forward tilt, side tilt)
- Shoulder alignment
- Hip-shoulder relationship
- Overall body alignment
- Sitting vs standing detection

### Visualization
- Color-coded landmarks (blue for face, green for body)
- Connection lines between related landmarks
- On-screen measurements and angles
- Warning messages for detected issues
