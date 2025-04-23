"""Process single images for posture analysis"""
import cv2
import mediapipe as mp
from posture_analysis.pose_detector import PoseDetector
from posture_analysis.visualization import draw_landmarks, draw_status
from posture_analysis.posture_analyzer import analyze_posture

def analyze_image(image_path):
    """Analyze posture from a single image."""
    # Initialize detector
    pose_detector = PoseDetector()
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process image
    keypoints = pose_detector.find_pose_image(image_rgb)
    if not keypoints:
        return None, "No pose detected"
    
    # Analyze posture
    angles, issues = analyze_posture(keypoints)
    
    # Draw visualization
    annotated_image = image.copy()
    draw_landmarks(annotated_image, keypoints)
    draw_status(annotated_image, angles, issues)
    print(issues)
    return annotated_image, issues

def main():
    """Process a single image and display results."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze posture from an image')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--output', '-o', help='Path to save the output image')
    args = parser.parse_args()
    
    try:
        # Process image
        annotated_image, issues = analyze_image(args.image_path)
        if annotated_image is None:
            print(f"Error: {issues}")
            return
        
        # Print issues
        print("\nPosture Analysis Results:")
        for issue in issues:
            print(f"- {issue}")
        
        # Save or display result
        if args.output:
            cv2.imwrite(args.output, annotated_image)
            print(f"\nSaved annotated image to: {args.output}")
        else:
            cv2.imshow('Posture Analysis', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == '__main__':
    main()
