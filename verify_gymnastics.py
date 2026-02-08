import cv2
import mediapipe as mp
import numpy as np
import os
import sys

def test_gymnastics_model():
    print("Testing Gymnastics Model Dependencies...")
    
    # 1. Check Imports
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return

    try:
        import mediapipe as mp
        print("✅ MediaPipe imported successfully")
    except ImportError as e:
        print(f"❌ MediaPipe import failed: {e}")
        return

    # 2. Check Model Initialization
    try:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1, # Try 1 first
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        print("✅ MediaPipe Pose model initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Pose model: {e}")
        return

    # 3. Create Dummy Image (Black background)
    # MediaPipe might not detect anything, but it shouldn't crash
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Test", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 4. Process Image
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        print("✅ Pose processing ran successfully")
        
        if results.pose_landmarks:
            print("ℹ️ Landmarks detected (Unexpected on blank image, but okay)")
        else:
            print("ℹ️ No landmarks detected (Expected on blank image)")
            
    except Exception as e:
        print(f"❌ Processing failed: {e}")

    print("\nSystem verification complete.")

if __name__ == "__main__":
    test_gymnastics_model()
