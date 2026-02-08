try:
    from deepface import DeepFace
    import cv2
    import numpy as np
    print("DeepFace and dependencies imported successfully.")
except Exception as e:
    print(f"Error importing DeepFace: {e}")
