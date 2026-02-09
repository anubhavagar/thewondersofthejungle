
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

try:
    print("1. Importing GymnasticsAnalyzer from refactored module...")
    from model_service.gymnastics import gymnastics_analyzer
    print("   Success! Module imported.")
except Exception as e:
    print(f"   FAILED to import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Path to a known artifact image
image_path = r"c:\Users\ur_an\projects\let_me_check_your_face_app\api\uploads\sample_image.jpg"

try:
    print(f"2. Loading image from {image_path}...")
    with open(image_path, "rb") as f:
        image_data = f.read()
    # Create data URL format expected by analyzer
    import base64
    b64_data = base64.b64encode(image_data).decode('utf-8')
    data_url = f"data:image/png;base64,{b64_data}"
    print("   Success!")
    
    print("3. Running analyze_media...")
    result = gymnastics_analyzer.analyze_media(data_url, "image")
    print("4. Result obtained:")
    import json
    print(json.dumps(result, indent=2))

except Exception as e:
    print(f"   CRASHED during analysis: {e}")
    import traceback
    traceback.print_exc()
