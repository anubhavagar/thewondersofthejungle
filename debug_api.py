
import urllib.request
import urllib.error
import json
import base64
import os

url = "http://localhost:8000/api/analyze/gymnastics"

# Image path
image_path = r"c:\Users\ur_an\projects\let_me_check_your_face_app\api\uploads\sample_image.jpg"
if not os.path.exists(image_path):
    image_path = r"c:\Users\ur_an\projects\let_me_check_your_face_app\gymnast_test.jpg"

print(f"Testing API with image: {image_path}")

try:
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    b64_data = base64.b64encode(image_data).decode('utf-8')
    # API expects full data URL or just base64? 
    # endpoints.py: gymnastics_analyzer.analyze_media(media_data)
    # gymnastics.py: if "base64," in media_data: split...
    # So data URL is safer.
    data_url = f"data:image/jpeg;base64,{b64_data}"
    
    payload = {
        "media_data": data_url,
        "media_type": "image"
    }
    
    data = json.dumps(payload).encode('utf-8')
    
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req) as response:
            print(f"Status Code: {response.status}")
            response_body = response.read().decode('utf-8')
            print("Response Body:")
            print(json.dumps(json.loads(response_body), indent=2))
            
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code}")
        print(e.read().decode('utf-8'))
        
except Exception as e:
    print(f"Error: {e}")
