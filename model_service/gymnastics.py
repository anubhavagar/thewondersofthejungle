
import cv2
import mediapipe as mp
import numpy as np
import math
import tempfile
import os
import base64
import random
from collections import Counter
import traceback
from types import SimpleNamespace

# Import MediaPipe Tasks API
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, ObjectDetector, ObjectDetectorOptions, RunningMode
from mediapipe.tasks.python import BaseOptions

class GymnasticsAnalyzer:
    def __init__(self):
        try:
            # Model Path (Lite)
            model_path = os.path.join(os.path.dirname(__file__), "models", "pose_landmarker_lite.task")
            if not os.path.exists(model_path):
                print(f"WARNING: Model not found at {model_path}. Analysis will fail.")
            
            # Create Landmarker for Image Mode
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.15,
                min_pose_presence_confidence=0.15,
                min_tracking_confidence=0.15,
                output_segmentation_masks=False
            )
            self.landmarker = PoseLandmarker.create_from_options(options)
            print("GymnasticsAnalyzer: PoseLandmarker initialized successfully.")

            # Create Object Detector (Replacing YOLO)
            detector_path = os.path.join(os.path.dirname(__file__), "models", "efficientdet_lite0.tflite")
            detector_options = ObjectDetectorOptions(
                base_options=BaseOptions(model_asset_path=detector_path),
                running_mode=RunningMode.IMAGE,
                score_threshold=0.2, # Lower threshold for skeletal apparatus
                max_results=5
            )
            self.detector = ObjectDetector.create_from_options(detector_options)
            print("GymnasticsAnalyzer: MediaPipe ObjectDetector initialized successfully.")
            
        except Exception as e:
            print(f"GymnasticsAnalyzer Initialization Failed: {e}")
            traceback.print_exc()
            self.landmarker = None
            self.detector = None

        self.apparatus_specs = {
            "Pommel Horse (PH)": {"length_cm": 160, "name": "Pommel Horse (PH)"},
            "Balance Beam (BB)": {"length_cm": 500, "name": "Balance Beam (BB)"},
            "Vault (VT-M)": {"length_cm": 120, "name": "Vault (VT-M)"},
            "Vault (VT-W)": {"length_cm": 120, "name": "Vault (VT-W)"},
            "Parallel Bars (PB)": {"length_cm": 350, "name": "Parallel Bars (PB)"},
            "Uneven Bars (UB)": {"length_cm": 240, "name": "Uneven Bars (UB)"},
            "Still Rings (SR)": {"length_cm": 100, "name": "Still Rings (SR)"},
            "Floor Exercise (FX-M)": {"length_cm": 1200, "name": "Floor Exercise (FX-M)"},
            "Floor Exercise (FX-W)": {"length_cm": 1200, "name": "Floor Exercise (FX-W)"},
            "Horizontal Bar (HB)": {"length_cm": 240, "name": "Horizontal Bar (HB)"}
        }

    def _calculate_angle(self, a, b, c):
        """Calculate angle between three points (tuples of x,y)."""
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle

    def detect_apparatus(self, frame, pose_landmarks=None):
        """Detect equipment using MediaPipe Object Detector. Rejects overlap with the person."""
        if self.detector is None:
            return None
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.detections:
            return None
        
        h, w, _ = frame.shape
        
        # Determine rejection zone (torso) if pose is available
        torso_bbox = None
        if pose_landmarks:
            points = [pose_landmarks[i] for i in [11, 12, 23, 24]]
            torso_bbox = [
                min(p['x'] for p in points) * w,
                min(p['y'] for p in points) * h,
                max(p['x'] for p in points) * w,
                max(p['y'] for p in points) * h
            ]

        best_apparatus = None
        max_score = 0
        
        for detection in detection_result.detections:
            category = detection.categories[0]
            label = category.category_name.lower() # EfficientDet returns names like 'bench', 'chair'
            score = category.score
            
            if label == "person": continue 
            
            # Get Bounding Box
            box = detection.bounding_box
            left = box.origin_x
            top = box.origin_y
            width = box.width
            height = box.height
            
            center_x = left + width / 2
            center_y = top + height / 2

            # Anti-Person Overlap Check
            if torso_bbox:
                if (torso_bbox[0] < center_x < torso_bbox[2] and 
                    torso_bbox[1] < center_y < torso_bbox[3]):
                    
                    # We WANT to keep apparatus even if they overlap the gymnast (e.g. Pommel Horse)
                    # Expanded list to include "horse" (Pommel Horse), "bench", "table", etc.
                    allowed_labels = [
                        "bench", "couch", "bed", "chair", "suitcase", "dining table", "refrigerator", 
                        "surfboard", "skis", "baseball bat", 
                        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", # Pommel Horse often looks like an animal to COCO
                        "toilet", "fire hydrant" # Sometimes supports look like this
                    ]
                    if label not in allowed_labels:
                         continue

            if score > max_score:
                max_score = score
                
                # Logic to refine generic COCO labels to Gymnastics Apparatus
                refined_label = category.category_name
                aspect_ratio = width / height if height > 0 else 1

                if label in ["bench", "dining table", "suitcase", "refrigerator"]:
                    if aspect_ratio > 8.0: refined_label = "Parallel Bars (PB)"
                    elif aspect_ratio > 1.1: refined_label = "Pommel Horse (PH)"
                    else: refined_label = "Vault (VT-M)"
                elif label in ["couch", "bed"]:
                    if aspect_ratio > 6.5: refined_label = "Balance Beam (BB)"
                    else: refined_label = "Pommel Horse (PH)"
                elif label in ["surfboard", "skis", "baseball bat"]:
                    if aspect_ratio > 4.0: refined_label = "Parallel Bars (PB)"
                    else: refined_label = "Horizontal Bar (HB)"
                elif label in ["horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]:
                    refined_label = "Pommel Horse (PH)"
                elif label in ["toilet", "fire hydrant"]:
                    refined_label = "Mushroom / Pommel Trainer"
                elif label == "chair":
                    refined_label = "Vault (VT-M)"
                elif label == "sports ball": 
                    refined_label = "Still Rings (SR)"

                best_apparatus = {
                    "label": refined_label,
                    "shorthand": refined_label.split('(')[1].split(')')[0] if '(' in refined_label else refined_label[:3].upper(),
                    "score": float(score),
                    "bbox": [float(left/w), float(top/h), float(width/w), float(height/h)], # Normalized
                    "mask_polygon": None, # MediaPipe Object Detector doesn't provide masks
                    "spec": self.apparatus_specs.get(refined_label, {"length_cm": 200, "name": refined_label})
                }
        
        # Calculate Calibration Factor (px to cm)
        if best_apparatus:
            bbox = best_apparatus["bbox"]
            px_width = bbox[2] # bbox is [left, top, width, height]
            real_cm = best_apparatus["spec"]["length_cm"]
            best_apparatus["px_per_cm"] = (px_width * w) / real_cm if real_cm > 0 else 1.0

        return best_apparatus

    def calibrate_and_normalize(self, landmarks, apparatus, frame_width=1, frame_height=1):
        """Normalize landmarks relative to apparatus anchor points."""
        if not apparatus or not landmarks:
            return landmarks
        
        # Use center of apparatus bounding box as origin
        # bbox from detect_apparatus is [x_norm, y_norm, w_norm, h_norm]
        bbox = apparatus["bbox"] 
        
        # Convert app center to 0-1 normalized range
        # bbox[0] = x (left), bbox[1] = y (top), bbox[2] = width, bbox[3] = height
        app_cx = bbox[0] + (bbox[2] / 2)
        app_cy = bbox[1] + (bbox[3] / 2)
        
        # We use 0.7 as the vertical anchor to allow more "headroom" for the athlete
        anchor_y = 0.7
        
        # Shift mask polygon as well to stay synced in stabilized view
        if apparatus.get("mask_polygon"):
            new_masks = []
            for poly in apparatus["mask_polygon"]:
                new_masks.append([[p[0] - app_cx + 0.5, p[1] - app_cy + anchor_y] for p in poly])
            apparatus["mask_polygon"] = new_masks

        normalized = []
        for lm in landmarks:
            # Check if it's a dict or object
            is_dict = isinstance(lm, dict)
            lx = lm["x"] if is_dict else lm.x
            ly = lm["y"] if is_dict else lm.y
            lz = lm["z"] if is_dict else lm.z
            lv = lm["visibility"] if is_dict else lm.visibility

            # Shift relative to apparatus center
            rel_x = lx - app_cx + 0.5
            rel_y = ly - app_cy + anchor_y
            
            if is_dict:
                normalized.append({"x": rel_x, "y": rel_y, "z": lz, "visibility": lv})
            else:
                normalized.append(SimpleNamespace(x=rel_x, y=rel_y, z=lz, visibility=lv))
        return normalized

    def identify_skill(self, landmarks):
        """Identify specific skills based on landmark positions."""
        # Landmarks is a list of NormalizedLandmark objects (x, y, z, visibility)
        try:
            left_foot_y = landmarks[27].y
            right_foot_y = landmarks[28].y
            head_y = landmarks[0].y
            left_hand_y = landmarks[15].y
            right_hand_y = landmarks[16].y
            left_shoulder_y = landmarks[11].y
            right_shoulder_y = landmarks[12].y
            
            # 1. Handstand Check
            if left_foot_y < head_y and right_foot_y < head_y and left_hand_y > head_y:
                return "Handstand"
                
            # 2. IRON CROSS Check (Rings)
            left_elbow_y = landmarks[13].y
            right_elbow_y = landmarks[14].y
            
            arms_level = (abs(left_shoulder_y - left_elbow_y) < 0.07 and 
                          abs(right_shoulder_y - right_elbow_y) < 0.07 and
                          abs(left_elbow_y - left_hand_y) < 0.07 and
                          abs(right_elbow_y - right_hand_y) < 0.07)
                          
            feet_close = abs(landmarks[27].x - landmarks[28].x) < 0.08
            upright = left_foot_y > head_y
            
            if arms_level and upright and feet_close:
                 return "Iron Cross"

            # 3. L-Sit / Support Check (Arms down, legs extended forward)
            hands_below_shoulders = (left_hand_y > left_shoulder_y and right_hand_y > right_shoulder_y)
            hips_high = (landmarks[23].y < landmarks[27].y) # Hips above ankles
            legs_forward = (landmarks[27].z < landmarks[23].z - 0.2) # Feet significantly closer than hips (pike)
            
            if hands_below_shoulders and legs_forward:
                return "L-Sit on Rings"

            # 4. Pommel Horse Support / Circle Check
            # Hands are below hips, one or both hands supporting weight
            hands_at_hip_level = (abs(left_hand_y - landmarks[23].y) < 0.2 and abs(right_hand_y - landmarks[23].y) < 0.2)
            legs_spread_wide = (abs(landmarks[27].x - landmarks[28].x) > 0.4)
            
            if hands_at_hip_level:
                if legs_spread_wide:
                    return "Pommel Flair/Circle"
                # If hands are low and legs are in support, but not rings
                if not hands_below_shoulders:
                    return "Pommel Support"

            # 3. Y-Balance / Scale Check
            foot_diff = abs(left_foot_y - right_foot_y)
            if foot_diff > 0.4:
                return "Y-Scale / Arabesque"
                
            # 4. Bridge Check
            hip_y = landmarks[23].y
            if hip_y < head_y and hip_y < left_foot_y: 
                return "Bridge"
                
        except IndexError:
            pass
            
        return "Pose"

    def analyze_media(self, media_data: str, media_type: str = 'image', category: str = 'Senior'):
        import traceback
        if not self.landmarker:
            return {"error": "Analyzer not initialized properly (Model missing?)"}

        try:
            # Decode Base64
            if "base64," in media_data:
                media_data = media_data.split("base64,")[1]
            
            file_bytes = base64.b64decode(media_data)
            
            suffix = '.mp4' if media_type == 'video' else '.jpg'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            try:
                # Load Image
                # Note: For video, we should iterate frames. 
                # For this implementation, let's grab the middle frame for video analysis
                # to keep it fast and compatible with the single-result return type.
                
                target_image = None
                
                if media_type == 'video':
                    cap = cv2.VideoCapture(tmp_path)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    
                    frames_data = []
                    processed_frames = 0
                    
                    # Process frames (Skip frames for performance if needed, e.g., every 3rd frame)
                    stride = 2 if frame_count > 100 else 1 
                    
                    curr_frame_idx = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        
                        if curr_frame_idx % stride == 0:
                            # Convert to RGB
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                            
                            # Detect
                            detection = self.landmarker.detect(mp_image)
                            
                            if detection.pose_landmarks:
                                normalized_landmarks = [
                                    {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility} 
                                    for lm in detection.pose_landmarks[0]
                                ]
                                timestamp = curr_frame_idx / fps
                                # Dynamic Apparatus Detection (Pass landmarks for anti-person check)
                                apparatus_info = self.detect_apparatus(frame, pose_landmarks=normalized_landmarks)
                                h, w, _ = frame.shape
                                normalized_landmarks = self.calibrate_and_normalize(normalized_landmarks, apparatus_info, w, h)
                                
                                frames_data.append({
                                    "time": timestamp,
                                    "landmarks": normalized_landmarks,
                                    "apparatus": apparatus_info
                                })
                            
                        curr_frame_idx += 1
                        
                    cap.release()
                    
                    if not frames_data:
                         return {"error": "Could not track person in video"}

                    # Use the frame with the highest extension or middle frame for scoring
                    # For now, stick to middle of processed frames for scoring stability
                    target_frame_data = frames_data[len(frames_data)//2]
                    
                    # Convert dicts to objects for the Analyzer
                    landmarks = [SimpleNamespace(**lm) for lm in target_frame_data['landmarks']]

                else:
                    # Image Mode
                    target_image = cv2.imread(tmp_path)
                    if target_image is None: return {"error": "Failed to load image"}
                    
                    image_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                    detection_result = self.landmarker.detect(mp_image)
                    
                    if not detection_result.pose_landmarks:
                        return {
                            "total_score": 0, 
                            "skill": "No Person", 
                            "difficulty": 0.0,
                            "execution": 0.0,
                            "artistry": 0.0,
                            "comment": "No gymnast found."
                        }
                        
                    raw_landmarks = detection_result.pose_landmarks[0]
                    target_image = cv2.imread(tmp_path)
                    apparatus_info = self.detect_apparatus(target_image)
                    h, w, _ = target_image.shape
                    
                    # Normalize landmarks relative to apparatus
                    landmarks = self.calibrate_and_normalize(raw_landmarks, apparatus_info, w, h)
                    
                    frames_data = [{
                        "time": 0.0,
                        "landmarks": [{"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility} for lm in landmarks],
                        "apparatus": apparatus_info
                    }]

                # --- Artistry Heuristic ---
                def get_artistry(frames):
                    if not frames: return 0.0
                    # Stability: Variance in nose position (landmarks[0])
                    noses = [f['landmarks'][0] for f in frames if f['landmarks'][0].get('visibility', 0) > 0.5]
                    if len(noses) < 2: return round(random.uniform(3.0, 4.5), 1)
                    var_x = np.var([n['x'] for n in noses])
                    var_y = np.var([n['y'] for n in noses])
                    stability = 1.0 - (var_x + var_y) * 50 # Scale variance to a 0-1 penalty
                    base_artistry = 3.5 + (stability * 1.5) # Base 3.5 + up to 1.5 for stability
                    return round(max(0.0, min(5.0, base_artistry)), 1)

                    
                # --- ARTISTRY & EXECUTION ---
                artistry_score = get_artistry(frames_data)
                
                # --- SKILL ANALYSIS & KNOWLEDGE INJECTION ---
                from model_service.skill_knowledge import SKILL_DATA
                
                skill_list = []
                
                if media_type == 'image':
                    # Single Frame Analysis
                    skill = self.identify_skill(landmarks)
                    wag_feedback = self.analyze_wag_skill(landmarks, skill, category)
                    
                    final_skill_name = wag_feedback.get('skill', skill)
                    
                    skill_list = [{
                        "name": final_skill_name,
                        "dv": wag_feedback.get('d_score_contribution', 0.0),
                        "type": wag_feedback.get('type', 'dance')
                    }]
                    
                else:
                    # Video Analysis (aggregator logic)
                    all_skills_found = []
                    for frame_data in frames_data:
                        f_landmarks = [SimpleNamespace(**lm) for lm in frame_data['landmarks']]
                        f_skill = self.identify_skill(f_landmarks)
                        f_wag = self.analyze_wag_skill(f_landmarks, f_skill)
                        if f_wag.get('d_score_contribution', 0) > 0:
                            all_skills_found.append({
                                "name": f_wag.get('skill', f_skill),
                                "dv": f_wag.get('d_score_contribution', 0.0),
                                "type": f_wag.get('type', 'dance')
                            })
                    
                    # Deduplicate skills (take the highest value if seen multiple times)
                    unique_skills = {}
                    for s in all_skills_found:
                        if s['name'] not in unique_skills or s['dv'] > unique_skills[s['name']]['dv']:
                            unique_skills[s['name']] = s
                    
                    skill_list = list(unique_skills.values())

                # --- COMMON RESULT CONSTRUCTION ---
                
                # Calculate D-Score using the full list
                from model_service.wag_d_score import WAGDScoreCalculator
                d_calculator = WAGDScoreCalculator()
                d_score_result = d_calculator.calculate_d_score(skill_list)
                total_d_score = float(d_score_result.get('total_d_score', 0.0))

                # Identify Best Skill for Details
                best_skill_name = "Pose"
                if skill_list:
                    best_skill_name = max(skill_list, key=lambda x: x['dv'])['name']
                
                skill_details = SKILL_DATA.get(best_skill_name, {})

                # Find Visualization Frame
                viz_frame_idx = len(frames_data) // 2
                if skill_list:
                     for idx, fd in enumerate(frames_data):
                         flm = [SimpleNamespace(**lm) for lm in fd['landmarks']]
                         if self.identify_skill(flm) == best_skill_name:
                             viz_frame_idx = idx
                             break
                
                target_frame_data = frames_data[viz_frame_idx]
                target_landmarks = [SimpleNamespace(**lm) for lm in target_frame_data['landmarks']]
                
                # Re-analyze best frame for specific feedback
                wag_feedback = self.analyze_wag_skill(target_landmarks, best_skill_name, category)

                result = {
                    "total_score": round(10.0 + total_d_score + artistry_score - 3.0, 2), # Simplified Total
                    "skill": best_skill_name,
                    "difficulty": total_d_score,
                    "d_score_contribution": total_d_score,
                    "execution": 7.0,
                    "artistry": artistry_score,
                    "metrics": wag_feedback.get('metrics', {}),
                    "status": wag_feedback.get('status', 'Neutral'),
                    "deduction": wag_feedback.get('deduction', None),
                    "d_score_breakdown": d_score_result,
                    "comment": f"Routine complete. Recognized {len(skill_list)} skill(s).",
                    "landmarks": target_frame_data['landmarks'],
                    "frames": frames_data,
                    "apparatus": target_frame_data.get('apparatus'),
                    # Coach's Corner Data
                    "technical_cue": skill_details.get("technicalCue", "Focus on form and stability."),
                    "focus_anatomy": skill_details.get("focusAnatomy", []),
                    "common_errors": skill_details.get("commonDeductions", []),
                    "skill_description": skill_details.get("description", "")
                }
                
                return result

            finally:
                pass 

        except Exception as e:
            print(f"Error in analyze_media: {e}")
            traceback.print_exc()
            return {"error": str(e)}
        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                try: os.unlink(tmp_path)
                except: pass

class WAGAnalyzer(GymnasticsAnalyzer):
    """Specialized Analyzer for Women's Artistic Gymnastics (2D)."""
    
    def check_split_angle(self, landmarks):
        """W-001: Split (Cross or Side) >= 180 degrees."""
        # Hip (23/24), Knee (25/26), Ankle (27/28)
        # Using vector math on 2D projection
        l_hip = np.array([landmarks[23].x, landmarks[23].y])
        l_knee = np.array([landmarks[25].x, landmarks[25].y])
        r_hip = np.array([landmarks[24].x, landmarks[24].y])
        r_knee = np.array([landmarks[26].x, landmarks[26].y])
        
        v_l = l_knee - l_hip
        v_r = r_knee - r_hip
        
        dot_product = np.dot(v_l, v_r)
        norm_product = np.linalg.norm(v_l) * np.linalg.norm(v_r)
        
        if norm_product == 0: return 0.0
        
        rad = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
        angle = np.degrees(rad)
        
        # If vectors are opposite (split), angle is close to 180
        # If they are parallel (standing), angle is 0
        return round(angle, 1)

        return round(angle, 1)

    def check_ring_arch(self, landmarks):
        """W-002: Upper Body Arch >= 80 degrees."""
        # Shoulder (12), Hip (24), Knee (26)
        return self._calculate_angle(
            [landmarks[12].x, landmarks[12].y],
            [landmarks[24].x, landmarks[24].y],
            [landmarks[26].x, landmarks[26].y]
        )

    def check_head_release(self, landmarks):
        """W-003: Head Release >= 100 degrees."""
        # Ear (8), Neck/Shoulder (12), Hip (24)
        return self._calculate_angle(
            [landmarks[8].x, landmarks[8].y],
            [landmarks[12].x, landmarks[12].y],
            [landmarks[24].x, landmarks[24].y]
        )

    def analyze_wag_skill(self, landmarks, skill_name, category="Senior"):
        """Evaluate specific WAG criteria for a given skill with strict biomechanical thresholds."""
        metrics = {}
        feedback = {
            "skill": skill_name,
            "metrics": metrics,
            "status": "Neutral",
            "d_score_contribution": 0.0,
            "type": "dance",
            "deduction": None,
            "lock_status": "NONE"
        }
        
        # Adjust thresholds based on category (Simulation)
        # Groups: U8, U10, U12, U14, Senior
        is_elite = "Senior" in category or "U14" in category
        is_developmental = "U8" in category or "U10" in category
        
        min_split = 180.0 if is_elite else 165.0 # Forgiven splits for U8/U10
        min_head_release = 100.0 if not is_developmental else 80.0
        min_ring_arch = 80.0 if not is_developmental else 60.0

        # 1. Split Check (W-001)
        split_angle = self.check_split_angle(landmarks)
        metrics["W-001 (Split)"] = f"{split_angle}°"
        
        is_split = split_angle > 140
        split_pass = split_angle >= min_split

        # 2. Ring Arch (W-002)
        ring_arch = self.check_ring_arch(landmarks)
        metrics["W-002 (Arch)"] = f"{ring_arch}°"
        arch_pass = ring_arch >= min_ring_arch

        # 3. Head Release (W-003)
        head_release = self.check_head_release(landmarks)
        metrics["W-003 (Head)"] = f"{head_release}°"
        head_pass = head_release >= min_head_release

        # 4. Ring Foot Height (W-004)
        back_foot_y = landmarks[28].y
        head_y = landmarks[0].y
        foot_height_pass = back_foot_y <= head_y
        metrics["W-004 (Foot Height)"] = "Pass" if foot_height_pass else "Low"

        # Logic for "Ring" Classification
        if (is_split and head_release > 60):
            feedback["skill"] = "Ring Leap/Jump"
            
            # Lock Status Logic
            conditions = [split_pass, arch_pass, head_pass, foot_height_pass]
            pass_count = sum(conditions)
            
            if pass_count == 4:
                feedback["status"] = "Pass"
                feedback["lock_status"] = "LOCKED"
                feedback["lock_status"] = "LOCKED"
                # Difficulty bonus for higher categories
                base_dv = 0.6 if is_elite else 0.4
                feedback["d_score_contribution"] = base_dv
            elif pass_count == 3:
                feedback["status"] = "Deduction"
                feedback["lock_status"] = "PARTIAL"
                feedback["deduction"] = "Minor Technical Error (0.1)"
                feedback["d_score_contribution"] = 0.4
            else:
                feedback["status"] = "Fail"
                feedback["lock_status"] = "RED"
                feedback["deduction"] = "Downgraded: Insufficient Amplitude"
                feedback["skill"] = "Split Leap/Jump" if is_split else "Jump"
                feedback["d_score_contribution"] = 0.3 if is_split else 0.1

        # Specific Skills
        elif skill_name == "Iron Cross":
            feedback["d_score_contribution"] = 0.8
            feedback["type"] = "acro"
            feedback["status"] = "Pass"
        elif skill_name == "Handstand":
            feedback["d_score_contribution"] = 0.1
            feedback["type"] = "acro"
            feedback["status"] = "Pass"
        elif is_split:
            feedback["skill"] = "Split Leap/Jump"
            if split_pass:
                feedback["status"] = "Pass"
                feedback["d_score_contribution"] = 0.5
            else:
                feedback["status"] = "Deduction"
                feedback["deduction"] = "W-001: Split < 180°"
                feedback["d_score_contribution"] = 0.3

        return feedback

# Verify usage
gymnastics_analyzer = WAGAnalyzer()
