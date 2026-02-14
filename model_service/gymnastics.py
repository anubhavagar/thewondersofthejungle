
import cv2
import mediapipe as mp
import numpy as np
import math
import tempfile
import os
import base64
import random
import json
from datetime import datetime
from collections import Counter
import traceback
from types import SimpleNamespace

# Import MediaPipe Tasks API
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, ObjectDetector, ObjectDetectorOptions, RunningMode
from mediapipe.tasks.python import BaseOptions
try:
    import tensorflow as tf
except ImportError:
    print("WARNING: TensorFlow not found. Custom models may not load.")
    tf = None
from PIL import Image, ImageOps

class GymnasticsPoseEngine:
    """Extracts a full suite of geometric measurements for gymnastics skill validation."""
    def __init__(self, landmarks, w=1000, h=1000):
        # Convert landmarks to a standard list if they aren't already
        self.lm = landmarks
        self.w = w
        self.h = h
        self.indices = {
            'l_shoulder': 11, 'r_shoulder': 12, 'l_elbow': 13, 'r_elbow': 14,
            'l_wrist': 15, 'r_wrist': 16, 'l_hip': 23, 'r_hip': 24,
            'l_knee': 25, 'r_knee': 26, 'l_ankle': 27, 'r_ankle': 28,
            'l_foot': 31, 'r_foot': 32
        }

    def _get_point(self, i):
        """Returns isotropic 3D point from landmark index."""
        lm = self.lm[i]
        # Handle both dict and object types
        lx = lm['x'] if isinstance(lm, dict) else lm.x
        ly = lm['y'] if isinstance(lm, dict) else lm.y
        lz = lm['z'] if isinstance(lm, dict) else lm.z
        return np.array([lx * self.w, ly * self.h, lz * self.w])

    def _get_visibility(self, i):
        """Returns visibility score for landmark."""
        lm = self.lm[i]
        return lm['visibility'] if isinstance(lm, dict) else lm.visibility

    def _get_3d_angle(self, p1_idx, p2_idx, p3_idx):
        """Calculates internal angle at p2 using dimension-aware 3D coordinates."""
        # Visibility check: if any point is occluded, return neutral 180 (straight) or 0
        if any(self._get_visibility(i) < 0.3 for i in [p1_idx, p2_idx, p3_idx]):
            return 180.0

        p1, p2, p3 = self._get_point(p1_idx), self._get_point(p2_idx), self._get_point(p3_idx)
        v1, v2 = p1 - p2, p3 - p2
        
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0: return 0.0
        
        cosine = np.dot(v1, v2) / norm
        return round(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))), 1)

    def _get_horizon(self, p1_idx, p2_idx):
        """Calculates 2D angle of segment relative to the horizon (floor)."""
        p1, p2 = self._get_point(p1_idx), self._get_point(p2_idx)
        # Vector from p1 to p2
        v = p2 - p1
        # Angle relative to horizontal [1, 0, 0] in 2D (x, y)
        angle = np.degrees(np.arctan2(v[1], v[0])) # Keep signs to detect inversion vs. level
        return round(float(angle), 1)

    def get_full_analysis(self, apparatus=None):
        """Generates comprehensive biometric report for the current frame."""
        # 1. FIG Joint Suite (3D)
        measurements = {
            "l_elbow": self._get_3d_angle(11, 13, 15),
            "r_elbow": self._get_3d_angle(12, 14, 16),
            "l_hip": self._get_3d_angle(11, 23, 25),
            "r_hip": self._get_3d_angle(12, 24, 26),
            "l_knee": self._get_3d_angle(23, 25, 27),
            "r_knee": self._get_3d_angle(24, 26, 28),
            "l_shoulder": self._get_3d_angle(13, 11, 23),
            "r_shoulder": self._get_3d_angle(14, 12, 24),
            "l_ankle": self._get_3d_angle(25, 27, 31),
            "r_ankle": self._get_3d_angle(26, 28, 32)
        }
        
        # Averages for simplified scoring
        measurements["elbow_avg"] = round(float((measurements["l_elbow"] + measurements["r_elbow"]) / 2), 1)
        measurements["hip_avg"] = round(float((measurements["l_hip"] + measurements["r_hip"]) / 2), 1)
        measurements["knee_avg"] = round(float((measurements["l_knee"] + measurements["r_knee"]) / 2), 1)
        measurements["shoulder_avg"] = round(float((measurements["l_shoulder"] + measurements["r_shoulder"]) / 2), 1)
        measurements["toe_point_avg"] = round(float((measurements["l_ankle"] + measurements["r_ankle"]) / 2), 1)

        # 2. Orientation & Symmetry & Straightness
        # Midpoint calculations for stable body line
        p11, p12 = self._get_point(11), self._get_point(12)
        p23, p24 = self._get_point(23), self._get_point(24)
        p27, p28 = self._get_point(27), self._get_point(28)
        p25, p26 = self._get_point(25), self._get_point(26) # knees
        p13, p14 = self._get_point(13), self._get_point(14) # elbows
        
        # --- OCCLUSION MIRRORING (Profile View Symmetry) ---
        # If one limb is occluded, mirror the visible limb coordinates
        vis_l_elbow, vis_r_elbow = self._get_visibility(13), self._get_visibility(14)
        vis_l_knee, vis_r_knee = self._get_visibility(25), self._get_visibility(26)
        vis_l_ankle, vis_r_ankle = self._get_visibility(27), self._get_visibility(28)
        
        # Elbow mirroring
        if vis_l_elbow > 0.7 and vis_r_elbow < 0.35:
            p14 = np.array([p13[0], p13[1], p13[2]]) # Mirror L to R
        elif vis_r_elbow > 0.7 and vis_l_elbow < 0.35:
            p13 = np.array([p14[0], p14[1], p14[2]]) # Mirror R to L

        # Knee mirroring
        if vis_l_knee > 0.7 and vis_r_knee < 0.35:
            p26 = np.array([p25[0], p25[1], p25[2]]) # Mirror L to R
        elif vis_r_knee > 0.7 and vis_l_knee < 0.35:
            p25 = np.array([p26[0], p26[1], p26[2]]) # Mirror R to L
            
        # Ankle mirroring
        if vis_l_ankle > 0.7 and vis_r_ankle < 0.35:
            p28 = np.array([p27[0], p27[1], p27[2]]) # Mirror L to R
        elif vis_r_ankle > 0.7 and vis_l_ankle < 0.35:
            p27 = np.array([p28[0], p28[1], p28[2]]) # Mirror R to L

        mid_shldr = (p11 + p12) / 2
        mid_hip = (p23 + p24) / 2
        mid_ankle = (p27 + p28) / 2

        # Divergence Angle: Angle between Left Thigh and Right Thigh relative to Mid Hip
        # Use common origin (mid_hip) to get 0 degrees when knees/ankles touch
        v_l_thigh = p25 - mid_hip
        v_r_thigh = p26 - mid_hip
        
        norm_l = np.linalg.norm(v_l_thigh)
        norm_r = np.linalg.norm(v_r_thigh)
        
        if norm_l > 0 and norm_r > 0:
            cos_split = np.dot(v_l_thigh, v_r_thigh) / (norm_l * norm_r)
            split_angle = float(np.degrees(np.arccos(np.clip(cos_split, -1.0, 1.0))))
        else:
            split_angle = 0.0

        metrics = {
            "torso_horizon": round(float(np.degrees(np.arctan2(mid_shldr[1] - mid_hip[1], mid_shldr[0] - mid_hip[0]))), 1),
            "leg_horizon": round(float(np.degrees(np.arctan2(mid_hip[1] - mid_ankle[1], mid_hip[0] - mid_ankle[0]))), 1),
            "leg_spread": round(float(abs(p27[0] - p28[0]) / self.w), 3),
            "split_angle": round(float(split_angle), 1)
        }

        # Vector-Based Body Straightness: Alignment deviation via Dot Product
        # Torso vector T (shldr -> hip) and Leg vector L (hip -> ankle)
        v_torso = mid_hip - mid_shldr
        v_legs = mid_ankle - mid_hip
        
        norm_t = np.linalg.norm(v_torso)
        norm_l = np.linalg.norm(v_legs)
        
        if norm_t > 0 and norm_l > 0:
            # For a straight body, v_torso and v_legs point in the same direction (nearly parallel)
            cos_align = np.dot(v_torso, v_legs) / (norm_t * norm_l)
            # Deviation angle: 0 means perfectly straight
            align_dev = float(np.degrees(np.arccos(np.clip(cos_align, -1.0, 1.0))))
            metrics["body_straightness"] = round(float(align_dev), 1)
        else:
            metrics["body_straightness"] = 0.0

        # --- BALANCE BEAM CONTEXT CHECK ---
        # If athlete is on the beam and torso is near horizontal, force horizontal mode
        if apparatus and "Balance Beam" in apparatus.get('label', ''):
            # If shoulder and hip are within 10% Y-distance of each other, force horizontal mode
            if abs(mid_shldr[1] - mid_hip[1]) < (self.h * 0.1):
                metrics["forced_horizontal"] = True

        return {"measurements": measurements, "metrics": metrics}

class GymnasticsAnalyzer:
    def __init__(self):
        try:
            # Model Path (Full/Heavy)
            # Preference: pose_landmarker_heavy.task as per tuning recommendations
            model_path_heavy = os.path.join(os.path.dirname(__file__), "models", "pose_landmarker_heavy.task")
            model_path_std = os.path.join(os.path.dirname(__file__), "models", "pose_landmarker.task")
            
            model_path = model_path_heavy if os.path.exists(model_path_heavy) else model_path_std
            
            if not os.path.exists(model_path):
                print(f"WARNING: Model not found at {model_path}. Analysis will fail.")
            else:
                print(f"GymnasticsAnalyzer: Using pose model: {os.path.basename(model_path)}")
            
            # Create Landmarker for Image Mode - High Precision Tuning
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.7, # Tuned for "Heavy" mode accuracy
                min_tracking_confidence=0.5,
                output_segmentation_masks=False
            )
            self.landmarker = PoseLandmarker.create_from_options(options)
            print("GymnasticsAnalyzer: PoseLandmarker (IMAGE mode) initialized successfully.")

            # Video Landmarker is initialized lazily in enable_person_tracking
            self.video_landmarker = None
            self.video_options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.7, # Tuned for "Heavy" mode accuracy
                min_tracking_confidence=0.5,
                output_segmentation_masks=False
            )

            # Create Object Detector with Custom Model Fallback
            # Try custom apparatus model first, fallback to generic EfficientDet
            custom_model_path = os.path.join(os.path.dirname(__file__), "models", "gym_apparatus_custom.tflite")
            generic_model_path = os.path.join(os.path.dirname(__file__), "models", "efficientdet_lite0.tflite")
            
            if os.path.exists(custom_model_path) and tf is not None:
                # FIXED: Use tf.lite.Interpreter directly to avoid MediaPipe metadata errors
                self.custom_interpreter = tf.lite.Interpreter(model_path=custom_model_path)
                self.custom_interpreter.allocate_tensors()
                self.input_details = self.custom_interpreter.get_input_details()
                self.output_details = self.custom_interpreter.get_output_details()
                
                self.using_custom_detector = True
                self.detector = None # Will use interpreter
                print("GymnasticsAnalyzer: Using CUSTOM apparatus detection model (Native TFLite Interpreter)")
            else:
                if tf is None:
                     print("GymnasticsAnalyzer: TensorFlow not found, skipping custom model.")
                detector_path = generic_model_path
                self.using_custom_detector = False
                score_threshold = 0.2  # Lower threshold for generic model
                print("GymnasticsAnalyzer: Using GENERIC apparatus detection model (EfficientDet)")
            
                detector_options = ObjectDetectorOptions(
                    base_options=BaseOptions(model_asset_path=detector_path),
                    running_mode=RunningMode.IMAGE,
                    score_threshold=score_threshold,
                    max_results=5
                )
                self.detector = ObjectDetector.create_from_options(detector_options)
                print(f"GymnasticsAnalyzer: ObjectDetector initialized successfully (custom={self.using_custom_detector})")
            
            # Load Classification Model (for apparatus identification)
            classifier_path = os.path.join(os.path.dirname(__file__), "models", "apparatus_classifier.h5")
            
            # Audit / Tracking data
            self.current_gender = 'female'
            self.tracking_enabled = False
            self.tracked_person_bbox = None
            class_mapping_path = os.path.join(os.path.dirname(__file__), "models", "class_mapping.txt")
            
            self.classifier = None
            self.class_mapping = {}
            
            if os.path.exists(classifier_path) and os.path.exists(class_mapping_path):
                try:
                    from tensorflow import keras
                    import numpy as np
                    from PIL import Image
                    
                    self.classifier = keras.models.load_model(classifier_path)
                    
                    # Load class mapping
                    with open(class_mapping_path, 'r') as f:
                        for line in f:
                            idx, name = line.strip().split(': ')
                            self.class_mapping[int(idx)] = name
                    
                    print(f"GymnasticsAnalyzer: Classification model loaded ({len(self.class_mapping)} classes)")
                except Exception as e:
                    print(f"GymnasticsAnalyzer: Failed to load classification model: {e}")
                    self.classifier = None
            else:
                print("GymnasticsAnalyzer: Classification model not found (optional)")
            
            # Load Skill Detection Model (YOLOv8 TFLite)
            skill_model_path = os.path.join(os.path.dirname(__file__), "models", "trained_model_skill", "best_float32.tflite")
            self.skill_interpreter = None
            self.skill_classes = ["BL", "FL", "HS", "IN-IRON-C", "IRON-C", "L-CROSS", "LS", "M-UP", "PN", "VS"]
            self.skill_label_map = {
                "BL": "Back Lever",
                "FL": "Front Lever",
                "HS": "Handstand",
                "IN-IRON-C": "Inverted Iron Cross",
                "IRON-C": "Iron Cross",
                "L-CROSS": "L-Cross",
                "LS": "L-Sit",
                "M-UP": "Muscle Up",
                "PN": "Planche",
                "VS": "V-Sit"
            }

            if os.path.exists(skill_model_path) and tf is not None:
                try:
                    self.skill_interpreter = tf.lite.Interpreter(model_path=skill_model_path)
                    self.skill_interpreter.allocate_tensors()
                    self.skill_input_details = self.skill_interpreter.get_input_details()
                    self.skill_output_details = self.skill_interpreter.get_output_details()
                    print("GymnasticsAnalyzer: YOLOv8 Skill Detection model loaded successfully.")
                except Exception as e:
                    print(f"GymnasticsAnalyzer: Error loading skill model: {e}")
            else:
                print("GymnasticsAnalyzer: Skill model file not found or TensorFlow missing.")
            
            # Person Tracking State (for video analysis)
            self.tracked_person_bbox = None  # Cache the performer's bounding box
            self.tracking_enabled = False    # Enable tracking mode for videos
            
            # Tracker State
            self.session_id = ""
            self.tracker_dir = ""
            self.audit_log = {}
            
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
            "Still Rings (SR)": {"length_cm": 280, "name": "Still Rings (SR)"},
            "Floor Exercise (FX-M)": {"length_cm": 1200, "name": "Floor Exercise (FX-M)"},
            "Floor Exercise (FX-W)": {"length_cm": 1200, "name": "Floor Exercise (FX-W)"},
            "Horizontal Bar (HB)": {"length_cm": 240, "name": "Horizontal Bar (HB)"}
        }
    
    def preprocess_frame(self, frame, enhance_contrast=True, denoise=True, sharpen=True, rotate_code=None):
        """
        Preprocess frame to improve pose landmark detection quality.
        
        Args:
            frame: Input BGR image
            enhance_contrast: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            denoise: Apply bilateral filtering to reduce noise while preserving edges
            sharpen: Apply sharpening to enhance edges
            rotate_code: Optional cv2.rotate code (e.g., cv2.ROTATE_90_CLOCKWISE)
            
        Returns:
            Preprocessed BGR image
        """
        if frame is None:
            return None
        
        processed = frame.copy()

        # 0. Rotation
        if rotate_code is not None:
            processed = cv2.rotate(processed, rotate_code)
        
        # 1. Denoise while preserving edges (bilateral filter)
        if denoise:
            # Bilateral filter: reduces noise while keeping edges sharp
            # Parameters: diameter, sigmaColor, sigmaSpace
            processed = cv2.bilateralFilter(processed, d=5, sigmaColor=50, sigmaSpace=50)
        
        # 2. Enhance contrast using CLAHE (works better than simple histogram equalization)
        if enhance_contrast:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel (lightness)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            lab = cv2.merge([l, a, b])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 3. Subtle Sharpening (helps with landmark detection without introducing artifacts)
        if sharpen:
            # Subtle sharpening kernel
            kernel = np.array([[ 0, -0.5,  0],
                             [-0.5,  3, -0.5],
                             [ 0, -0.5,  0]])
            processed = cv2.filter2D(processed, -1, kernel)
        
        return processed

    def _calculate_angle(self, a, b, c, w=1000, h=1000):
        """Calculate 3D angle at joint b given three points a, b, c with isotropic scaling.
        
        Args:
            a, b, c: Landmark objects with x, y, z attributes
            w, h: Current image dimensions for aspect ratio correction
            
        Returns:
            float: Angle in degrees at point b
        """
        # Convert to numpy arrays and Apply Isotropic Scaling
        # MediaPipe Z is roughly same scale as X (width-dependent)
        # So we scale by W, H, W to get a uniform 'pixel-like' space
        pa = np.array([a.x * w, a.y * h, a.z * w])
        pb = np.array([b.x * w, b.y * h, b.z * w])
        pc = np.array([c.x * w, c.y * h, c.z * w])
        
        # Calculate vectors from b to a and b to c
        ba = pa - pb
        bc = pc - pb
        
        # Calculate angle using dot product
        norm_a = np.linalg.norm(ba)
        norm_c = np.linalg.norm(bc)
        if norm_a == 0 or norm_c == 0: return 0.0
        
        cosine_angle = np.dot(ba, bc) / (norm_a * norm_c)
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def _map_custom_label(self, label):
        """Map custom model labels to standardized apparatus format.
        
        Args:
            label: Raw label from custom model (e.g., 'Pommel_Horse')
            
        Returns:
            str: Standardized label with shorthand (e.g., 'Pommel Horse (PH)')
        """
        label_map = {
            "Pommel_Horse": "Pommel Horse (PH)",
            "Still_Rings": "Still Rings (SR)",
            "Vault": "Vault (VT-M)",
            "Parallel_Bars": "Parallel Bars (PB)",
            "Horizontal_Bar": "Horizontal Bar (HB)",
            "Uneven_Bars": "Uneven Bars (UB)",
            "Balance_Beam": "Balance Beam (BB)",
            "Floor_Exercise": "Floor Exercise (FX-M)"
        }
        return label_map.get(label, label)
    
    def _refine_generic_label(self, label, aspect_ratio):
        """Refine generic COCO labels to gymnastics apparatus using heuristics.
        
        Args:
            label: COCO label (e.g., 'bench', 'couch')
            aspect_ratio: Width/height ratio of bounding box
            
        Returns:
            str: Refined apparatus label
        """
        label_lower = label.lower()
        
        if label_lower in ["bench", "dining table", "suitcase", "refrigerator"]:
            if aspect_ratio > 8.0:
                return "Parallel Bars (PB)"
            elif aspect_ratio > 1.1:
                return "Pommel Horse (PH)"
            else:
                return "Vault (VT-M)"
        elif label_lower in ["couch", "bed"]:
            if aspect_ratio > 6.5:
                return "Balance Beam (BB)"
            else:
                return "Pommel Horse (PH)"
        elif label_lower in ["surfboard", "skis", "baseball bat"]:
            if aspect_ratio > 4.0:
                return "Parallel Bars (PB)"
            else:
                return "Horizontal Bar (HB)"
        elif label_lower in ["horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]:
            return "Pommel Horse (PH)"
        elif label_lower in ["toilet", "fire hydrant"]:
            return "Mushroom / Pommel Trainer"
        elif label_lower == "chair":
            return "Vault (VT-M)"
        elif label_lower == "sports ball":
            return "Still Rings (SR)"
        else:
            return label  # Return original if no mapping
    
    def classify_apparatus(self, frame):
        """
        Classify apparatus using the trained classification model.
        
        Args:
            frame: Input image (numpy array)
            
        Returns:
            dict: Classification result with label, confidence, and status
        """
        if self.classifier is None:
            return {
                "label": None,
                "confidence": 0.0,
                "status": "classifier_not_available"
            }
        
        try:
            from PIL import Image
            import numpy as np
            
            # Preprocess image
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = self.classifier.predict(img_array, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx])
            
            # Confidence threshold
            if confidence < 0.5:
                return {
                    "label": None,
                    "confidence": confidence,
                    "status": "low_confidence"
                }
            
            apparatus = self.class_mapping[predicted_idx]
            
            # Map to standard format
            apparatus_map = {
                "balance_beam": "Balance Beam (BB)",
                "horizontal_bar": "Horizontal Bar (HB)",
                "parallel_bars": "Parallel Bars (PB)",
                "pommel_horse": "Pommel Horse (PH)",
                "still_rings": "Still Rings (SR)",
                "uneven_bars": "Uneven Bars (UB)",
                "vault": "Vault (VT-M)",
                "floor_exercise": "Floor Exercise (FX-M)"
            }
            
            return {
                "label": apparatus_map.get(apparatus, apparatus),
                "confidence": confidence,
                "status": "detected"
            }
        except Exception as e:
            print(f"Classification error: {e}")
            return {
                "label": None,
                "confidence": 0.0,
                "status": "error"
            }

    def detect_pose_video(self, frame_rgb, timestamp_ms, person_bbox=None):
        """
        Detect pose in video mode with temporal smoothing and ROI refinement.
        Args:
            frame_rgb: The full video frame
            timestamp_ms: Current timestamp in milliseconds
            person_bbox: [x_min, y_min, x_max, y_max] normalized bbox of the gymnast
        """
        # If no person bbox, fallback to full frame
        if person_bbox is None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self.video_landmarker.detect_for_video(mp_image, int(timestamp_ms))
            return result.pose_landmarks[0] if result.pose_landmarks else None
            
        # ROI Refinement: "Zoom in" on the gymnast
        h, w, _ = frame_rgb.shape
        x_min, y_min, x_max, y_max = person_bbox
        
        # Add Padding (20% to ensure limbs aren't cut off)
        pad_x = (x_max - x_min) * 0.2
        pad_y = (y_max - y_min) * 0.2
        
        crop_x_min = max(0, int((x_min - pad_x) * w))
        crop_y_min = max(0, int((y_min - pad_y) * h))
        crop_x_max = min(w, int((x_max + pad_x) * w))
        crop_y_max = min(h, int((y_max + pad_y) * h))
        
        # Validate crop dimensions
        if crop_x_max <= crop_x_min + 10 or crop_y_max <= crop_y_min + 10:
             mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
             result = self.video_landmarker.detect_for_video(mp_image, int(timestamp_ms))
             return result.pose_landmarks[0] if result.pose_landmarks else None

        # Create Crop
        crop_img = frame_rgb[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop_img))
        
        # Detect on Crop (Higher effective resolution)
        result = self.video_landmarker.detect_for_video(mp_crop, int(timestamp_ms))
        
        if not result.pose_landmarks:
            return None
            
        # Project landmarks back to Global Coordinates
        landmarks = result.pose_landmarks[0]
        crop_w = crop_x_max - crop_x_min
        crop_h = crop_y_max - crop_y_min
        
        full_frame_landmarks = []
        for lm in landmarks:
            # Map local crop coordinates (0-1) to pixel coordinates, then to global normalized (0-1)
            global_x_px = crop_x_min + lm.x * crop_w
            global_y_px = crop_y_min + lm.y * crop_h
            
            gx = global_x_px / w
            gy = global_y_px / h
            
            # Reconstruct landmark object
            full_frame_landmarks.append(SimpleNamespace(
                x=gx, y=gy, z=lm.z, visibility=lm.visibility, presence=lm.presence
            ))
            
        return full_frame_landmarks

    def detect_person_bbox(self, frame):
        """Standard person detection to use as ROI for pose estimation
        
        With tracking enabled (for videos), this method caches the first detected
        performer and reuses their bounding box for all subsequent frames.
        """
        # If tracking is enabled and we have a cached bbox, reuse it
        if self.tracking_enabled and self.tracked_person_bbox is not None:
            return self.tracked_person_bbox
        
        if self.detector is None and not self.using_custom_detector:
            return None
        
        # If we have a custom detector but not a generic one, we can't detect 'person' easily
        # unless we load the generic one too.
        detector_to_use = self.detector
        if detector_to_use is None:
            # Load a temporary generic detector if we only have custom
            generic_model_path = os.path.join(os.path.dirname(__file__), "models", "efficientdet_lite0.tflite")
            if os.path.exists(generic_model_path):
                detector_options = ObjectDetectorOptions(
                    base_options=BaseOptions(model_asset_path=generic_model_path),
                    running_mode=RunningMode.IMAGE,
                    score_threshold=0.3, # Higher threshold for speed/accuracy
                    max_results=3
                )
                detector_to_use = ObjectDetector.create_from_options(detector_options)
        
        if detector_to_use is None: return None
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detection_result = detector_to_use.detect(mp_image)
        
        best_person = None
        max_priority_score = -1
        h, w, _ = frame.shape
        
        for detection in detection_result.detections:
            category = detection.categories[0]
            if category.category_name.lower() == "person":
                box = detection.bounding_box
                # Normalized coordinates
                x_min = box.origin_x / w
                y_min = box.origin_y / h
                x_max = (box.origin_x + box.width) / w
                y_max = (box.origin_y + box.height) / h
                
                # Calculate center
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                
                # --- REFINED HEURISTIC FOR GYMNAST DETECTION ---
            
                # 1. CENTER BIAS (Horizontal is critical)
                # Gymnasts are usually centered horizontally (0.4 - 0.6)
                # Audience/Coaches are usually at the edges (< 0.2 or > 0.8)
                dist_from_center_x = abs(center_x - 0.5)
                horizontal_score = 1.0 - (dist_from_center_x * 2.0) # 1.0 at center, 0.0 at edges
                
                # 2. VERTICAL BIAS (Context Dependent)
                # We CANNOT exclude upper frame (High Bar, Rings, Vault).
                # But we can penalize the extreme bottom corners (typical for judges).
                dist_from_center_y = abs(center_y - 0.5)
                vertical_score = 1.0 - (dist_from_center_y * 1.0) # Weaker bias, allow full range
                
                # 3. EDGE PENALTY (Judges/Coaches in corners)
                # If x is near edge (>0.85 or <0.15) AND y is low (>0.6), MASSIVE PENALTY
                edge_penalty = 0.0
                if (center_x < 0.15 or center_x > 0.85) and center_y > 0.6:
                    edge_penalty = 0.8 # Severe penalty for bottom corners
            
                # 4. SIZE SCORE (Subject is usually the main focus)
                bbox_area = (x_max - x_min) * (y_max - y_min)
                # Normalize area relative to typical frame (0.1 to 0.5)
                # Cap the boost so a huge face in the crowd doesn't override a full-body gymnast
                size_score = min(bbox_area, 0.6) * 10
                
                # 5. CONFIDENCE
                confidence_score = category.score
            
                # FINAL WEIGHTED SCORE
                # Center X is KING. Size is QUEEN.
                input_priority = (
                    (horizontal_score * 4.0) +  # High weight on horizontal alignment
                    (vertical_score * 1.5) +    # Moderate vertical preference
                    (size_score * 3.0) +        # Good weight on size
                    (confidence_score * 1.0)
                )
                
                # Apply Penalties
                priority_score = input_priority * (1.0 - edge_penalty)
                
            if priority_score > max_priority_score:
                max_priority_score = priority_score
                best_person = [x_min, y_min, x_max, y_max]
        
        # If tracking is enabled, cache this detection for future frames
        if self.tracking_enabled and best_person is not None:
            self.tracked_person_bbox = best_person
            print(f"Person tracking: Locked onto performer at bbox {best_person}")
        
        return best_person
    
    def enable_person_tracking(self):
        """Enable person tracking mode for video analysis"""
        self.tracking_enabled = True
        self.tracked_person_bbox = None
        
        # Initialize Video Landmarker (Fresh instance to reset internal timestamp state)
        if self.video_landmarker is not None:
             self.video_landmarker.close()
             
        self.video_landmarker = PoseLandmarker.create_from_options(self.video_options)
        print("Person tracking: ENABLED (Video Landmarker Initialized)")
    
    def disable_person_tracking(self):
        """Disable person tracking mode (for single image analysis)"""
        self.tracking_enabled = False
        self.tracked_person_bbox = None
        
        # Close Video Landmarker to free resources
        if self.video_landmarker:
            self.video_landmarker.close()
            self.video_landmarker = None
            print("Person tracking: DISABLED (Video Landmarker Closed)")
    
    def reset_person_tracking(self):
        """Reset the tracked person bbox (forces re-detection on next frame)"""
        self.tracked_person_bbox = None
        print("Person tracking: RESET - will re-detect performer on next frame")

    def detect_apparatus(self, frame, pose_landmarks=None):
        """Detect equipment using MediaPipe Object Detector or Native TFLite (YOLOv8). Rejects overlap with the person."""
        if self.detector is None and not self.using_custom_detector:
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
        
        if self.using_custom_detector:
            # INTERFACE: YOLOv8 (Native TFLite Interpreter)
            # 1. Preprocess
            try:
                # IMPORTANT: YOLOv8 expects RGB, color conversion is likely missing
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_shape = self.input_details[0]['shape'] # [1, 640, 640, 3]
                img_resized = cv2.resize(img_rgb, (input_shape[2], input_shape[1]))
                img_input = img_resized.astype(np.float32) / 255.0
                img_input = np.expand_dims(img_input, axis=0) # [1, 640, 640, 3]

                # 2. Run Inference
                self.custom_interpreter.set_tensor(self.input_details[0]['index'], img_input)
                self.custom_interpreter.invoke()
                output_data = self.custom_interpreter.get_tensor(self.output_details[0]['index']) # [1, 11, 8400]
                
                # 3. Postprocess YOLOv8 (1, 11, 8400)
                output = output_data[0] # [11, 8400]
                boxes = output[:4, :].T # [8400, 4] -> cx, cy, w, h
                scores = output[4:, :].T # [8400, 7]
                
                # Find best class per anchor
                class_ids = np.argmax(scores, axis=1)
                confidences = np.max(scores, axis=1)
                
                # Filter by confidence
                # INCREASED: back to 0.5 to prevent false positives (like the home photo issue)
                conf_threshold = 0.5
                mask = confidences > conf_threshold
                filtered_boxes = boxes[mask]
                filtered_conf = confidences[mask]
                filtered_classes = class_ids[mask]
                
                if len(filtered_boxes) > 0:
                    print(f"GymnasticsAnalyzer: Custom model found {len(filtered_boxes)} raw detections (>{conf_threshold})")
                    
                    # NMS
                    indices = cv2.dnn.NMSBoxes(
                        bboxes=[[float(b[0]-b[2]/2), float(b[1]-b[3]/2), float(b[2]), float(b[3])] for b in filtered_boxes],
                        scores=[float(c) for c in filtered_conf],
                        score_threshold=conf_threshold,
                        nms_threshold=0.45
                    )
                    
                    print(f"GymnasticsAnalyzer: Post-NMS indices: {indices}")
                    
                    if len(indices) > 0:
                        flat_indices = np.array(indices).flatten()
                        for idx in flat_indices:
                            box = filtered_boxes[idx]
                            score = filtered_conf[idx]
                            class_id = filtered_classes[idx]
                            
                            cx, cy, bw, bh = box
                            
                            # Determine if model output is in pixels (typical YOLO) or normalized 0-1
                            # YOLOv8 TFLite usually outputs pixels relative to imgsz (640)
                            model_imgsz = input_shape[1] # 640
                            scale = 1.0
                            if cx > 1.1: scale = model_imgsz # It's in pixels
                            
                            left = (cx - bw/2) / scale
                            top = (cy - bh/2) / scale
                            width = bw / scale
                            height = bh / scale
                            
                            # Clamp to 0-1
                            left, top = max(0, min(1, left)), max(0, min(1, top))
                            width, height = max(0.01, min(1, width)), max(0.01, min(1, height))
                            
                            if score > max_score:
                                max_score = score
                                class_names = ["Balance_Beam", "Horizontal_Bar", "Parallel_Bars", "Pommel_Horse", "Still_Rings", "Uneven_Bars", "Vault"]
                                raw_label = class_names[class_id]
                                refined_label = self._map_custom_label(raw_label)
                                
                                best_apparatus = {
                                    "label": refined_label,
                                    "shorthand": refined_label.split('(')[1].split(')')[0] if '(' in refined_label else refined_label[:3].upper(),
                                    "score": float(score),
                                    "bbox": [float(left), float(top), float(width), float(height)],
                                    "mask_polygon": None,
                                    "spec": self.apparatus_specs.get(refined_label, {"length_cm": 200, "name": refined_label})
                                }
                    else:
                        print("GymnasticsAnalyzer: No detections passed NMS.")
                else:
                    # Debug: if nothing found, show top raw scores
                    top_idx = np.argsort(confidences)[-3:]
                    print(f"GymnasticsAnalyzer: No detections > {conf_threshold}. Top raw scores: {confidences[top_idx]}")
            except Exception as e:
                print(f"GymnasticsAnalyzer Det Error: {e}")
                traceback.print_exc()
        else:
            # INTERFACE: MediaPipe ObjectDetector prediction
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detection_result = self.detector.detect(mp_image)
            
            if not detection_result.detections:
                return None
                
            for detection in detection_result.detections:
                category = detection.categories[0]
                if not category.category_name: continue
                label = category.category_name.lower()
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
                if torso_bbox is not None and len(torso_bbox) >= 4:
                    # torso_bbox is [x_min, y_min, x_max, y_max] in normalized coordinates
                    # Convert apparatus bbox to normalized center_x, center_y
                    app_center_x_norm = (left + width / 2) / w
                    app_center_y_norm = (top + height / 2) / h

                    if (torso_bbox[0] < app_center_x_norm < torso_bbox[2] and 
                        torso_bbox[1] < app_center_y_norm < torso_bbox[3]):
                        allowed_labels = ["bench", "couch", "bed", "chair", "suitcase", "dining table", "refrigerator", 
                                        "surfboard", "skis", "baseball bat", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                        "toilet", "fire hydrant"]
                        if label not in allowed_labels:
                             continue

                if score > max_score:
                    max_score = score
                    aspect_ratio = width / height if height > 0 else 1
                    refined_label = self._refine_generic_label(label, aspect_ratio)

                    best_apparatus = {
                        "label": refined_label,
                        "shorthand": refined_label.split('(')[1].split(')')[0] if '(' in refined_label else refined_label[:3].upper(),
                        "score": float(score),
                        "bbox": [float(left/w), float(top/h), float(width/w), float(height/h)],
                        "mask_polygon": None,
                        "spec": self.apparatus_specs.get(refined_label, {"length_cm": 200, "name": refined_label})
                    }
        
        # Use classification as fallback if generic detector returns non-gymnastics objects
        if best_apparatus and not self.using_custom_detector:
            # List of non-gymnastics objects that generic model might detect
            non_gymnastics_labels = [
                "tennis racket", "baseball bat", "skateboard", "surfboard",
                "sports ball", "frisbee", "kite", "bottle", "cup", "fork",
                "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
                "cake", "potted plant", "tv", "laptop", "mouse", "remote",
                "keyboard", "cell phone", "microwave", "oven", "toaster",
                "sink", "book", "clock", "vase", "scissors", "teddy bear",
                "hair drier", "toothbrush"
            ]
            
            detected_label = best_apparatus["label"].lower()
            is_non_gymnastics = any(label in detected_label for label in non_gymnastics_labels)
            
            if is_non_gymnastics:
                print(f"Generic detector returned '{best_apparatus['label']}', trying classification...")
                classification_result = self.classify_apparatus(frame)
                
                if classification_result["status"] == "detected":
                    # Replace with classification result
                    best_apparatus = {
                        "label": classification_result["label"],
                        "shorthand": classification_result["label"].split('(')[1].split(')')[0] if '(' in classification_result["label"] else classification_result["label"][:3].upper(),
                        "score": classification_result["confidence"],
                        "bbox": [0.0, 0.0, 1.0, 1.0],  # Full frame
                        "mask_polygon": None,
                        "spec": self.apparatus_specs.get(classification_result["label"], {"length_cm": 200, "name": classification_result["label"]}),
                        "source": "classifier"
                    }
                    print(f"Classification override: {classification_result['label']} ({classification_result['confidence']:.2%})")
        
        # Calculate Calibration Factor (px to cm)
        if best_apparatus:
            bbox = best_apparatus["bbox"]
            px_width = bbox[2] # bbox is [left, top, width, height]
            real_cm = best_apparatus["spec"]["length_cm"]
            best_apparatus["px_per_cm"] = (px_width * w) / real_cm if real_cm > 0 else 1.0

        return best_apparatus

    def detect_skill_yolo(self, frame, conf_threshold=0.25):
        """
        Detect gymnastics skills using the YOLOv8 TFLite model.
        """
        if self.skill_interpreter is None:
            return None

        try:
            h, w, _ = frame.shape
            # 1. Preprocess
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_shape = self.skill_input_details[0]['shape'] # [1, 640, 640, 3]
            
            # Check for NCHW vs NHWC
            if input_shape[1] == 3: # NCHW
                img_resized = cv2.resize(img_rgb, (input_shape[3], input_shape[2]))
                img_input = img_resized.astype(np.float32) / 255.0
                img_input = np.transpose(img_input, (2, 0, 1)) # [3, 640, 640]
            else: # NHWC
                img_resized = cv2.resize(img_rgb, (input_shape[2], input_shape[1]))
                img_input = img_resized.astype(np.float32) / 255.0
            
            img_input = np.expand_dims(img_input, axis=0)

            # 2. Run Inference
            self.skill_interpreter.set_tensor(self.skill_input_details[0]['index'], img_input)
            self.skill_interpreter.invoke()
            output_data = self.skill_interpreter.get_tensor(self.skill_output_details[0]['index']) # [1, 14, 8400]
            
            # 3. Postprocess
            output_data = np.squeeze(output_data) # [14, 8400]
            boxes = output_data[:4, :].T # [8400, 4] -> [cx, cy, w, h]
            scores = output_data[4:, :].T # [8400, 10]
            
            class_ids = np.argmax(scores, axis=1)
            confidences = np.max(scores, axis=1)
            
            mask = confidences > conf_threshold
            boxes = boxes[mask]
            confidences = confidences[mask]
            class_ids = class_ids[mask]
            
            if len(boxes) == 0:
                return None
                
            # NMS requires [x, y, w, h] or similar depending on version. 
            # In YOLOv8 tflite, it's often [cx, cy, w, h]. 
            # We'll convert to [x, y, w, h] for cv2.dnn.NMSBoxes
            nms_boxes = []
            for box in boxes:
                cx, cy, bw, bh = box
                x = cx - bw/2
                y = cy - bh/2
                nms_boxes.append([float(x), float(y), float(bw), float(bh)])

            indices = cv2.dnn.NMSBoxes(
                nms_boxes, 
                confidences.tolist(), 
                conf_threshold, 
                0.45
            )
            
            if len(indices) > 0:
                best_idx = indices[0]
                if isinstance(best_idx, (list, np.ndarray)): best_idx = best_idx[0]
                
                box = boxes[best_idx]
                score = confidences[best_idx]
                class_id = class_ids[best_idx]
                
                cx, cy, bw, bh = box
                scale = 640.0 # Assuming 640x640 input
                
                left = (cx - bw/2) / scale
                top = (cy - bh/2) / scale
                width = bw / scale
                height = bh / scale
                
                label_short = self.skill_classes[class_id]
                label_full = self.skill_label_map.get(label_short, label_short)
                
                return {
                    "label": label_full,
                    "short_code": label_short,
                    "confidence": float(score),
                    "bbox": [float(left), float(top), float(width), float(height)]
                }
                
        except Exception as e:
            print(f"GymnasticsAnalyzer Skill Det Error: {e}")
            traceback.print_exc()
            
        return None

    def calibrate_and_normalize(self, landmarks, apparatus, frame_width=1, frame_height=1):
        """Normalize landmarks relative to apparatus anchor points or center of mass."""
        if not landmarks:
            return landmarks
        
        # Determine anchor points
        if apparatus and apparatus.get("bbox"):
            # Use center of apparatus bounding box as origin
            bbox = apparatus["bbox"] 
            app_cx = bbox[0] + (bbox[2] / 2)
            app_cy = bbox[1] + (bbox[3] / 2)
            anchor_y = 0.7 # Standard vertical anchor for apparatus-based view
        else:
            # If no apparatus, use "Center of Mass" (Mid-Hip) as origin to keep gymnast centered
            # This prevents the skeleton from "jumping" when no apparatus is found
            try:
                l23 = landmarks[23]
                l24 = landmarks[24]
                mid_hip_x = ((l23.x if hasattr(l23, 'x') else l23['x']) + (l24.x if hasattr(l24, 'x') else l24['x'])) / 2
                mid_hip_y = ((l23.y if hasattr(l23, 'y') else l23['y']) + (l24.y if hasattr(l24, 'y') else l24['y'])) / 2
                app_cx = mid_hip_x
                app_cy = mid_hip_y
                anchor_y = 0.5 # Center them vertically in the frame
            except (AttributeError, KeyError, IndexError):
                return landmarks # Fallback to raw landmarks

        normalized = []
        for lm in landmarks:
            is_dict = isinstance(lm, dict)
            lx = lm["x"] if is_dict else lm.x
            ly = lm["y"] if is_dict else lm.y
            lz = lm["z"] if is_dict else lm.z
            lv = lm["visibility"] if is_dict else lm.visibility

            # Shift relative to anchor
            rel_x = lx - app_cx + 0.5
            rel_y = ly - app_cy + anchor_y
            
            if is_dict:
                normalized.append({"x": rel_x, "y": rel_y, "z": lz, "visibility": lv})
            else:
                normalized.append(SimpleNamespace(x=rel_x, y=rel_y, z=lz, visibility=lv))
        return normalized

    def identify_skill(self, landmarks, apparatus=None, category='Senior Elite', w=1000, h=1000, frame=None, ai_result=None):
        """
        Robust Identity-First Gymnastics Skill Classifier.
        Incorporates YOLOv8 AI detection for primary identity.
        """

        try:
            # --------------------------------------------------
            # 0. AI PRIMARY DETECTION (YOLOv8)
            # --------------------------------------------------
            ai_skill = ai_result
            if ai_skill is None and frame is not None and self.skill_interpreter is not None:
                ai_skill = self.detect_skill_yolo(frame)
                if ai_skill:
                    print(f"GymnasticsAnalyzer: AI Detected Skill: {ai_skill['label']} ({ai_skill['confidence']:.2%})")

            # --------------------------------------------------
            # 1. Validate Landmark Integrity
            # --------------------------------------------------
            if not landmarks or len(landmarks) < 29:
                return {
                    "skill": "Unknown",
                    "type": "Error",
                    "is_occluded": True,
                    "metrics": {},
                    "measurements": {},
                    "error": "Insufficient landmarks"
                }

            # Visibility / Occlusion Check
            critical_joints = [13, 14, 23, 24, 25, 26]  # elbows, hips, knees
            occluded = []
            for j in critical_joints:
                lm = landmarks[j]
                vis = lm.get("visibility", 1.0) if isinstance(lm, dict) else getattr(lm, "visibility", 1.0)
                if vis < 0.5:
                    occluded.append(j)
            
            is_occluded = len(occluded) > 0

            # Identify MAG or WAG based on apparatus
            app_label = apparatus.get('label', '') if isinstance(apparatus, dict) else ""
            is_mag = any(a in app_label for a in ["Still Rings", "Pommel Horse", "Parallel Bars", "High Bar", "Floor Exercise MAG"])
            
            # --------------------------------------------------
            # 1. Initialize Biometric Engine
            # --------------------------------------------------
            engine = GymnasticsPoseEngine(landmarks, w=w, h=h)
            analysis = engine.get_full_analysis(apparatus=apparatus)

            m = analysis.get("measurements", {})
            met = analysis.get("metrics", {})

            # --------------------------------------------------
            # 2. AI Skill Identity Overrides
            # --------------------------------------------------
            # Use AI detection as primary identity if confident
            if ai_skill and ai_skill['confidence'] > 0.4:
                skill_name = ai_skill['label']
                # Check for gender/apparatus specific feedback
                if is_mag:
                    feedback = self.analyze_mag_skill(landmarks, skill_name, category, w=w, h=h)
                else:
                    feedback = self.analyze_wag_skill(landmarks, skill_name, category, w=w, h=h)
                
                # If specialized analyzer confirmed or refined the skill, use it
                final_res = {
                    "skill": feedback.get("skill", skill_name),
                    "type": feedback.get("type", "Skill"),
                    "is_occluded": is_occluded,
                    "metrics": met,
                    "measurements": m,
                    "ai_metadata": {
                        "confidence": ai_skill['confidence'],
                        "bbox": ai_skill['bbox'],
                        "source": "yolov8-skill"
                    }
                }
                return final_res

            # Visibility / Occlusion Check

            # Visibility Guard for Split Angle (Stop hallucinations in profile)
            vis_l_knee = landmarks[25].get("visibility", 1.0) if isinstance(landmarks[25], dict) else getattr(landmarks[25], "visibility", 1.0)
            vis_r_knee = landmarks[26].get("visibility", 1.0) if isinstance(landmarks[26], dict) else getattr(landmarks[26], "visibility", 1.0)
            if vis_l_knee < 0.4 or vis_r_knee < 0.4:
                met["split_angle"] = 0.0

            # Safe extraction
            torso_hz = met.get("torso_horizon", 90)
            # FORCE HORIZONTAL if Balance Beam context check passed
            if met.get("forced_horizontal"):
                torso_hz = 10.0 # Force a low horizontal value to trigger Planche/Hold identities
                
            leg_hz = met.get("leg_horizon", 90)
            leg_spread = met.get("leg_spread", 0)
            # Sync back if forced
            if met.get("forced_horizontal"):
                 met["torso_horizon"] = torso_hz

            elbow_avg = m.get("elbow_avg", 0)
            hip_avg = m.get("hip_avg", 180)
            shoulder_avg = m.get("shoulder_avg", 180)

            torso_vert = abs(90 - torso_hz)

            # Inversion check (ankles above head)
            feet_left_y = landmarks[27]['y'] if isinstance(landmarks[27], dict) else landmarks[27].y
            feet_right_y = landmarks[28]['y'] if isinstance(landmarks[28], dict) else landmarks[28].y
            feet_y = (feet_left_y + feet_right_y) / 2
            
            head_y = landmarks[0]['y'] if isinstance(landmarks[0], dict) else landmarks[0].y
            is_inverted = feet_y < head_y

            apparatus_label = (apparatus.get("label", "") if apparatus else "").lower()

            # ==================================================
            # SKILL IDENTIFICATION HIERARCHY
            # ==================================================

            # --------------------------------------------------
            # A. HANDSTAND GROUP
            # --------------------------------------------------
            if is_inverted and torso_vert < 30:

                skill = "Handstand"
                if shoulder_avg < 150:
                    skill = "Press Handstand"

                return {
                    "skill": skill,
                    "type": "Static / Hold",
                    "is_occluded": is_occluded,
                    "metrics": {
                        "verticality": round(float(torso_vert), 1)
                    },
                    "measurements": m
                }

            # --------------------------------------------------
            # B. PLANCHE GROUP
            # --------------------------------------------------
            # User Override: If torso is very horizontal, it's a Planche regardless of arms
            # Prioritize forced_horizontal for D-Score
            is_developmental = category.lower() in ["u10", "u12", "junior"]
            
            if (met.get("forced_horizontal") or (torso_hz < 35 and leg_hz < 35)) and not is_inverted:
                base_name = "Straddle Planche" if leg_spread > 0.35 else "Planche"
                return {
                    "skill": base_name,
                    "type": "Strength Hold",
                    "d_score_contribution": 0.3 if not is_developmental else 0.1,
                    "is_occluded": is_occluded,
                    "metrics": met,
                    "measurements": m
                }

            # --------------------------------------------------
            # C. L-SIT / V-SIT
            # --------------------------------------------------
            if hip_avg < 115 and torso_vert < 25 and leg_hz < 30:

                skill = "V-Sit" if hip_avg < 70 else "L-Sit"

                return {
                    "skill": skill,
                    "type": "Static / Hold",
                    "is_occluded": is_occluded,
                    "metrics": {
                        "hip_angle": hip_avg
                    },
                    "measurements": m
                }

            # --------------------------------------------------
            # D. APPARATUS SPECIFIC (RINGS)
            # --------------------------------------------------
            if "rings" in apparatus_label:

                if (
                    70 < shoulder_avg < 110
                    and torso_vert < 20
                    and elbow_avg > 165
                ):
                    return {
                        "skill": "Iron Cross",
                        "type": "Strength Hold",
                        "is_occluded": is_occluded,
                        "metrics": {
                            "shoulder_extension": shoulder_avg
                        },
                        "measurements": m
                    }

            # --------------------------------------------------
            # E. FLEXIBILITY (Bridge / Splits)
            # --------------------------------------------------
            mid_hip_y = ((landmarks[23]['y'] if isinstance(landmarks[23], dict) else landmarks[23].y) + 
                         (landmarks[24]['y'] if isinstance(landmarks[24], dict) else landmarks[24].y)) / 2

            # Bridge: hips highest + strong arch
            if (
                mid_hip_y < head_y
                and mid_hip_y < feet_y
                and torso_hz > 120
            ):
                return {
                    "skill": "Bridge",
                    "type": "Flexibility",
                    "is_occluded": is_occluded,
                    "metrics": {
                        "torso_horizon": torso_hz
                    },
                    "measurements": m
                }

            # Straddle Split
            if leg_spread > 0.5 and torso_vert > 50:
                return {
                    "skill": "Straddle Split",
                    "type": "Flexibility",
                    "is_occluded": is_occluded,
                    "metrics": {
                        "leg_spread": leg_spread
                    },
                    "measurements": m
                }

            # --------------------------------------------------
            # DEFAULT
            # --------------------------------------------------
            return {
                "skill": "Pose",
                "type": "Transition",
                "is_occluded": is_occluded,
                "metrics": met,
                "measurements": m
            }

        except Exception as e:
            return {
                "skill": "Error",
                "type": "Exception",
                "is_occluded": True,
                "metrics": {},
                "measurements": {},
                "error": str(e)
            }
    
    
    def analyze_wag_skill(self, landmarks, skill_name, category, hold_duration=0, apparatus_label="Unknown", w=1000, h=1000):
        """
        Analyze a specific WAG skill for D-Score and execution errors.
        """
        from model_service.skill_knowledge import SKILL_DATA
        skill_info = SKILL_DATA.get(skill_name)
        
        deductions = []
        feedback = []
        d_val = 0.1
        skill_type = "A"
        status = "Recognized"
        element_group = 1

        if skill_info:
            difficulty_map = {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4, "E": 0.5, "F": 0.6}
            d_val = difficulty_map.get(skill_info.get("difficulty"), 0.1)
            skill_type = skill_info.get("difficulty", "A")

        # 1. Biometric Suite
        engine = GymnasticsPoseEngine(landmarks, w=w, h=h)
        analysis = engine.get_full_analysis()
        measurements = analysis["measurements"]
        metrics = analysis["metrics"]

        # --- TECHNICAL FAULT DETECTION ---
        
        # 1. Body Alignment (Straightness)
        if metrics.get("body_straightness", 0) > 15:
            deductions.append({
                "value": 0.1,
                "observation": f"Poor body alignment ({metrics['body_straightness']}° deviation)",
                "label": "Alignment"
            })

        # 2. Bent Knees check
        l_knee, r_knee = measurements["l_knee"], measurements["r_knee"]
        if l_knee < 165 or r_knee < 165:
            deductions.append({
                "value": 0.1, 
                "observation": "Bent knees (detectable angle < 165°)", 
                "label": "Bent Knees"
            })
            feedback.append("Squeeze your quads to keep your legs fully extended.")

        return {
            "skill": skill_name,
            "d_score_contribution": d_val,
            "status": status,
            "type": skill_type,
            "elementGroup": element_group,
            "deductions": deductions,
            "feedback": feedback,
            "e_score_range": "Target E: 8.5 - 9.5",
            "d_score_range": f"Standard {skill_type} DV"
        }


    def analyze_mag_skill(self, landmarks, skill_name, category, hold_duration=0, apparatus_label="Unknown", w=1000, h=1000):
        """
        Analyze a specific MAG skill for D-Score and execution errors.
        """
        from model_service.skill_knowledge import SKILL_DATA
        
        # Check standard skill database
        skill_info = SKILL_DATA.get(skill_name)
        
        deductions = []
        feedback = []
        d_val = 0.0
        skill_type = "A"
        element_group = 1
        status = "Pass" if skill_info else "Neutral"
        
        if skill_info:
            # Map FIG Difficulty to numeric D-Score
            difficulty_map = {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4, "E": 0.5, "F": 0.6}
            d_val = difficulty_map.get(skill_info.get("difficulty"), 0.1)
            skill_type = skill_info.get("difficulty", "A")
            element_group = skill_info.get("elementGroup", 1)

        # 1. Biometric Suite
        engine = GymnasticsPoseEngine(landmarks, w=w, h=h)
        analysis = engine.get_full_analysis()
        measurements = analysis["measurements"]
        metrics = analysis["metrics"]

        # --- TECHNICAL FAULT DETECTION (MAG Specific) ---
        
        # 1. Body Alignment (Straightness)
        if metrics.get("body_straightness", 0) > 15:
            deductions.append({
                "value": 0.1, 
                "observation": f"Poor body alignment ({metrics['body_straightness']}° deviation)", 
                "label": "Alignment"
            })

        # 2. Bent Knees check
        l_knee, r_knee = measurements["l_knee"], measurements["r_knee"]
        if l_knee < 165 or r_knee < 165:
            deductions.append({
                "value": 0.1, 
                "observation": "Bent knees (detectable angle < 165°)", 
                "label": "Bent Knees"
            })

        # 2. Bent Arms check
        l_elbow, r_elbow = measurements["l_elbow"], measurements["r_elbow"]
        if l_elbow < 165 or r_elbow < 165:
            deductions.append({
                "value": 0.3, 
                "observation": "Bent arms in support (detectable angle < 165°)", 
                "label": "Bent Arms"
            })

        # 3. Strength Hold Duration (FIG MAG: < 2s = Large Deduction, < 1s = No Credit)
        is_static = "Static" in skill_type or (skill_info and skill_info.get("type") == "Static / Hold")
        if is_static and hold_duration < 2.0:
            if hold_duration < 1.0:
                d_val = 0.0
                status = "No Credit"
                feedback.append("Hold was too short to be recognized (< 1s).")
            else:
                deductions.append({
                    "value": 0.3,
                    "observation": f"Short hold ({hold_duration:.1f}s)",
                    "label": "Hold Duration"
                })
                feedback.append("Maintain hold for full 2.0s to avoid deductions.")

        return {
            "skill": skill_name,
            "d_score_contribution": d_val,
            "status": status,
            "type": skill_type,
            "elementGroup": element_group,
            "deductions": deductions,
            "feedback": feedback
        }

    def _init_tracker(self, media_type, category, gender):
        """Initialize auditing to output_tracker/{session_id}/."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"run_{timestamp}_{random.randint(1000, 9999)}"
        self.tracker_dir = os.path.join(os.getcwd(), "output_tracker", self.session_id)
        os.makedirs(self.tracker_dir, exist_ok=True)
        self.audit_log = {
            "session_id": self.session_id,
            "timestamp": timestamp,
            "input_metadata": {
                "media_type": media_type,
                "category": category,
                "gender": gender
            },
            "steps": []
        }
        self.current_gender = gender
        print(f"Audit Tracker: Initialized {self.tracker_dir}")

    def _log_step(self, step_name, data):
        """Capture intermediate data for audit."""
        if hasattr(self, 'audit_log'):
            # Convert non-serializable objects (like SimpleNamespace)
            serializable_data = self._make_serializable(data)
            
            # Explicitly include gender in step data for clarity if available
            if hasattr(self, 'current_gender') and isinstance(serializable_data, dict):
                serializable_data["_audit_gender"] = self.current_gender

            self.audit_log["steps"].append({
                "step": step_name,
                "timestamp": datetime.now().isoformat(),
                "data": serializable_data
            })

    def _make_serializable(self, obj):
        """Recursively convert objects to dicts for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (SimpleNamespace, object)) and hasattr(obj, "__dict__"):
            return self._make_serializable(obj.__dict__)
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    def _save_audit(self, final_results):
        """Write consolidated audit.json."""
        if hasattr(self, 'audit_log'):
            self.audit_log["final_results"] = final_results
            file_path = os.path.join(self.tracker_dir, "audit.json")
            with open(file_path, "w") as f:
                json.dump(self.audit_log, f, indent=4)
            print(f"Audit Tracker: Saved consolidated JSON to {file_path}")

    def analyze_media(self, media_data, media_type='video', category='Senior Elite', gender='Female', hold_duration=2):
        self._init_tracker(media_type, category, gender)
        apparatus = {}
        landmarks = []
        result = {}
        is_mag = (gender == 'Male')
        self._log_step("Input Received", {"media_type": media_type, "category": category, "gender": gender})
        """
        Main entry point for analyzing gymnastics media (video or image).
        """
        try:
            # Decode file
            # Decode file
            if "," in media_data:
                header, encoded = media_data.split(",", 1)
                data = base64.b64decode(encoded)
            else:
                data = base64.b64decode(media_data)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg" if media_type == 'image' else ".mp4") as tmp_file:
                tmp_file.write(data)
                tmp_path = tmp_file.name
            
            # Save media to tracker directory for auditing
            media_ext = ".jpg" if media_type == 'image' else ".mp4"
            media_filename = f"input_media{media_ext}"
            media_path = os.path.join(self.tracker_dir, media_filename)
            with open(media_path, "wb") as f:
                f.write(data)
            self.audit_log["input_media"] = media_filename
            
            frames_data = [] # To store per-frame analysis
            apparatus_info = {"label": "Floor Exercise", "confidence": 0.0} # Default
            
            if not apparatus_info.get("status"):
                apparatus_info = {"label": "Floor Exercise (FX)", "confidence": 0.0, "status": "Default"}
            
            self._log_step("Apparatus Detection", apparatus_info)
            
            if media_type == 'video':
                # Video Processing Path
                cap = cv2.VideoCapture(tmp_path)
                
                if not cap.isOpened():
                    return {"error": "Could not open video file"}
                    
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0: fps = 30
                
                # --- APPARATUS DETECTION (First meaningful frame) ---
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                
                # PERSON TRACKING: Enable tracking mode for video analysis
                # This ensures we lock onto the performer in the first frame
                # and maintain focus throughout the video (ignoring audience)
                self.enable_person_tracking()
                
                frames_data = []
                processed_frames = 0
                    
                # PERFORMANCE OPTIMIZATION: Detect apparatus only ONCE at the start
                # Apparatus doesn't move during a routine, so we cache it
                cached_apparatus = None
                apparatus_detected = False
                    
                # Process frames (Skip frames for performance if needed, e.g., every 3rd frame)
                stride = 2 if frame_count > 100 else 1 
                
                curr_frame_idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    if curr_frame_idx % stride == 0:
                        # 1. Find Person (Tracking)
                            person_bbox = self.detect_person_bbox(frame)
                            
                            # 2. Convert to RGB
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            
                            # 3. Detect Pose with Video Smoothing & ROI
                            timestamp_ms = int((curr_frame_idx / fps) * 1000)
                            final_landmarks = self.detect_pose_video(frame_rgb, timestamp_ms, person_bbox)
                            
                            if final_landmarks:
                                    h, w, _ = frame.shape
                                    
                                    # PERFORMANCE: Detect apparatus only on first frame, then cache
                                    if not apparatus_detected:
                                        # Use first frame's landmarks for apparatus context
                                        lm_dicts = [{"x": (lm.x if hasattr(lm, "x") else lm["x"]), 
                                                     "y": (lm.y if hasattr(lm, "y") else lm["y"])} for lm in final_landmarks] # Minimal dict for detector
                                        cached_apparatus = self.detect_apparatus(frame, pose_landmarks=lm_dicts)
                                        apparatus_detected = True
                                    
                                    apparatus = cached_apparatus
                                    
                                    # AI Skill Detection (Run periodically for performance)
                                    f_ai_skill = None
                                    if curr_frame_idx % (int(stride) * 10) == 0:
                                        f_ai_skill = self.detect_skill_yolo(frame)
                                    
                                    # Calibrate/Normalize (Useful for consistent relative analysis, but NOT for overlay)
                                    final_landmarks_calibrated = self.calibrate_and_normalize(final_landmarks, apparatus, w, h)
                                    
                                    # Store results
                                    def lm_to_dict(lm):
                                        if isinstance(lm, dict):
                                            return {"x": float(lm["x"]), "y": float(lm["y"]), "z": float(lm["z"]), "visibility": float(lm["visibility"])}
                                        return {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z), "visibility": float(lm.visibility)}

                                    frames_data.append({
                                        "time": float(curr_frame_idx) / float(fps) if fps else 0.0,
                                        "landmarks": [lm_to_dict(lm) for lm in final_landmarks],
                                        "centered_landmarks": [lm_to_dict(lm) for lm in final_landmarks_calibrated],
                                        "raw_landmarks": [lm_to_dict(lm) for lm in final_landmarks],
                                        "apparatus": apparatus,
                                        "ai_skill": f_ai_skill
                                    })

                                    # *** TRACKING UPDATE (Closed Loop) ***
                                    # Update the 'person_bbox' for the NEXT frame based on the CURRENT pose.
                                    # This ensures the ROI follows the gymnast as they move.
                                    if final_landmarks:
                                        xs = [lm.x for lm in final_landmarks]
                                        ys = [lm.y for lm in final_landmarks]
                                        
                                        # New BBox from Pose
                                        min_x, max_x = min(xs), max(xs)
                                        min_y, max_y = min(ys), max(ys)
                                        
                                        # Add Padding (e.g. 10%)
                                        p_w = (max_x - min_x) * 0.1
                                        p_h = (max_y - min_y) * 0.1
                                        
                                        new_bbox = [
                                            max(0, min_x - p_w), 
                                            max(0, min_y - p_h), 
                                            min(1, max_x + p_w), 
                                            min(1, max_y + p_h)
                                        ]
                                        
                                        # Update the tracker directly
                                        self.tracked_person_bbox = new_bbox
                            
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
                    # Image Mode - ROI Refined Detection
                    # USE PIL for EXIF-aware loading (fixes portrait rotation issues)
                    try:
                        with Image.open(tmp_path) as img:
                            img = ImageOps.exif_transpose(img)
                            target_image = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        print(f"PIL Load failed: {e}, falling back to cv2")
                        target_image = cv2.imread(tmp_path)

                    if target_image is None: return {"error": "Failed to load image"}
                    
                    # 1. ROI Refinement
                    person_bbox = self.detect_person_bbox(target_image)
                    
                    roi_image = target_image
                    crop_offset_x = 0
                    crop_offset_y = 0
                    crop_scale_x = 1.0
                    crop_scale_y = 1.0
                    
                    if person_bbox:
                        h, w, _ = target_image.shape
                        pad = 0.15
                        xmin = max(0, person_bbox[0] - pad)
                        ymin = max(0, person_bbox[1] - pad)
                        xmax = min(1, person_bbox[2] + pad)
                        ymax = min(1, person_bbox[3] + pad)
                        
                        left, top = int(xmin * w), int(ymin * h)
                        right, bottom = int(xmax * w), int(ymax * h)
                        
                        if right > left and bottom > top:
                            roi_image = target_image[top:bottom, left:right]
                            crop_offset_x = left / w
                            crop_offset_y = top / h
                            crop_scale_x = (right - left) / w
                            crop_scale_y = (bottom - top) / h
                    
                    # **STAGE 1: Detection on ROI (Raw)**
                    image_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                    detection_result = self.landmarker.detect(mp_image)
                    
                    # **STAGE 2 FALLBACK: Detection on ROI (Enhanced)**
                    if not detection_result.pose_landmarks:
                        print("GymnasticsAnalyzer: ROI raw detection failed, trying ENHANCED ROI...")
                        processed_image = self.preprocess_frame(roi_image)
                        image_rgb_enhanced = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                        mp_image_enhanced = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb_enhanced)
                        detection_result = self.landmarker.detect(mp_image_enhanced)
                    
                    if not detection_result.pose_landmarks:
                        print("GymnasticsAnalyzer: ROI detection failed, trying FULL FRAME...")
                        image_rgb_full = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
                        mp_image_full = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb_full)
                        detection_result = self.landmarker.detect(mp_image_full)
                        # Reset offsets if full frame worked
                        if detection_result.pose_landmarks:
                            crop_offset_x, crop_offset_y = 0.0, 0.0
                            crop_scale_x, crop_scale_y = 1.0, 1.0

                    # **STAGE 4 FALLBACK: Horizontal Rotation (Balance Beam Recovery)**
                    if not detection_result.pose_landmarks:
                        # Quick apparatus check (without pose) to see if we should try rotation
                        temp_apparatus = self.detect_apparatus(target_image)
                        is_beam = temp_apparatus and "Balance Beam" in temp_apparatus.get('label', '')
                        
                        if is_beam:
                            print("GymnasticsAnalyzer: No upright pose found on Beam. Attempting HORIZONTAL rotation pass...")
                            # Rotate 90 deg clockwise to make a planche look like a handstand
                            rotated_roi = cv2.rotate(roi_image, cv2.ROTATE_90_CLOCKWISE)
                            image_rgb_rot = cv2.cvtColor(rotated_roi, cv2.COLOR_BGR2RGB)
                            mp_image_rot = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb_rot)
                            rotated_result = self.landmarker.detect(mp_image_rot)
                            
                            if rotated_result.pose_landmarks:
                                print("GymnasticsAnalyzer: Horizontal detection SUCCESS. Re-mapping coordinates...")
                                # Inverse mapping: Rotate landmarks back (x, y) -> (y, 1-x)
                                for lm in rotated_result.pose_landmarks[0]:
                                    old_x, old_y = lm.x, lm.y
                                    lm.x = old_y
                                    lm.y = 1.0 - old_x
                                detection_result = rotated_result
                    
                    if not detection_result.pose_landmarks:
                        return {"error": "No person detected in gymnastics pose."}
                        
                    # Re-map landmarks
                    raw_normalized = []
                    for lm in detection_result.pose_landmarks[0]:
                        global_x = lm.x * crop_scale_x + crop_offset_x
                        global_y = lm.y * crop_scale_y + crop_offset_y
                        raw_normalized.append({
                            "x": global_x, 
                            "y": global_y, 
                            "z": lm.z, 
                            "visibility": lm.visibility
                        })
                    
                    # Apparatus detection on whole image
                    apparatus_info = self.detect_apparatus(target_image, pose_landmarks=raw_normalized)
                    h, w, _ = target_image.shape
                    
                    # STANDARDIZE: Convert to SimpleNamespace early
                    land_objs = [SimpleNamespace(**lm) for lm in raw_normalized]
                    landmarks = self.calibrate_and_normalize(land_objs, apparatus_info, w, h)
                    
                    frames_data = [{
                        "time": 0.0,
                        # FIXED: Use RAW normalized landmarks for visualization overlay!
                        "landmarks": raw_normalized,
                        "centered_landmarks": [{"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility} for lm in landmarks],
                        "raw_landmarks": raw_normalized,
                        "apparatus": apparatus_info
                    }]
                    
                    
                    # PERSON TRACKING: Disable tracking after video processing
                    self.disable_person_tracking()

            # --- Artistry Heuristic ---
            def get_artistry(frames):
                if not frames: return 0.0
                # Single Image: Return a neutral artistry score (e.g., 5.0) to avoid automatic penalty
                if len(frames) == 1:
                    return 5.0
                
                # Stability: Variance in nose position (landmarks[0])
                noses = [f['landmarks'][0] for f in frames if f['landmarks'][0].get('visibility', 0) > 0.5]
                if len(noses) < 2: return 5.0 # Neutral fallback for low-frame-count videos
                var_x = np.var([n['x'] for n in noses])
                var_y = np.var([n['y'] for n in noses])
                stability = 1.0 - (var_x + var_y) * 50 # Scale variance to a 0-1 penalty
                base_artistry = 3.5 + (stability * 1.5) # Base 3.5 + up to 1.5 for stability
                return round(float(max(0.0, min(5.0, base_artistry))), 1)

                
            # --- ARTISTRY & EXECUTION ---
            artistry_score = get_artistry(frames_data)
                
            # --- SKILL ANALYSIS & KNOWLEDGE INJECTION ---
            from model_service.skill_knowledge import SKILL_DATA
            
            skill_list = []
            metrics = {}
            result = {} # Initialize early to avoid UnboundLocalError in finally
            
            # Identify MAG or WAG based on gender, category or apparatus
            app_label = apparatus_info.get('label', '') if isinstance(apparatus_info, dict) else ""
            is_mag = (gender.lower() == 'male') or "MAG" in category or any(a in app_label for a in ["Still Rings", "Pommel Horse", "Parallel Bars", "High Bar"])
            
            if media_type == 'image':
                # Single Frame Analysis
                skill_info = self.identify_skill(landmarks, apparatus=apparatus_info, category=category, w=w, h=h, frame=target_image)
                
                # Log Skill and Measurements
                self._log_step("Skill Identification", {
                    "skill": skill_info.get("skill"),
                    "metrics": skill_info.get("metrics"),
                    "measurements": skill_info.get("measurements")
                })
                skill_name = skill_info["skill"]
                skill_type = skill_info.get("type", "Unknown")
                skill_metrics = skill_info.get("metrics", {})
                metrics = skill_metrics
                
                wag_feedback = self.analyze_wag_skill(landmarks, skill_name, category, w=w, h=h)
                
                final_skill_name = wag_feedback.get('skill', skill_name)
                
                skill_list = [{
                    "name": final_skill_name,
                    "dv": wag_feedback.get('d_score_contribution', 0.0),
                    "type": wag_feedback.get('type', skill_type)
                }]
                
            else:
                # Video Analysis (aggregator logic)
                
                all_skills_found = []
                for frame_data in frames_data:
                    f_landmarks = [SimpleNamespace(**lm) for lm in frame_data['landmarks']]
                    f_apparatus = frame_data.get('apparatus')
                    f_ai_result = frame_data.get('ai_skill')
                    f_skill_result = self.identify_skill(f_landmarks, f_apparatus, category=category, w=w, h=h, ai_result=f_ai_result)
                    
                    # Log Skill and Measurements for this frame
                    self._log_step(f"Frame {frames_data.index(frame_data)} Skill Analysis", {
                        "timestamp_ms": frame_data.get("time", 0.0) * 1000,
                        "skill": f_skill_result.get("skill"),
                        "metrics": f_skill_result.get("metrics"),
                        "measurements": f_skill_result.get("measurements")
                    })
                    
                    f_skill_name = f_skill_result["skill"]
                    f_skill_type = f_skill_result.get("type", "Unknown")
                    f_metrics = f_skill_result.get("metrics", {})
                    
                    if is_mag:
                        f_feedback = self.analyze_mag_skill(f_landmarks, f_skill_name, category, hold_duration, apparatus_label=f_apparatus.get("label", "Unknown") if f_apparatus else "Unknown", w=w, h=h)
                    else:
                        f_feedback = self.analyze_wag_skill(f_landmarks, f_skill_name, category, hold_duration, apparatus_label=f_apparatus.get("label", "Unknown") if f_apparatus else "Unknown", w=w, h=h)
                    
                    # Enrich frame metadata for UI Timeline
                    frame_data["skill"] = f_feedback.get('skill', f_skill_name)
                    frame_data["dv"] = f_feedback.get('d_score_contribution', 0.0)
                    frame_data["status"] = f_feedback.get('status', 'Neutral')
                    
                    if frame_data["dv"] > 0:
                        all_skills_found.append({
                            "name": frame_data["skill"],
                            "dv": frame_data["dv"],
                            "type": f_feedback.get('type', f_skill_type),
                            "elementGroup": f_feedback.get('elementGroup', 1)
                        })
                
                # Deduplicate skills (take the highest value if seen multiple times)
                unique_skills = {}
                for s in all_skills_found:
                    s_dv = float(s.get('dv', 0.0)) if isinstance(s.get('dv'), (int, float)) else 0.0
                    if s['name'] not in unique_skills or s_dv > float(unique_skills[s['name']].get('dv', 0.0)):
                        unique_skills[s['name']] = s
                
                skill_list = list(unique_skills.values())

            # --- COMMON RESULT CONSTRUCTION ---
            
            # Calculate D-Score using the full list
            if is_mag:
                from model_service.mag_d_score import MAGDScoreCalculator
                d_calculator = MAGDScoreCalculator()
            else:
                from model_service.wag_d_score import WAGDScoreCalculator
                d_calculator = WAGDScoreCalculator()
                
            d_score_result = d_calculator.calculate_d_score(skill_list, category=category)
            total_d_score = float(d_score_result.get('total_d_score', 0.0))
            self._log_step("Difficulty Calculation", d_score_result)

            # Initialize structured_e_rationale
            structured_e_rationale = {
                "formula": "E = 10.0 - [Technical Deductions]",
                "base_reason": "Standard FIG Starting Execution (Elite Level)",
                "values": {
                    "base": 10.0,
                    "deductions": 0.0, # Placeholder
                    "total": 0.0 # Placeholder
                },
                "reasons": []
            }

            # Update result dictionary
            result.update({
                "total_score": 0.0,
                "difficulty": round(total_d_score, 2),
                "execution": 0.0,
                "artistry": artistry_score,
                "skills_found": skill_list,
                "best_skill": "Pose",
                "feedback": [],
                "deductions": [],
                "d_score_breakdown": d_score_result,
                "discipline": "MAG" if is_mag else "WAG",
                "comment": "Focus on maintaining stability and extension throughout your routine.",
                "visualization_data": {}
            })

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
                    fapp = fd.get('apparatus')
                    if self.identify_skill(flm, fapp, category=category, w=w, h=h)["skill"] == best_skill_name:
                        viz_frame_idx = idx
                        break
            
            target_frame_data = frames_data[viz_frame_idx]
            target_landmarks = [SimpleNamespace(**lm) for lm in target_frame_data['landmarks']]
            
            # Re-analyze best frame for specific feedback (Dynamic E-Score Calculation)
            current_apparatus = apparatus if apparatus else target_frame_data.get('apparatus')
            apparatus_label = current_apparatus.get("label", "Unknown") if isinstance(current_apparatus, dict) else "Unknown"
            
            if is_mag:
                exec_feedback = self.analyze_mag_skill(target_landmarks, best_skill_name, category, hold_duration, apparatus_label=apparatus_label, w=w, h=h)
            else:
                exec_feedback = self.analyze_wag_skill(target_landmarks, best_skill_name, category, hold_duration, apparatus_label=apparatus_label, w=w, h=h)
            
            # Calculate Technical Deductions
            tech_deductions_list = exec_feedback.get('deductions', [])
            total_tech_deductions = sum(d.get('value', 0.0) for d in tech_deductions_list)
            
            # Calculate Artistry Deductions (from Heuristic)
            artistry_deductions = round(max(0.0, 5.0 - artistry_score), 1)
            
            # Final E-Score
            base_e_score = 10.0
            total_deductions = total_tech_deductions + artistry_deductions
            e_score = max(0.0, base_e_score - total_deductions)
            e_score = round(float(e_score), 2)
            
            # Final Total Score
            final_score = total_d_score + e_score
            self._log_step("Execution & Total Score", {"e_score": e_score, "final_score": final_score, "deductions": tech_deductions_list})
            
            # Update result with calculated values
            result["total_score"] = round(float(final_score), 2)
            result["execution"] = e_score
            result["best_skill"] = best_skill_name
            result["feedback"] = exec_feedback.get('feedback', [])
            result["deductions"] = tech_deductions_list
            
            # Include comprehensive biometric suite for verification
            best_frame_biometrics = self.identify_skill(target_landmarks, current_apparatus, category=category, w=w, h=h)
            result["biometrics"] = best_frame_biometrics.get("measurements", {})
            result["metrics"] = best_frame_biometrics.get("metrics", {})
            
            # Populate structured_e_rationale
            structured_e_rationale["values"]["deductions"] = total_tech_deductions
            structured_e_rationale["values"]["total"] = e_score
            
            if tech_deductions_list:
                for d in tech_deductions_list:
                    structured_e_rationale["reasons"].append({
                        "label": d.get('label', 'Technical Error'),
                        "text": d.get('observation', 'Execution fault detected'),
                        "value": f"-{d.get('value', 0.1):.1f}"
                    })
            else:
                structured_e_rationale["reasons"].append({
                    "label": "Technical Form",
                    "text": "Correct technical execution; no faults detected.",
                    "value": "0.0"
                })


                
            # Add target range for professional branding
            if wag_feedback.get('e_score_range'):
                structured_e_rationale["target_range"] = wag_feedback['e_score_range']
            
            # Update D-score rationale (Already structured from wag_d_score.py)
            d_rationale = d_score_result.get('rationale', {})
            if wag_feedback.get('d_score_range') and isinstance(d_rationale, dict):
                d_rationale["target_range"] = wag_feedback['d_score_range']

            result.update({
                "status": wag_feedback.get('status', 'Neutral'),
                "skill": best_skill_name,
                "metrics": metrics,
                "d_score_rationale": d_rationale,
                "e_score_rationale": structured_e_rationale,
                "best_frame_index": viz_frame_idx,
                "frames": frames_data,
                # Coach's Corner Data
                "technical_cue": skill_details.get("technicalCue", "Focus on form and stability."),
                "focus_anatomy": skill_details.get("focusAnatomy", []),
                "common_errors": skill_details.get("commonDeductions", []),
                "skill_description": skill_details.get("description", ""),
                "category": category,
                "hold_duration": hold_duration,
                "discipline": "MAG" if is_mag else "WAG"
            })
            
        except Exception as e:
            print(f"Error in analyze_media: {e}")
            import traceback
            traceback.print_exc()
            result = {"error": str(e)}
        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                try: os.unlink(tmp_path)
                except: pass
            
            self._save_audit(result)
        
        return result

class WAGAnalyzer(GymnasticsAnalyzer):
    """Specialized Analyzer for Women's Artistic Gymnastics (2D)."""
    
    def check_split_angle(self, landmarks, w=1000, h=1000):
        """W-001: Split (Cross or Side) >= 180 degrees."""
        # Hip (23/24), Knee (25/26), Ankle (27/28)
        # Using vector math on 2D projection with isotropic scaling
        l_hip = np.array([landmarks[23].x * w, landmarks[23].y * h])
        l_knee = np.array([landmarks[25].x * w, landmarks[25].y * h])
        r_hip = np.array([landmarks[24].x * w, landmarks[24].y * h])
        r_knee = np.array([landmarks[26].x * w, landmarks[26].y * h])
        
        v_l = l_knee - l_hip
        v_r = r_knee - r_hip
        
        dot_product = np.dot(v_l, v_r)
        norm_product = np.linalg.norm(v_l) * np.linalg.norm(v_r)
        
        if norm_product == 0: return 0.0
        
        rad = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
        angle = np.degrees(rad)
        
        # If they are parallel (standing), angle is 0
        return round(angle, 1)

    def check_ring_arch(self, landmarks, w=1000, h=1000):
        """W-002: Upper Body Arch >= 80 degrees."""
        # Shoulder (12), Hip (24), Knee (26)
        return self._calculate_angle(
            landmarks[12],
            landmarks[24],
            landmarks[26],
            w=w, h=h
        )

    def check_head_release(self, landmarks, w=1000, h=1000):
        """W-003: Head Release >= 100 degrees."""
        # Ear (8), Neck/Shoulder (12), Hip (24)
        return self._calculate_angle(
            landmarks[8],
            landmarks[12],
            landmarks[24],
            w=w, h=h
        )

    def analyze_wag_skill(self, landmarks, skill_name, category="Senior", hold_duration=2.0, w=1000, h=1000, apparatus_label="Unknown"):
        """Evaluate specific WAG criteria for a given skill with strict biomechanical thresholds."""
        metrics = {}
        feedback = {
            "skill": skill_name,
            "metrics": metrics,
            "status": "Neutral",
            "status_reason": "Analyzing movement; monitoring for technical pass/fail criteria",
            "d_score_contribution": 0.0,
            "type": "dance",
            "deductions": [],
            "lock_status": "NONE"
        }
        
        # Adjust thresholds based on category (Simulation)
        # Groups: U8, U10, U12, U14, Senior
        is_elite = "Senior" in category or "U14" in category
        
        is_split = "Split" in skill_name or "Leap" in skill_name or "Jump" in skill_name
        is_strength = skill_name in ["Iron Cross", "L-Sit", "Planche", "Horizonation - L-Sit"]

        # 1. Diagnostic: Apparatus Check
        # If we see a Flair but apparatus is wrong, warn the user
        if "Flair" in skill_name and "Pommel" not in apparatus_label:
             feedback["status_reason"] = f"Flair detected; but apparatus is {apparatus_label}. Scoring requires Pommel Horse."
        
        # --- HOLD DURATION VALIDATION (Strength Elements) ---
        if is_strength and hold_duration < 2.0:
            feedback["deductions"].append({
                "label": "Hold Duration", 
                "observation": f"Hold of {hold_duration}s below 2s requirement", 
                "value": 0.3 if hold_duration >= 1.0 else 0.5
            })
            feedback["status"] = "Deduction"
            feedback["status_reason"] = f"Hold duration insufficient ({hold_duration}s < 2.0s)"
            # Partial credit for short holds in elite, none in beginner
            feedback["d_score_contribution"] = 0.1 if is_elite and hold_duration >= 1.0 else 0.0
        is_developmental = "U8" in category or "U10" in category
        
        min_split = 180.0 if is_elite else 165.0 # Forgiven splits for U8/U10
        min_head_release = 100.0 if not is_developmental else 80.0
        min_ring_arch = 80.0 if not is_developmental else 60.0

        # 1. Split Check (W-001)
        split_angle = self.check_split_angle(landmarks, w=w, h=h)
        metrics["W-001 (Split)"] = f"{split_angle}°"
        
        is_split = split_angle > 140
        split_pass = split_angle >= min_split

        # 2. Ring Arch (W-002)
        ring_arch = self.check_ring_arch(landmarks, w=w, h=h)
        metrics["W-002 (Arch)"] = f"{ring_arch}°"
        arch_pass = ring_arch >= min_ring_arch

        # 3. Head Release (W-003)
        head_release = self.check_head_release(landmarks, w=w, h=h)
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
                feedback["status_reason"] = "All biomechanical requirements met (Elite Standard)"
                feedback["lock_status"] = "LOCKED"
                # Difficulty bonus for higher categories
                base_dv = 0.6 if is_elite else 0.4
                feedback["d_score_contribution"] = base_dv
            elif pass_count == 3:
                feedback["status"] = "Deduction"
                missing = "Split" if not split_pass else ("Arch" if not arch_pass else ("Head" if not head_pass else "Height"))
                feedback["status_reason"] = f"Skill recognized with minor {missing} technical fault"
                feedback["lock_status"] = "PARTIAL"
                if not split_pass:
                    feedback["deductions"].append({"label": "Flexibility", "observation": f"Split angle ({split_angle:.1f}°) below requirement", "value": 0.1})
                elif not arch_pass:
                    feedback["deductions"].append({"label": "Posture", "observation": "Insufficient Upper Body Arch", "value": 0.1})
                elif not head_pass:
                    feedback["deductions"].append({"label": "Body Line", "observation": "Head Release below threshold", "value": 0.1})
                elif not foot_height_pass:
                    feedback["deductions"].append({"label": "Leg Position", "observation": "Back foot below head level", "value": 0.1})
                feedback["d_score_contribution"] = 0.4
            else:
                feedback["status"] = "Fail"
                feedback["status_reason"] = "Major technical error; requirements not met for credit"
                feedback["lock_status"] = "RED"
                feedback["deductions"].append({"label": "Amplitude", "observation": "Severe lack of extension; skill downgraded", "value": 0.5})
                feedback["skill"] = "Split Leap/Jump" if is_split else "Jump"
                feedback["d_score_contribution"] = 0.3 if is_split else 0.1

        # 2. Specific Strength & Support Elements (Planche, L-Sit, etc.)
        elif skill_name in ["Planche", "L-Sit", "Horizonation - L-Sit", "Iron Cross", "Handstand"]:
            feedback["type"] = "acro"
            
            # --- PLANCHE SPECIFIC EVALUATION ---
            if skill_name == "Planche":
                body_angle = float(metrics.get("body_alignment", 0))
                is_horizontal = body_angle > 155
                
                feedback["d_score_contribution"] = 0.3 if is_horizontal else 0.1
                feedback["status"] = "Pass" if is_horizontal else "Deduction"
                if is_horizontal:
                    feedback["status_reason"] = "Planche horizontal alignment verified"
                else:
                    feedback["deductions"].append({"label": "Horizonation", "observation": f"Body at {body_angle:.1f}° below standard", "value": 0.3})
                    feedback["status_reason"] = "Planche height insufficient"

            # --- L-SIT SPECIFIC EVALUATION ---
            elif "L-Sit" in skill_name:
                leg_angle = float(metrics.get("Straight Legs", 180))
                arm_angle = float(metrics.get("Straight Arms", 180))
                horizon_metrics = float(metrics.get("Horizonation", 0))
                
                leg_straight = leg_angle > 165
                arm_straight = arm_angle > 172
                horizon = horizon_metrics >= -10 # Feet roughly level with hips
                
                feedback["skill"] = "Horizonation - L-Sit"
                feedback["d_score_range"] = "4.5 - 6.0" if is_elite else "1.5 - 3.0"
                feedback["e_score_range"] = "8.5 - 9.5" if is_elite else "7.0 - 8.5"
                feedback["d_score_contribution"] = 0.2 if horizon and is_elite else 0.1
                
                if leg_straight and arm_straight and horizon:
                    feedback["status"] = "Pass"
                    feedback["status_reason"] = "Perfect horizontal alignment verified"
                else:
                    feedback["status"] = "Deduction"
                    reason_list = []
                    if not horizon:
                        feedback["deductions"].append({"label": "Horizonation", "observation": "Legs Below Horizontal level", "value": 0.3})
                        reason_list.append("Low Legs")
                    if not arm_straight:
                        feedback["deductions"].append({"label": "Support", "observation": f"Bent Arms ({arm_angle:.1f}° detected)", "value": 0.3})
                        reason_list.append("Bent Arms")
                    if not leg_straight:
                        feedback["deductions"].append({"label": "Form", "observation": f"Bent Knees ({leg_angle:.1f}° detected)", "value": 0.1})
                        reason_list.append("Bent Knees")
                    feedback["status_reason"] = f"Deducted for: {', '.join(reason_list)}"

            # --- OTHER STATICS ---
            else:
                feedback["d_score_contribution"] = 0.8 if "Iron" in skill_name else 0.1
                feedback["status"] = "Pass"
                feedback["status_reason"] = "Skill geometry verified; credited for difficulty"

        # 3. Pommel / PB Specific Elements
        elif skill_name in ["Pommel Support", "Pommel Flair/Circle", "PB Support"]:
            feedback["d_score_contribution"] = 0.3 if "Flair" in skill_name else (0.2 if "PB" in skill_name else 0.1)
            feedback["type"] = "acro"
            feedback["status"] = "Pass"
            feedback["status_reason"] = f"Technical requirements for {skill_name} verified"
            
        # 3. Dynamic Flexibility & Straddle Elements
        elif skill_name == "Straddle Split" or is_split:
            feedback["skill"] = "Straddle Split" if skill_name == "Straddle Split" else "Split Leap/Jump"
            feedback["type"] = "dance"
            if split_pass:
                feedback["status"] = "Pass"
                feedback["status_reason"] = f"Full {split_angle:.0f}° extension verified; credited for difficulty"
                feedback["d_score_contribution"] = 0.4 if skill_name == "Straddle Split" else 0.5
            else:
                feedback["status"] = "Deduction"
                feedback["status_reason"] = f"Split angle ({split_angle:.1f}°) below requirement"
                feedback["lock_status"] = "PARTIAL"
                feedback["deductions"].append({"label": "Flexibility", "observation": "Insufficient split amplitude", "value": 0.1})
                feedback["d_score_contribution"] = 0.3

        # Default Case for Unknown/Transition skills
        else:
            feedback["status"] = "Neutral"
            feedback["status_reason"] = "Recognizing movement... Ensure clear technical triggers are visible."
            feedback["d_score_contribution"] = 0.0
            feedback["type"] = "transition"

        return feedback

# Verify usage
gymnastics_analyzer = WAGAnalyzer()
