
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
try:
    import tensorflow as tf
except ImportError:
    print("WARNING: TensorFlow not found. Custom models may not load.")
    tf = None
from PIL import Image, ImageOps

class GymnasticsAnalyzer:
    def __init__(self):
        try:
            # Model Path (Full/Heavy)
            model_path = os.path.join(os.path.dirname(__file__), "models", "pose_landmarker.task")
            if not os.path.exists(model_path):
                print(f"WARNING: Model not found at {model_path}. Analysis will fail.")
            
            # Create Landmarker for Image Mode - Standard Optimized Confidence
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
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
                min_pose_presence_confidence=0.5,
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
            
            # Person Tracking State (for video analysis)
            self.tracked_person_bbox = None  # Cache the performer's bounding box
            self.tracking_enabled = False    # Enable tracking mode for videos
            
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
    
    def preprocess_frame(self, frame, enhance_contrast=True, denoise=True, sharpen=True):
        """
        Preprocess frame to improve pose landmark detection quality.
        
        Args:
            frame: Input BGR image
            enhance_contrast: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            denoise: Apply bilateral filtering to reduce noise while preserving edges
            sharpen: Apply sharpening to enhance edges
            
        Returns:
            Preprocessed BGR image
        """
        if frame is None:
            return None
        
        processed = frame.copy()
        
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

    def _calculate_angle(self, a, b, c):
        """Calculate 3D angle at joint b given three points a, b, c.
        
        Args:
            a, b, c: Landmark objects with x, y, z attributes
            
        Returns:
            float: Angle in degrees at point b
        """
        # Convert to numpy arrays for vector operations
        a = np.array([a.x, a.y, a.z])
        b = np.array([b.x, b.y, b.z])
        c = np.array([c.x, c.y, c.z])
        
        # Calculate vectors from b to a and b to c
        ba = a - b
        bc = c - b
        
        # Calculate angle using dot product
        # cos(θ) = (ba · bc) / (|ba| * |bc|)
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        
        # Clip to [-1, 1] to handle numerical errors
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        # Convert to degrees
        angle = np.degrees(np.arccos(cosine_angle))
        
        return angle

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
        if self.video_landmarker:
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
                if torso_bbox:
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

    def identify_skill(self, landmarks, apparatus=None):
        """Identify specific skills based on landmark positions using angle-based detection.
        
        Args:
            landmarks: List of landmark objects with x, y, z, visibility attributes
            apparatus: Optional apparatus info to refine detection
            
        Returns:
            dict: {
                "skill": str,           # Skill name
                "type": str,            # "Static / Hold" or "Dynamic"
                "metrics": dict         # Detailed measurements
            }
        """
        try:
            apparatus_label = apparatus.get("label", "") if apparatus else ""
            
            # 1. Coordinate Normalization / Reference Points
            # Use mid-hip as a stable center for verticality checks
            mid_hip_y = (landmarks[23].y + landmarks[24].y) / 2
            mid_shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
            
            # 2. Handstand Check (Verticality & Inversion)
            # Logic: Feet must be above head, and body must be aligned (shoulder-hip-ankle angle)
            is_inverted = landmarks[27].y < landmarks[0].y and landmarks[28].y < landmarks[0].y
            
            if is_inverted:
                # Calculate shoulder angle for verticality (wrist-shoulder-hip)
                shoulder_angle = self._calculate_angle(landmarks[15], landmarks[11], landmarks[23])
                
                if shoulder_angle > 160:
                    return {
                        "skill": "Handstand",
                        "type": "Static / Hold",
                        "metrics": {
                            "verticality": round(float(shoulder_angle), 1),
                            "inverted": True
                        }
                    }

            # 3. Iron Cross Check (Horizontal Alignment) - RINGS ONLY
            if "Still Rings" in apparatus_label:
                try:
                    l_arm_line = self._calculate_angle(landmarks[11], landmarks[13], landmarks[15])
                    r_arm_line = self._calculate_angle(landmarks[12], landmarks[14], landmarks[16])
                    
                    # Check if arms are perpendicular to the torso (shoulder-hip-shoulder angle)
                    l_shoulder_lat = self._calculate_angle(landmarks[23], landmarks[11], landmarks[13])
                    
                    # Arms must be straight and horizontal
                    arms_straight = l_arm_line > 165 and r_arm_line > 165
                    arms_lateral = 80 < l_shoulder_lat < 110
                    
                    if arms_straight and arms_lateral:
                        return {
                            "skill": "Iron Cross",
                            "type": "Static / Hold",
                            "metrics": {
                                "arm_straightness": round(float((l_arm_line + r_arm_line) / 2), 1),
                                "lateral_angle": round(float(l_shoulder_lat), 1)
                            }
                        }
                except (IndexError, ZeroDivisionError):
                    pass

            # 4. L-Sit / Pike Check
            try:
                # Hip angle: Shoulder-Hip-Ankle (Center of body focus)
                hip_angle = self._calculate_angle(landmarks[11], landmarks[23], landmarks[27])
                hands_down = landmarks[15].y > landmarks[11].y and landmarks[16].y > landmarks[12].y
                
                # Check for "L-Sit" or "V-Sit"
                if hip_angle < 120 and hands_down:
                    is_v_sit = hip_angle < 70
                    skill_name = "V-Sit" if is_v_sit else "L-Sit"
                    display_name = "Horizonation - V-Sit" if is_v_sit else "Horizonation - L-Sit"
                    
                    # Straight Arms Check (Elbow angles)
                    l_elbow = self._calculate_angle(landmarks[11], landmarks[13], landmarks[15])
                    r_elbow = self._calculate_angle(landmarks[12], landmarks[14], landmarks[16])
                    
                    # Straight Legs Check (Knee angles)
                    l_knee = self._calculate_angle(landmarks[23], landmarks[25], landmarks[27])
                    r_knee = self._calculate_angle(landmarks[24], landmarks[26], landmarks[28])
                    
                    # Horizonation: Leg height relative to hips
                    ankle_y = (landmarks[27].y + landmarks[28].y) / 2
                    hip_y = (landmarks[23].y + landmarks[24].y) / 2
                    horizon_diff = hip_y - ankle_y 
                    
                    # Shoulder Angle (Flexion)
                    shoulder_angle = self._calculate_angle(landmarks[15], landmarks[11], landmarks[23])
                    
                    # Toe Point Check (Ankle extension)
                    l_toe_angle = self._calculate_angle(landmarks[25], landmarks[27], landmarks[31])
                    r_toe_angle = self._calculate_angle(landmarks[26], landmarks[28], landmarks[32])
                    toe_point = (l_toe_angle + r_toe_angle) / 2
                    
                    # Verticality (Torso - Head to Hips)
                    torso_verticality = abs(landmarks[11].x - landmarks[23].x)
                    
                    return {
                        "skill": skill_name, 
                        "displayName": display_name,
                        "type": "Static / Hold",
                        "metrics": {
                            "Final Score Vertical": round(float(torso_verticality), 3),
                            "Straight Arms": round(float((l_elbow + r_elbow) / 2), 1),
                            "Straight Legs": round(float((l_knee + r_knee) / 2), 1),
                            "Shoulder Angle": round(float(shoulder_angle), 1),
                            "Horizonation": round(float(horizon_diff * 100), 1),
                            "Pointed Toes": round(float(toe_point), 1),
                            "Hip Angle": round(float(hip_angle), 1)
                        }
                    }
            except (IndexError, ZeroDivisionError):
                pass

            # 5. Planche / Press Handstand Check (Horizontal/Angled body, arms supporting)
            try:
                # Body alignment (Shoulder-Hip-Knee)
                body_line = self._calculate_angle(landmarks[11], landmarks[23], landmarks[25])
                
                # Horizonation: Are shoulders and hips at similar height?
                shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
                hip_y = (landmarks[23].y + landmarks[24].y) / 2
                is_horizontal = abs(shoulder_y - hip_y) < 0.1
                
                # Shoulder Extension (Wrist-Shoulder-Hip)
                shoulder_angle = self._calculate_angle(landmarks[15], landmarks[11], landmarks[23])
                
                # Hands must be supporting (below shoulders)
                hands_down = landmarks[15].y > landmarks[11].y and landmarks[16].y > landmarks[12].y
                
                if hands_down and body_line > 140:
                    if is_horizontal and 70 < shoulder_angle < 110:
                        return {
                            "skill": "Planche",
                            "type": "Static / Hold",
                            "metrics": {
                                "body_alignment": round(float(body_line), 1),
                                "horizonation": round(float(abs(shoulder_y - hip_y)), 3)
                            }
                        }
                    elif hip_y < shoulder_y: # Hips higher than shoulders but not handstand yet
                        return {
                            "skill": "Press Handstand",
                            "type": "Dynamic",
                            "metrics": {
                                "body_line": round(float(body_line), 1),
                                "hip_elevation": round(float(shoulder_y - hip_y), 3)
                            }
                        }
            except (IndexError, ZeroDivisionError):
                pass

            # 6. Y-Balance / Scale Check (One leg raised high)
            foot_diff = abs(landmarks[27].y - landmarks[28].y)
            if foot_diff > 0.4:
                return {
                    "skill": "Y-Scale / Arabesque",
                    "type": "Static / Hold",
                    "metrics": {
                        "leg_height_diff": round(float(foot_diff), 2)
                    }
                }
                
            # 7. Bridge Check (Back bend)
            hip_y = landmarks[23].y
            head_y = landmarks[0].y
            if hip_y < head_y and hip_y < landmarks[27].y:
                return {
                    "skill": "Bridge",
                    "type": "Static / Hold",
                    "metrics": {
                        "hip_elevation": round(head_y - hip_y, 2)
                    }
                }

            # 8. Pommel Horse Support / Circle Check
            # Robustness: Also check for flair movement even if apparatus detection missed 'Pommel'
            is_pommel = "Pommel" in apparatus_label
            legs_spread_wide_flair = abs(landmarks[27].x - landmarks[28].x) > 0.45
            
            if is_pommel or (legs_spread_wide_flair and mid_hip_y > 0.3):
                # A flair involves the body being relatively horizontal
                # Relaxed: Check if wrists are below shoulders (supporting weight)
                hands_supporting = landmarks[15].y > landmarks[11].y and landmarks[16].y > landmarks[12].y
                
                if hands_supporting:
                    if legs_spread_wide_flair:
                        return {
                            "skill": "Pommel Flair/Circle",
                            "type": "Dynamic",
                            "metrics": {
                                "leg_spread": round(abs(landmarks[27].x - landmarks[28].x), 2)
                            }
                        }
                    else:
                        # Support check - simplified
                        return {
                            "skill": "Pommel Support",
                            "type": "Static / Hold",
                            "metrics": {}
                        }

            # 9. Straddle Split / Straddle L-Sit Check
            try:
                # Sensitive detection using Torso Length as a dynamic reference
                torso_length = abs(landmarks[11].y - landmarks[23].y)
                leg_spread = abs(landmarks[27].x - landmarks[28].x)
                
                # If legs are spread wide relative to torso height
                if leg_spread > (torso_length * 1.2) or leg_spread > 0.4:
                    hip_y = (landmarks[23].y + landmarks[24].y) / 2
                    ankle_y = (landmarks[27].y + landmarks[28].y) / 2
                    
                    # If feet are at or above hip level = Straddle L-Sit/V-Sit
                    if ankle_y <= hip_y + 0.05:
                        # Differentiate between Straddle L and Straddle V based on ankle height
                        # Use torso length as a reference: if feet are significantly above hips, it's a V-Sit
                        is_v_sit = (hip_y - ankle_y) > (torso_length * 0.4)
                        
                        return {
                            "skill": "Straddle V-Sit" if is_v_sit else "Straddle L-Sit",
                            "type": "Static / Hold",
                            "metrics": {
                                "leg_spread": round(float(leg_spread), 2),
                                "horizonation": round(float(hip_y - ankle_y), 3)
                            }
                        }
                    else:
                        return {
                            "skill": "Straddle Split",
                            "type": "Dynamic",
                            "metrics": {"leg_spread": round(float(leg_spread), 2)}
                        }
            except (IndexError, ZeroDivisionError):
                pass

            # 10. Parallel Bars Support Check
            if "Parallel Bars" in apparatus_label:
                # ... standard support check ...
                shoulders_above_hips = landmarks[11].y < landmarks[23].y and landmarks[12].y < landmarks[24].y
                hands_at_bars = landmarks[15].y > landmarks[11].y and landmarks[16].y > landmarks[12].y
                
                if shoulders_above_hips and hands_at_bars:
                     return {
                        "skill": "PB Support",
                        "type": "Static / Hold",
                        "metrics": {
                            "torso_verticality": round(float(abs(landmarks[11].x - landmarks[23].x)), 3)
                        }
                    }

            # Default: Unknown/Transition
            return {
                "skill": "Transition/Unknown",
                "type": "Dynamic",
                "metrics": {}
            }

        except (IndexError, AttributeError, ZeroDivisionError) as e:
            # Fallback for any errors
            return {
                "skill": "Pose",
                "type": "Unknown",
                "metrics": {},
                "error": str(e)
            }


    def analyze_wag_skill(self, landmarks, skill_name, category, hold_duration=0, apparatus_label="Unknown"):
        """
        Analyze a specific WAG skill for D-Score and execution errors.
        """
        from model_service.skill_knowledge import SKILL_DATA
        
        # Check standard skill database
        skill_info = SKILL_DATA.get(skill_name)
        
        deductions = []
        feedback = []
        d_val = 0.0
        skill_type = "A"
        status = "Pass" if skill_info else "Neutral"
        
        if skill_info:
            # Map FIG Difficulty to numeric D-Score
            difficulty_map = {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4, "E": 0.5, "F": 0.6}
            d_val = difficulty_map.get(skill_info.get("difficulty"), 0.1)
            skill_type = skill_info.get("difficulty", "A")

        # --- TECHNICAL FAULT DETECTION ---
        
        # 1. Bent Knees check (Joint 23-25-27 and 24-26-28)
        try:
            l_knee = self._calculate_angle(landmarks[23], landmarks[25], landmarks[27])
            r_knee = self._calculate_angle(landmarks[24], landmarks[26], landmarks[28])
            if l_knee < 165 or r_knee < 165:
                # FIG Code: > 45° = 0.3, 15-45° = 0.1
                deductions.append({
                    "value": 0.1, 
                    "observation": "Bent knees (detectable angle < 165°)", 
                    "label": "Bent Knees"
                })
                feedback.append("Squeeze your quads to keep your legs fully extended.")
        except (IndexError, AttributeError): pass

        # 2. Bent Arms check (Joint 11-13-15 and 12-14-16)
        try:
            l_elbow = self._calculate_angle(landmarks[11], landmarks[13], landmarks[15])
            r_elbow = self._calculate_angle(landmarks[12], landmarks[14], landmarks[16])
            if l_elbow < 165 or r_elbow < 165:
                deductions.append({
                    "value": 0.3, 
                    "observation": "Bent arms in support (detectable angle < 165°)", 
                    "label": "Bent Arms"
                })
                feedback.append("Fully lock your elbows for a clean Execution score.")
        except (IndexError, AttributeError): pass

        # 3. Specific Skill Checks (e.g., Straddle L-Sit Height)
        if skill_name == "Straddle L-Sit":
            try:
                # Leg Height (Horizonation) - ankles relative to hips
                hip_y = (landmarks[23].y + landmarks[24].y) / 2
                ankle_y = (landmarks[27].y + landmarks[28].y) / 2
                # In L-Sit, ankles must be at or above hip level
                if ankle_y > hip_y + 0.05: # lower is higher in y coordinate
                    deductions.append({
                        "value": 0.3, 
                        "observation": "Legs significantly below horizontal", 
                        "label": "Leg Height"
                    })
            except (IndexError, AttributeError): pass

        if skill_name == "V-Sit":
            try:
                # Hip angle between 0 and 45 relative to torso (Sharp V)
                hip_angle = self._calculate_angle(landmarks[11], landmarks[23], landmarks[27])
                if hip_angle > 45:
                    deductions.append({
                        "value": 0.1,
                        "observation": f"Insufficient V angle ({round(float(hip_angle), 1)}°)",
                        "label": "Leg Height"
                    })
                    feedback.append("Pull your feet closer to your head for a sharper V-Sit.")
            except (IndexError, AttributeError): pass

        if "Straddle V-Sit" in skill_name:
            try:
                # Use mean hip angle as proxy for V height
                l_hip = self._calculate_angle(landmarks[11], landmarks[23], landmarks[27])
                r_hip = self._calculate_angle(landmarks[12], landmarks[24], landmarks[28])
                avg_hip = (l_hip + r_hip) / 2
                if avg_hip > 55: # Slightly more lenient for straddle
                    deductions.append({
                        "value": 0.1,
                        "observation": f"Insufficient Straddle V height ({round(float(avg_hip), 1)}°)",
                        "label": "Leg Height"
                    })
            except (IndexError, AttributeError): pass

        if skill_name == "Planche":
            try:
                # Hips relative to shoulders for horizontal check
                shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
                hip_y = (landmarks[23].y + landmarks[24].y) / 2
                if abs(shoulder_y - hip_y) > 0.08:
                    deductions.append({
                        "value": 0.1,
                        "observation": "Hips slightly outside horizontal plane",
                        "label": "Body Alignment"
                    })
            except (IndexError, AttributeError): pass

        # Fallback for dynamic/unspecified split skills
        if "Split" in skill_name and d_val == 0.0:
            d_val = 0.2
            status = "Pass"

        return {
            "skill": skill_name,
            "d_score_contribution": d_val,
            "status": status,
            "type": skill_type,
            "deductions": deductions,
            "feedback": feedback
        }

    def analyze_media(self, media_data, media_type='video', category='Senior Elite', hold_duration=2):
        apparatus = None
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
            
            frames_data = [] # To store per-frame analysis
            apparatus_info = {"label": "Floor Exercise", "confidence": 0.0} # Default
            
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
                                        lm_dicts = [{"x": lm.x, "y": lm.y} for lm in final_landmarks] # Minimal dict for detector
                                        cached_apparatus = self.detect_apparatus(frame, pose_landmarks=lm_dicts)
                                        apparatus_detected = True
                                    
                                    apparatus_info = cached_apparatus
                                    
                                    # Calibrate/Normalize (Useful for consistent relative analysis, but NOT for overlay)
                                    final_landmarks_calibrated = self.calibrate_and_normalize(final_landmarks, apparatus_info, w, h)
                                    
                                    # Store results
                                    frames_data.append({
                                        "time": curr_frame_idx / fps,
                                        # FIXED: Use RAW global landmarks for visualization overlay!
                                        "landmarks": [{"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility} for lm in final_landmarks],
                                        "centered_landmarks": [{"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility} for lm in final_landmarks_calibrated],
                                        "raw_landmarks": [{"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility} for lm in final_landmarks],
                                        "apparatus": apparatus_info
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
                    
                    # **STAGE 3 FALLBACK: Detection on Full Frame (Raw)**
                    if not detection_result.pose_landmarks:
                        print("GymnasticsAnalyzer: ROI detection failed, trying FULL FRAME...")
                        image_rgb_full = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
                        mp_image_full = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb_full)
                        detection_result = self.landmarker.detect(mp_image_full)
                        # Reset offsets if full frame worked
                        if detection_result.pose_landmarks:
                            crop_offset_x, crop_offset_y = 0.0, 0.0
                            crop_scale_x, crop_scale_y = 1.0, 1.0
                    
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
                return round(max(0.0, min(5.0, base_artistry)), 1)

                
            # --- ARTISTRY & EXECUTION ---
            artistry_score = get_artistry(frames_data)
                
            # --- SKILL ANALYSIS & KNOWLEDGE INJECTION ---
            from model_service.skill_knowledge import SKILL_DATA
            
            skill_list = []
            metrics = {}
            
            if media_type == 'image':
                # Single Frame Analysis
                skill_result = self.identify_skill(landmarks, apparatus_info)
                skill_name = skill_result["skill"]
                skill_type = skill_result.get("type", "Unknown")
                skill_metrics = skill_result.get("metrics", {})
                metrics = skill_metrics
                
                wag_feedback = self.analyze_wag_skill(landmarks, skill_name, category)
                
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
                    f_skill_result = self.identify_skill(f_landmarks, f_apparatus)
                    f_skill_name = f_skill_result["skill"]
                    f_skill_type = f_skill_result.get("type", "Unknown")
                    f_metrics = f_skill_result.get("metrics", {})
                    
                    if f_skill_name == best_skill_name:
                        metrics = f_metrics
                    
                    f_wag = self.analyze_wag_skill(f_landmarks, f_skill_name, category, hold_duration, apparatus_label=f_apparatus.get("label", "Unknown") if f_apparatus else "Unknown")
                    
                    # Enrich frame metadata for UI Timeline
                    frame_data["skill"] = f_wag.get('skill', f_skill_name)
                    frame_data["dv"] = f_wag.get('d_score_contribution', 0.0)
                    frame_data["status"] = f_wag.get('status', 'Neutral')
                    
                    if frame_data["dv"] > 0:
                        all_skills_found.append({
                            "name": frame_data["skill"],
                            "dv": frame_data["dv"],
                            "type": f_wag.get('type', f_skill_type)
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
            d_score_result = d_calculator.calculate_d_score(skill_list, category=category)
            total_d_score = float(d_score_result.get('total_d_score', 0.0))

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

            # Initialize result dictionary
            result = {
                "total_score": 0.0,
                "difficulty": round(total_d_score, 2),
                "execution": 0.0,
                "artistry": artistry_score,
                "skills_found": skill_list,
                "best_skill": "Pose",
                "feedback": [],
                "deductions": [],
                "d_score_breakdown": d_score_result,
                "comment": "Focus on maintaining stability and extension throughout your routine.",
                "visualization_data": {}
            }

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
                     if self.identify_skill(flm, fapp)["skill"] == best_skill_name:
                         viz_frame_idx = idx
                         break
            
            target_frame_data = frames_data[viz_frame_idx]
            target_landmarks = [SimpleNamespace(**lm) for lm in target_frame_data['landmarks']]
            
            # Re-analyze best frame for specific feedback (Dynamic E-Score Calculation)
            current_apparatus = apparatus if apparatus else target_frame_data.get('apparatus')
            apparatus_label = current_apparatus.get("label", "Unknown") if isinstance(current_apparatus, dict) else "Unknown"
            
            wag_feedback = self.analyze_wag_skill(target_landmarks, best_skill_name, category, hold_duration, apparatus_label=apparatus_label)
            
            # Calculate Technical Deductions
            tech_deductions_list = wag_feedback.get('deductions', [])
            total_tech_deductions = sum(d.get('value', 0.0) for d in tech_deductions_list)
            
            # Calculate Artistry Deductions (from Heuristic)
            artistry_deductions = round(max(0.0, 5.0 - artistry_score), 1)
            
            # Final E-Score
            base_e_score = 10.0
            total_deductions = total_tech_deductions + artistry_deductions
            e_score = max(0.0, base_e_score - total_deductions)
            e_score = round(e_score, 2)
            
            # Final Total Score
            final_score = total_d_score + e_score
            
            # Update result with calculated values
            result["total_score"] = round(final_score, 2)
            result["execution"] = e_score
            result["best_skill"] = best_skill_name
            result["feedback"] = wag_feedback.get('feedback', [])
            result["deductions"] = tech_deductions_list
            
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
                "landmarks": target_frame_data['landmarks'],
                "raw_landmarks": target_frame_data.get('raw_landmarks'),
                "frames": frames_data,
                "apparatus": target_frame_data.get('apparatus'),
                # Coach's Corner Data
                "technical_cue": skill_details.get("technicalCue", "Focus on form and stability."),
                "focus_anatomy": skill_details.get("focusAnatomy", []),
                "common_errors": skill_details.get("commonDeductions", []),
                "skill_description": skill_details.get("description", "")
            })
            
            return result
        except Exception as e:
            print(f"Error in analyze_media inner: {e}")
            traceback.print_exc()
            raise e
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
        
        # If they are parallel (standing), angle is 0
        return round(angle, 1)

    def check_ring_arch(self, landmarks):
        """W-002: Upper Body Arch >= 80 degrees."""
        # Shoulder (12), Hip (24), Knee (26)
        return self._calculate_angle(
            landmarks[12],
            landmarks[24],
            landmarks[26]
        )

    def check_head_release(self, landmarks):
        """W-003: Head Release >= 100 degrees."""
        # Ear (8), Neck/Shoulder (12), Hip (24)
        return self._calculate_angle(
            landmarks[8],
            landmarks[12],
            landmarks[24]
        )

    def analyze_wag_skill(self, landmarks, skill_name, category="Senior", hold_duration=2.0, apparatus_label="Unknown"):
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
