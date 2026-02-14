"""
Training Script for Custom Gymnastics Skill Detection
Uses MediaPipe Model Maker to train a TFLite object detection model
"""

import os
import tensorflow as tf
from mediapipe_model_maker import object_detector

# ==================== CONFIGURATION ====================
TRAIN_DATA_DIR = "gym_data/skill_detect_dataset/train"
VALIDATION_DATA_DIR = "gym_data/skill_detect_dataset/valid"
OUTPUT_MODEL_NAME = "gym_skill_detector.tflite"

# Skill classes (identified from dataset XMLs)
SKILL_CLASSES = [
    "BL",         # Back Lever
    "FL",         # Front Lever
    "HS",         # Handstand
    "IN-IRON-C",  # Inverted Iron Cross
    "IRON-C",     # Iron Cross
    "L-CROSS",    # L-Cross
    "LS",         # L-Sit
    "M-UP",       # Muscle Up
    "PN",         # Planche
    "VS"          # V-Sit
]

# ==================== TRAINING FUNCTION ====================
def train_skill_model():
    """Train custom skill detection model using transfer learning."""
    
    print("=" * 60)
    print("Training Custom Gymnastics Skill Detector")
    print("=" * 60)
    print(f"\nTarget Classes: {', '.join(SKILL_CLASSES)}")
    print(f"Output Model: {OUTPUT_MODEL_NAME}\n")
    
    # 1. Load Training Data
    print("[1/6] Loading training data...")
    try:
        train_data = object_detector.Dataset.from_pascal_voc_folder(TRAIN_DATA_DIR)
        print(f"✓ Loaded {len(train_data)} training images")
    except Exception as e:
        print(f"❌ ERROR loading training data: {e}")
        return
    
    # 2. Load Validation Data
    print("\n[2/6] Loading validation data...")
    try:
        validation_data = object_detector.Dataset.from_pascal_voc_folder(VALIDATION_DATA_DIR)
        print(f"✓ Loaded {len(validation_data)} validation images")
    except Exception as e:
        print(f"❌ ERROR loading validation data: {e}")
        return

    # 3. Configure Model
    print("\n[3/6] Configuring model...")
    spec = object_detector.SupportedModels.MOBILENET_V2
    
    options = object_detector.ObjectDetectorOptions(
        supported_model=spec,
        hparams=object_detector.HParams(
            epochs=100,              # Consistent with apparatus training
            batch_size=8,            # Adjust based on memory
            learning_rate=0.01,
            export_archive_format=object_detector.HParams.export_archive_format.TFLITE
        )
    )
    
    print(f"✓ Model Architecture: MobileNet V2")
    print(f"✓ Training Config: {options.hparams.epochs} epochs, batch size {options.hparams.batch_size}")

    # 4. Train Model
    print("\n[4/6] Starting training...")
    try:
        model = object_detector.ObjectDetector.create(
            train_data=train_data,
            validation_data=validation_data,
            options=options
        )
        print("\n✓ Training complete!")
    except Exception as e:
        print(f"\n❌ ERROR during training: {e}")
        return

    # 5. Evaluate Model
    print("\n[5/6] Evaluating model...")
    try:
        loss, coco_metrics = model.evaluate(validation_data, batch_size=8)
        print(f"✓ Validation Loss: {loss:.4f}")
        for metric_name, metric_value in coco_metrics.items():
            print(f"  - {metric_name}: {metric_value:.4f}")
    except Exception as e:
        print(f"⚠️  WARNING: Evaluation failed: {e}")

    # 6. Export TFLite Model
    print(f"\n[6/6] Exporting model to {OUTPUT_MODEL_NAME}...")
    try:
        model.export_model(OUTPUT_MODEL_NAME)
        file_size_mb = os.path.getsize(OUTPUT_MODEL_NAME) / (1024 * 1024)
        print(f"✓ Model exported successfully!")
        print(f"✓ File size: {file_size_mb:.2f} MB")
    except Exception as e:
        print(f"❌ ERROR during export: {e}")
        return
    
    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print(f"📦 Model Location: {os.path.abspath(OUTPUT_MODEL_NAME)}")
    print("=" * 60)

# ==================== VALIDATION CHECKS ====================
def validate_setup():
    """Validate data directories."""
    errors = []
    for directory in [TRAIN_DATA_DIR, VALIDATION_DATA_DIR]:
        if not os.path.exists(directory):
            errors.append(f"Directory not found: {directory}")
        else:
            images = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.jpeg', '.png'))]
            xmls = [f for f in os.listdir(directory) if f.endswith('.xml')]
            if len(images) == 0 or len(xmls) == 0:
                errors.append(f"No valid data in {directory}")
            else:
                print(f"✓ {directory}: {len(images)} images, {len(xmls)} annotations")
    return errors

if __name__ == "__main__":
    v_errors = validate_setup()
    if v_errors:
        for err in v_errors: print(f"❌ {err}")
        exit(1)
    
    response = input("\n▶️  Start training? (y/n): ").strip().lower()
    if response == 'y':
        train_skill_model()
    else:
        print("Training cancelled.")
