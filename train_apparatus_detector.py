"""
Enhanced Training Script for Custom Gymnastics Apparatus Detection
Uses MediaPipe Model Maker to train a TFLite object detection model
"""

import os
import tensorflow as tf
from mediapipe_model_maker import object_detector

# ==================== CONFIGURATION ====================
TRAIN_DATA_DIR = "gym_data/train"
VALIDATION_DATA_DIR = "gym_data/val"
OUTPUT_MODEL_NAME = "gym_apparatus_custom.tflite"

# Apparatus classes (must match your XML annotation labels)
APPARATUS_CLASSES = [
    "Pommel_Horse",      # PH
    "Still_Rings",       # SR
    "Vault",             # VT
    "Parallel_Bars",     # PB
    "Horizontal_Bar",    # HB
    "Uneven_Bars",       # UB
    "Balance_Beam",      # BB
    "Floor_Exercise"     # FX (mat/boundary)
]

# ==================== TRAINING FUNCTION ====================
def train_gym_model():
    """Train custom apparatus detection model using transfer learning."""
    
    print("=" * 60)
    print("Training Custom Gymnastics Apparatus Detector")
    print("=" * 60)
    print(f"\nTarget Classes: {', '.join(APPARATUS_CLASSES)}")
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
            epochs=100,              # Increased for better convergence
            batch_size=8,            # Adjust based on GPU memory (4, 8, 16)
            learning_rate=0.01,      # Lower for fine-tuning (0.01-0.05)
            export_archive_format=object_detector.HParams.export_archive_format.TFLITE
        )
    )
    
    print(f"✓ Model Architecture: MobileNet V2")
    print(f"✓ Training Config: {options.hparams.epochs} epochs, batch size {options.hparams.batch_size}")
    print(f"✓ Learning Rate: {options.hparams.learning_rate}")

    # 4. Train Model (Transfer Learning)
    print("\n[4/6] Starting training...")
    print("⏳ This may take 30-60 minutes depending on dataset size and hardware")
    print("-" * 60)
    
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
    print("\n[5/6] Evaluating model on validation set...")
    try:
        loss, coco_metrics = model.evaluate(validation_data, batch_size=8)
        print(f"✓ Validation Loss: {loss:.4f}")
        print(f"✓ COCO Metrics:")
        for metric_name, metric_value in coco_metrics.items():
            print(f"  - {metric_name}: {metric_value:.4f}")
    except Exception as e:
        print(f"⚠️  WARNING: Evaluation failed: {e}")

    # 6. Export TFLite Model
    print(f"\n[6/6] Exporting model to {OUTPUT_MODEL_NAME}...")
    try:
        model.export_model(OUTPUT_MODEL_NAME)
        
        # Check file size
        file_size_mb = os.path.getsize(OUTPUT_MODEL_NAME) / (1024 * 1024)
        print(f"✓ Model exported successfully!")
        print(f"✓ File size: {file_size_mb:.2f} MB")
    except Exception as e:
        print(f"❌ ERROR during export: {e}")
        return
    
    # Success Summary
    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print("=" * 60)
    print(f"\n📦 Model Location: {os.path.abspath(OUTPUT_MODEL_NAME)}")
    print(f"📊 Model Size: {file_size_mb:.2f} MB")
    print("\n📋 Next Steps:")
    print(f"  1. Copy {OUTPUT_MODEL_NAME} to model_service/models/")
    print(f"  2. Update gymnastics.py to use the custom model")
    print(f"  3. Test with sample gymnastics images")
    print(f"  4. Compare accuracy with generic EfficientDet model")
    print("\n" + "=" * 60)

# ==================== VALIDATION CHECKS ====================
def validate_setup():
    """Validate that data directories exist and contain required files."""
    
    errors = []
    
    # Check training directory
    if not os.path.exists(TRAIN_DATA_DIR):
        errors.append(f"Training directory not found: {TRAIN_DATA_DIR}")
    else:
        train_images = [f for f in os.listdir(TRAIN_DATA_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
        train_xmls = [f for f in os.listdir(TRAIN_DATA_DIR) if f.endswith('.xml')]
        
        if len(train_images) == 0:
            errors.append(f"No images found in {TRAIN_DATA_DIR}")
        if len(train_xmls) == 0:
            errors.append(f"No XML annotations found in {TRAIN_DATA_DIR}")
        
        print(f"✓ Training data: {len(train_images)} images, {len(train_xmls)} annotations")
    
    # Check validation directory
    if not os.path.exists(VALIDATION_DATA_DIR):
        errors.append(f"Validation directory not found: {VALIDATION_DATA_DIR}")
    else:
        val_images = [f for f in os.listdir(VALIDATION_DATA_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
        val_xmls = [f for f in os.listdir(VALIDATION_DATA_DIR) if f.endswith('.xml')]
        
        if len(val_images) == 0:
            errors.append(f"No images found in {VALIDATION_DATA_DIR}")
        if len(val_xmls) == 0:
            errors.append(f"No XML annotations found in {VALIDATION_DATA_DIR}")
        
        print(f"✓ Validation data: {len(val_images)} images, {len(val_xmls)} annotations")
    
    return errors

# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n🔍 Validating setup...")
    print("-" * 60)
    
    validation_errors = validate_setup()
    
    if validation_errors:
        print("\n❌ Setup validation failed:")
        for error in validation_errors:
            print(f"  - {error}")
        print("\n💡 Tips:")
        print("  1. Create directories: gym_data/train/ and gym_data/val/")
        print("  2. Add annotated images (.jpg + .xml pairs)")
        print("  3. Use LabelImg to create Pascal VOC annotations")
        print("  4. Ensure class names match APPARATUS_CLASSES list")
        exit(1)
    
    print("\n✓ Setup validation passed!")
    print("-" * 60)
    
    # Confirm before starting
    response = input("\n▶️  Start training? (y/n): ").strip().lower()
    if response != 'y':
        print("Training cancelled.")
        exit(0)
    
    # Start training
    train_gym_model()
