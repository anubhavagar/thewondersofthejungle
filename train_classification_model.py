# -*- coding: utf-8 -*-
"""
Gymnastics Apparatus Image Classification Model Training
Uses TensorFlow/Keras with transfer learning (MobileNetV2) for multi-class classification.

Installation:
    pip install tensorflow pillow matplotlib

Usage:
    python train_classification_model.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = "gym_data/raw_images"
MODEL_NAME = "apparatus_classifier.h5"
TFLITE_MODEL_NAME = "apparatus_classifier.tflite"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
VALIDATION_SPLIT = 0.2

# Class names (will be auto-detected from folder names)
APPARATUS_CLASSES = [
    "balance_beam",
    "floor_exercise", 
    "horizontal_bar",
    "parallel_bars",
    "pommel_horse",
    "still_rings",
    "uneven_bars",
    "vault"
]


def create_model(num_classes):
    """
    Create a transfer learning model using MobileNetV2.
    
    Args:
        num_classes: Number of apparatus classes
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained MobileNetV2 (without top classification layer)
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add custom classification head
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def plot_training_history(history, save_path='training_history.png'):
    """Plot and save training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n[OK] Training history saved to {save_path}")


def main():
    """Main training function."""
    print("="*60)
    print("Gymnastics Apparatus Classification Model Training")
    print("="*60)
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"\n[ERROR] Data directory not found: {DATA_DIR}")
        return
    
    # Count images per class
    print(f"\nData Directory: {DATA_DIR}")
    print("\nClass Distribution:")
    total_images = 0
    for class_name in sorted(os.listdir(DATA_DIR)):
        class_path = os.path.join(DATA_DIR, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  {class_name:20s}: {count:3d} images")
            total_images += count
    
    print(f"\nTotal Images: {total_images}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Validation Split: {VALIDATION_SPLIT*100:.0f}%")
    
    # Confirm before starting
    response = input("\n>> Start training? (y/n): ").strip().lower()
    if response != 'y':
        print("Training cancelled.")
        return
    
    print("\n" + "="*60)
    print("[1/5] Setting up data generators...")
    print("="*60)
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=VALIDATION_SPLIT
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=VALIDATION_SPLIT
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    val_generator = val_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    num_classes = len(train_generator.class_indices)
    print(f"\n[OK] Found {num_classes} classes")
    print(f"[OK] Training samples: {train_generator.samples}")
    print(f"[OK] Validation samples: {val_generator.samples}")
    
    print("\n" + "="*60)
    print("[2/5] Creating model...")
    print("="*60)
    
    model = create_model(num_classes)
    model.summary()
    
    print("\n" + "="*60)
    print("[3/5] Training model...")
    print("="*60)
    print("\nThis may take 10-20 minutes depending on your hardware...")
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        verbose=1
    )
    
    print("\n" + "="*60)
    print("[4/5] Evaluating model...")
    print("="*60)
    
    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
    print(f"\n[OK] Validation Accuracy: {val_accuracy*100:.2f}%")
    print(f"[OK] Validation Loss: {val_loss:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    print("\n" + "="*60)
    print("[5/5] Saving models...")
    print("="*60)
    
    # Save Keras model
    model.save(MODEL_NAME)
    print(f"\n[OK] Keras model saved: {MODEL_NAME}")
    
    # Convert to TFLite with compatible settings
    print("\nConverting to TFLite (this may take a moment)...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Use experimental settings to avoid MLIR errors
        converter.experimental_new_converter = False
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()
        
        with open(TFLITE_MODEL_NAME, 'wb') as f:
            f.write(tflite_model)
        
        tflite_size_mb = os.path.getsize(TFLITE_MODEL_NAME) / (1024 * 1024)
        print(f"[OK] TFLite model saved: {TFLITE_MODEL_NAME} ({tflite_size_mb:.2f} MB)")
    except Exception as e:
        print(f"[WARNING] TFLite conversion failed: {e}")
        print("[INFO] You can still use the Keras model (.h5) for predictions")
        print("[INFO] TFLite is only needed for mobile deployment")
    
    # Save class mapping
    class_mapping = {v: k for k, v in train_generator.class_indices.items()}
    with open('class_mapping.txt', 'w') as f:
        for idx, class_name in sorted(class_mapping.items()):
            f.write(f"{idx}: {class_name}\n")
    print(f"[OK] Class mapping saved: class_mapping.txt")
    
    print("\n" + "="*60)
    print("[SUCCESS] Training Complete!")
    print("="*60)
    print(f"\nFinal Validation Accuracy: {val_accuracy*100:.2f}%")
    print(f"\nGenerated Files:")
    print(f"  1. {MODEL_NAME} - Full Keras model")
    print(f"  2. {TFLITE_MODEL_NAME} - Optimized TFLite model")
    print(f"  3. class_mapping.txt - Class index mapping")
    print(f"  4. training_history.png - Training curves")
    
    print("\nNext Steps:")
    print(f"  1. Review training_history.png for overfitting")
    print(f"  2. Test model with: python test_classifier.py")
    print(f"  3. Integrate into gymnastics.py if accuracy is good")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Training interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
