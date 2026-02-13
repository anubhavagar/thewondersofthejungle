# -*- coding: utf-8 -*-
"""
Manual TFLite Converter for Apparatus Classification Model
Use this if the automatic conversion in train_classification_model.py fails.

Usage:
    python convert_to_tflite.py
"""

import os
import tensorflow as tf
from tensorflow import keras

# Configuration
KERAS_MODEL = "apparatus_classifier.h5"
TFLITE_OUTPUT = "apparatus_classifier.tflite"

def convert_model():
    """Convert Keras model to TFLite with fallback methods."""
    print("="*60)
    print("TFLite Model Converter")
    print("="*60)
    
    if not os.path.exists(KERAS_MODEL):
        print(f"\n[ERROR] Keras model not found: {KERAS_MODEL}")
        print("Please train the model first: python train_classification_model.py")
        return
    
    print(f"\nLoading Keras model: {KERAS_MODEL}")
    model = keras.models.load_model(KERAS_MODEL)
    print("[OK] Model loaded successfully")
    
    # Method 1: Try with experimental settings disabled
    print("\n[Method 1] Trying conversion with legacy converter...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_new_converter = False
        converter._experimental_lower_tensor_list_ops = False
        
        tflite_model = converter.convert()
        
        with open(TFLITE_OUTPUT, 'wb') as f:
            f.write(tflite_model)
        
        size_mb = os.path.getsize(TFLITE_OUTPUT) / (1024 * 1024)
        print(f"[SUCCESS] TFLite model saved: {TFLITE_OUTPUT} ({size_mb:.2f} MB)")
        return
    except Exception as e:
        print(f"[FAILED] Method 1 failed: {e}")
    
    # Method 2: Try without optimizations
    print("\n[Method 2] Trying conversion without optimizations...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.experimental_new_converter = False
        
        tflite_model = converter.convert()
        
        with open(TFLITE_OUTPUT, 'wb') as f:
            f.write(tflite_model)
        
        size_mb = os.path.getsize(TFLITE_OUTPUT) / (1024 * 1024)
        print(f"[SUCCESS] TFLite model saved: {TFLITE_OUTPUT} ({size_mb:.2f} MB)")
        print("[NOTE] Model is not optimized - larger file size")
        return
    except Exception as e:
        print(f"[FAILED] Method 2 failed: {e}")
    
    # Method 3: Save as SavedModel first, then convert
    print("\n[Method 3] Trying SavedModel intermediate conversion...")
    try:
        saved_model_dir = "saved_model_temp"
        model.export(saved_model_dir)
        
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        with open(TFLITE_OUTPUT, 'wb') as f:
            f.write(tflite_model)
        
        # Cleanup
        import shutil
        shutil.rmtree(saved_model_dir)
        
        size_mb = os.path.getsize(TFLITE_OUTPUT) / (1024 * 1024)
        print(f"[SUCCESS] TFLite model saved: {TFLITE_OUTPUT} ({size_mb:.2f} MB)")
        return
    except Exception as e:
        print(f"[FAILED] Method 3 failed: {e}")
    
    print("\n" + "="*60)
    print("[ERROR] All conversion methods failed")
    print("="*60)
    print("\nYou can still use the Keras model (.h5) for predictions.")
    print("TFLite is only needed for mobile/edge deployment.")
    print("\nAlternative: Use ONNX format instead")
    print("  pip install tf2onnx")
    print("  python -m tf2onnx.convert --keras apparatus_classifier.h5 --output model.onnx")


if __name__ == "__main__":
    convert_model()
