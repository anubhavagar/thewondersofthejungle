"""
Alternative: TensorFlow Lite Model Maker for Object Detection
Compatible with Python 3.14+

This script uses TensorFlow Lite Model Maker instead of MediaPipe Model Maker
for better Python version compatibility.
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Configuration
class Config:
    # Paths
    RAW_DATA_DIR = Path(r"C:\Users\ur_an\projects\let_me_check_your_face_app\gym_data\raw_object_detect_pascalvoc")
    OUTPUT_DIR = Path(r"C:\Users\ur_an\projects\let_me_check_your_face_app\gym_data\tflite_dataset")
    MODEL_OUTPUT_DIR = Path(r"C:\Users\ur_an\projects\let_me_check_your_face_app\model_service\models")
    
    # Training parameters
    EPOCHS = 50
    BATCH_SIZE = 8
    
    # Data split
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Classes
    CLASSES = [
        "Balance_Beam",
        "Horizontal_Bar",
        "Parallel_Bars",
        "Pommel_Horse",
        "Still_Rings",
        "Uneven_Bars",
        "Vault"
    ]
    
    MODEL_SPEC = "efficientdet_lite2"


def parse_pascal_voc(xml_file):
    """Parse Pascal VOC XML annotation."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    filename = root.find('filename').text
    
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        objects.append({
            'class': name,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    
    return {
        'filename': filename,
        'width': width,
        'height': height,
        'objects': objects
    }


def convert_to_coco_format(annotations_list, class_mapping, split_name):
    """Convert to COCO format."""
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    for class_name, class_id in class_mapping.items():
        coco_data["categories"].append({
            "id": class_id + 1,
            "name": class_name,
            "supercategory": "apparatus"
        })
    
    annotation_id = 1
    
    for image_id, annotation in enumerate(annotations_list, start=1):
        coco_data["images"].append({
            "id": image_id,
            "file_name": annotation['filename'],
            "width": annotation['width'],
            "height": annotation['height']
        })
        
        for obj in annotation['objects']:
            class_name = obj['class']
            if class_name not in class_mapping:
                continue
            
            xmin, ymin, xmax, ymax = obj['bbox']
            width = xmax - xmin
            height = ymax - ymin
            
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_mapping[class_name] + 1,
                "bbox": [xmin, ymin, width, height],
                "area": width * height,
                "iscrowd": 0
            })
            annotation_id += 1
    
    return coco_data


def prepare_dataset(config):
    """Prepare COCO format dataset."""
    print("=" * 80)
    print("PREPARING DATASET (COCO FORMAT)")
    print("=" * 80)
    
    for split in ['train', 'val', 'test']:
        (config.OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)
    
    class_mapping = {name: idx for idx, name in enumerate(config.CLASSES)}
    
    image_files = list(config.RAW_DATA_DIR.glob("*.jpg"))
    print(f"\nFound {len(image_files)} images")
    
    print("\nParsing annotations...")
    all_annotations = []
    valid_image_files = []
    
    for img_file in tqdm(image_files, desc="Parsing"):
        xml_file = img_file.with_suffix('.xml')
        
        if not xml_file.exists():
            continue
        
        try:
            annotation = parse_pascal_voc(xml_file)
            if annotation['objects']:
                all_annotations.append(annotation)
                valid_image_files.append(img_file)
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    print(f"Valid images: {len(valid_image_files)}")
    
    train_idx, temp_idx = train_test_split(
        range(len(valid_image_files)), 
        train_size=config.TRAIN_SPLIT, 
        random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=config.VAL_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT),
        random_state=42
    )
    
    splits = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx
    }
    
    print(f"\nTrain: {len(train_idx)}")
    print(f"Val: {len(val_idx)}")
    print(f"Test: {len(test_idx)}")
    
    split_paths = {}
    
    for split_name, indices in splits.items():
        print(f"\nProcessing {split_name}...")
        
        split_annotations = [all_annotations[i] for i in indices]
        split_images = [valid_image_files[i] for i in indices]
        
        coco_data = convert_to_coco_format(split_annotations, class_mapping, split_name)
        
        for img_file in tqdm(split_images, desc=f"Copying {split_name}"):
            dst_img = config.OUTPUT_DIR / split_name / img_file.name
            shutil.copy(img_file, dst_img)
        
        coco_json_path = config.OUTPUT_DIR / split_name / "annotations.json"
        with open(coco_json_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        split_paths[split_name] = {
            'images': str(config.OUTPUT_DIR / split_name),
            'annotations': str(coco_json_path)
        }
        
        print(f"{split_name.upper()}: {len(coco_data['images'])} images, {len(coco_data['annotations'])} objects")
    
    return split_paths


def train_model(config, split_paths):
    """Train TFLite model."""
    print("\n" + "=" * 80)
    print("TRAINING TFLITE OBJECT DETECTION MODEL")
    print("=" * 80)
    
    try:
        import tensorflow as tf
        from tflite_model_maker import object_detector
        from tflite_model_maker import model_spec
    except ImportError as e:
        print(f"\nERROR: TFLite Model Maker not installed")
        print(f"Details: {e}")
        print("\nTrying alternative installation...")
        
        # Try installing
        import subprocess
        subprocess.check_call([
            "pip", "install", 
            "tflite-model-maker",
            "pycocotools"
        ])
        
        # Retry import
        from tflite_model_maker import object_detector
        from tflite_model_maker import model_spec
    
    print(f"\nModel: {config.MODEL_SPEC}")
    print(f"Epochs: {config.EPOCHS}")
    
    # Load data
    print("\nLoading training data...")
    train_data = object_detector.DataLoader.from_pascal_voc(
        split_paths['train']['images'],
        split_paths['train']['images'],
        label_map={i: name for i, name in enumerate(config.CLASSES)}
    )
    
    print("Loading validation data...")
    val_data = object_detector.DataLoader.from_pascal_voc(
        split_paths['val']['images'],
        split_paths['val']['images'],
        label_map={i: name for i, name in enumerate(config.CLASSES)}
    )
    
    # Create model
    spec = model_spec.get(config.MODEL_SPEC)
    
    # Train
    print("\nTraining...")
    model = object_detector.create(
        train_data,
        model_spec=spec,
        batch_size=config.BATCH_SIZE,
        train_whole_model=True,
        epochs=config.EPOCHS,
        validation_data=val_data
    )
    
    # Evaluate
    print("\nEvaluating...")
    metrics = model.evaluate(val_data)
    print(f"Metrics: {metrics}")
    
    # Export
    export_dir = config.MODEL_OUTPUT_DIR / 'apparatus_detector_tflite'
    export_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = export_dir / 'gym_apparatus_detector.tflite'
    model.export(export_dir=str(export_dir), tflite_filename='gym_apparatus_detector.tflite')
    
    print(f"\nModel exported to: {model_path}")
    
    # Save labels
    label_path = export_dir / 'labels.txt'
    with open(label_path, 'w') as f:
        for class_name in config.CLASSES:
            f.write(f"{class_name}\n")
    
    return model_path


def main():
    """Main pipeline."""
    config = Config()
    
    print("\n" + "=" * 80)
    print("TFLITE APPARATUS DETECTION TRAINING")
    print("=" * 80)
    
    # Prepare dataset
    split_paths = prepare_dataset(config)
    
    # Train model
    try:
        model_path = train_model(config, split_paths)
        print("\n✅ Training complete!")
        print(f"Model: {model_path}")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("\nPlease install Python 3.8-3.11 for MediaPipe Model Maker")
        print("Or use Google Colab for training (see guide)")


if __name__ == "__main__":
    main()
