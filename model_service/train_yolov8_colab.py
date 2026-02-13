# -*- coding: utf-8 -*-
"""
🤸 Gymnastics Apparatus Detection - YOLOv8 Training (Local Terminal Version)

This script trains a YOLOv8 object detection model and exports to TFLite for MediaPipe integration.
Optimized for local execution with GPU support.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import yaml
from tqdm import tqdm
from ultralytics import YOLO
import torch
import zipfile
import json
from datetime import datetime

# ==================== CONFIGURATION ====================
# Paths
BASE_DIR = Path(r"C:\Users\ur_an\projects\let_me_check_your_face_app")
DATASET_ZIP = BASE_DIR / "gym_data" / "raw_object_detect_pascalvoc.zip"
TRAIN_ROOT = BASE_DIR / "model_service" / "yolo_training"
RAW_DATA_DIR = TRAIN_ROOT / "raw_data"
YOLO_DATASET_DIR = TRAIN_ROOT / "dataset"
RUNS_DIR = TRAIN_ROOT / "runs"

# Training parameters
EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = 640

# Data split
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1

# Classes (alphabetically sorted for YOLO)
CLASSES = [
    "Balance_Beam",
    "Horizontal_Bar",
    "Parallel_Bars",
    "Pommel_Horse",
    "Still_Rings",
    "Uneven_Bars",
    "Vault"
]

def setup_environment():
    """Check GPU and create directories."""
    print("🤸 YOLOv8 Local Training Setup")
    print("="*40)
    
    print(f"🎮 GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("   ⚠️  No GPU found, using CPU (will be slow)")
        device = 'cpu'
        
    # Create directories
    for d in [TRAIN_ROOT, RAW_DATA_DIR, YOLO_DATASET_DIR, RUNS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        
    return device

def extract_dataset():
    """Extract Pascal VOC dataset from ZIP."""
    if not DATASET_ZIP.exists():
        print(f"❌ Error: Dataset ZIP not found at {DATASET_ZIP}")
        return False
        
    print(f"📦 Extracting {DATASET_ZIP.name}...")
    with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
        zip_ref.extractall(RAW_DATA_DIR)
    print(f"✅ Extracted to {RAW_DATA_DIR}")
    return True

def parse_pascal_voc(xml_file):
    """Parse Pascal VOC XML annotation."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

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

    return {'width': width, 'height': height, 'objects': objects}

def convert_to_yolo_format(annotation, class_mapping):
    """Convert Pascal VOC bbox to YOLO format (normalized center x, y, width, height)."""
    img_width = annotation['width']
    img_height = annotation['height']

    yolo_annotations = []
    for obj in annotation['objects']:
        class_name = obj['class']
        if class_name not in class_mapping:
            continue

        class_id = class_mapping[class_name]
        xmin, ymin, xmax, ymax = obj['bbox']

        # Convert to YOLO format
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # Clamp to [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return yolo_annotations

def prepare_dataset():
    """Convert Pascal VOC to YOLO format and split into train/val/test."""
    print("\nPreparing YOLO dataset...")
    
    # Create output subdirectories
    for split in ['train', 'val', 'test']:
        (YOLO_DATASET_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (YOLO_DATASET_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

    class_mapping = {name: idx for idx, name in enumerate(CLASSES)}

    # Get all image files (search recursively if needed, but assuming flat for now)
    image_files = list(RAW_DATA_DIR.rglob("*.jpg"))
    if not image_files:
        print("❌ No .jpg files found in extracted data")
        return None

    print(f"Found {len(image_files)} images")

    valid_data = []
    for img_file in tqdm(image_files, desc="Parsing annotations"):
        xml_file = img_file.with_suffix('.xml')
        if not xml_file.exists():
            continue

        try:
            annotation = parse_pascal_voc(xml_file)
            if annotation['objects']:
                valid_data.append((img_file, annotation))
        except Exception as e:
            continue

    if not valid_data:
        print("❌ No valid annotations found")
        return None

    # Split dataset
    train_idx, temp_idx = train_test_split(range(len(valid_data)), train_size=TRAIN_SPLIT, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, train_size=VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT), random_state=42)

    indices = {'train': train_idx, 'val': val_idx, 'test': test_idx}

    for split_name, idx_list in indices.items():
        for idx in tqdm(idx_list, desc=f"Configuring {split_name}"):
            img_file, annotation = valid_data[idx]
            yolo_ann = convert_to_yolo_format(annotation, class_mapping)
            if not yolo_ann: continue
            
            # Copy image
            shutil.copy(img_file, YOLO_DATASET_DIR / split_name / 'images' / img_file.name)
            # Write label
            with open(YOLO_DATASET_DIR / split_name / 'labels' / img_file.with_suffix('.txt').name, 'w') as f:
                f.write('\n'.join(yolo_ann))

    # Create data.yaml
    data_yaml = {
        'path': str(YOLO_DATASET_DIR.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(CLASSES),
        'names': CLASSES
    }

    yaml_path = YOLO_DATASET_DIR / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
        
    return yaml_path

def main():
    device = setup_environment()
    if not extract_dataset(): return
    
    yaml_path = prepare_dataset()
    if not yaml_path: return

    print("\n" + "="*40)
    print("🚀 STARTING TRAINING")
    print("="*40)

    model = YOLO('yolov8m.pt')
    
    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=20,
        save=True,
        device=device,
        project=str(RUNS_DIR),
        name='apparatus_detector',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        verbose=True,
        seed=42,
        # Augmentation
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10.0, translate=0.1, scale=0.5, shear=2.0,
        flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.1
    )

    print("\nEvaluating Model...")
    metrics = model.val()
    
    print("\n" + "="*40)
    print("📊 RESULTS summary")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")

    print("\nExporting to TFLite...")
    tflite_path = model.export(format='tflite', imgsz=IMG_SIZE, optimize=True)
    
    # Save metadata
    metadata = {
        'model': 'YOLOv8m',
        'trained_at': datetime.now().isoformat(),
        'metrics': {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr)
        },
        'classes': CLASSES
    }
    
    with open(RUNS_DIR / 'apparatus_detector' / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Training Complete!")
    print(f"📍 Model saved in: {RUNS_DIR / 'apparatus_detector' / 'weights'}")
    print(f"📍 TFLite exported: {tflite_path}")

if __name__ == "__main__":
    main()

