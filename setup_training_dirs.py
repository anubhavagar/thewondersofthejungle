"""
Setup script for creating the training data directory structure.
Run this before collecting images for apparatus detection training.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path("gym_data")

# Directory structure
DIRECTORIES = [
    "train",
    "val",
    "raw_images/pommel_horse",
    "raw_images/still_rings",
    "raw_images/vault",
    "raw_images/parallel_bars",
    "raw_images/horizontal_bar",
    "raw_images/uneven_bars",
    "raw_images/balance_beam",
    "raw_images/floor_exercise"
]

def setup_directories():
    """Create all necessary directories for training data."""
    print("🏗️  Setting up training data directory structure...\n")
    
    for dir_path in DIRECTORIES:
        full_path = BASE_DIR / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {full_path}")
    
    print("\n" + "="*60)
    print("✅ Directory structure created successfully!")
    print("="*60)
    print("\n📂 Next Steps:")
    print("1. Download gymnastics apparatus images from Google Images")
    print("2. Organize images into the apparatus folders:")
    print(f"   - {BASE_DIR}/raw_images/pommel_horse/")
    print(f"   - {BASE_DIR}/raw_images/still_rings/")
    print(f"   - {BASE_DIR}/raw_images/vault/")
    print(f"   - {BASE_DIR}/raw_images/parallel_bars/")
    print(f"   - {BASE_DIR}/raw_images/horizontal_bar/")
    print(f"   - {BASE_DIR}/raw_images/uneven_bars/")
    print(f"   - {BASE_DIR}/raw_images/balance_beam/")
    print(f"   - {BASE_DIR}/raw_images/floor_exercise/")
    print("\n3. Install LabelImg: pip install labelImg")
    print("4. Annotate images: labelImg gym_data\\raw_images")
    print("\n💡 Aim for 100-200 images per apparatus class!")

if __name__ == "__main__":
    setup_directories()
