# -*- coding: utf-8 -*-
"""
Automated Image Downloader for Gymnastics Apparatus Detection Training
Uses icrawler to download images from Google Images for each apparatus class.

Installation:
    pip install icrawler

Usage:
    python download_apparatus_images.py
"""

import os
from icrawler.builtin import GoogleImageCrawler

# Configuration
BASE_DIR = "gym_data/raw_images"
IMAGES_PER_APPARATUS = 150  # Adjust as needed (100-200 recommended)

# Gymnastics apparatus with optimized search queries
APPARATUS_CONFIG = {
    "pommel_horse": {
        "name": "Pommel Horse",
        "queries": [
            "gymnast performing on pommel horse competition",
            "pommel horse gymnastics apparatus",
            "men's pommel horse olympic gymnastics"
        ]
    },
    "still_rings": {
        "name": "Still Rings",
        "queries": [
            "gymnast performing on still rings competition",
            "still rings gymnastics apparatus",
            "men's rings olympic gymnastics"
        ]
    },
    "vault": {
        "name": "Vault",
        "queries": [
            "gymnast performing vault competition",
            "gymnastics vault apparatus",
            "olympic vault gymnastics"
        ]
    },
    "parallel_bars": {
        "name": "Parallel Bars",
        "queries": [
            "gymnast performing on parallel bars competition",
            "parallel bars gymnastics apparatus",
            "men's parallel bars olympic gymnastics"
        ]
    },
    "horizontal_bar": {
        "name": "Horizontal Bar",
        "queries": [
            "gymnast performing on horizontal bar competition",
            "high bar gymnastics apparatus",
            "men's horizontal bar olympic gymnastics"
        ]
    },
    "uneven_bars": {
        "name": "Uneven Bars",
        "queries": [
            "gymnast performing on uneven bars competition",
            "uneven bars gymnastics apparatus",
            "women's uneven bars olympic gymnastics"
        ]
    },
    "balance_beam": {
        "name": "Balance Beam",
        "queries": [
            "gymnast performing on balance beam competition",
            "balance beam gymnastics apparatus",
            "women's balance beam olympic gymnastics"
        ]
    },
    "floor_exercise": {
        "name": "Floor Exercise",
        "queries": [
            "gymnast performing floor exercise competition",
            "gymnastics floor exercise mat",
            "olympic floor exercise gymnastics"
        ]
    }
}


def download_apparatus_images(apparatus_key, config, images_per_query):
    """
    Download images for a specific apparatus using multiple search queries.
    
    Args:
        apparatus_key: Folder name (e.g., 'pommel_horse')
        config: Configuration dict with name and queries
        images_per_query: Number of images to download per query
    """
    folder_path = os.path.join(BASE_DIR, apparatus_key)
    
    # Create folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    print(f"\n{'='*60}")
    print(f"Downloading: {config['name']}")
    print(f"{'='*60}")
    
    total_downloaded = 0
    
    for idx, query in enumerate(config['queries'], 1):
        print(f"\n[Query {idx}/{len(config['queries'])}] {query}")
        
        # Initialize Google Image Crawler
        google_crawler = GoogleImageCrawler(
            storage={'root_dir': folder_path},
            downloader_threads=4,  # Parallel downloads
            log_level='ERROR'  # Reduce verbosity
        )
        
        try:
            google_crawler.crawl(
                keyword=query,
                max_num=images_per_query,
                min_size=(400, 400),  # Minimum image size
                file_idx_offset=total_downloaded  # Continue numbering
            )
            total_downloaded += images_per_query
            print(f"[OK] Downloaded ~{images_per_query} images")
        except Exception as e:
            print(f"[WARNING] Error downloading: {e}")
    
    print(f"\n[OK] Total images for {config['name']}: ~{total_downloaded}")


def main():
    """Main function to download all apparatus images."""
    print("="*60)
    print("Gymnastics Apparatus Image Downloader")
    print("="*60)
    print(f"\nTarget: {IMAGES_PER_APPARATUS} images per apparatus")
    print(f"Output: {BASE_DIR}/")
    print("\n[WARNING] This may take 15-30 minutes depending on your connection")
    
    # Confirm before starting
    response = input("\n>> Start downloading? (y/n): ").strip().lower()
    if response != 'y':
        print("Download cancelled.")
        return
    
    # Calculate images per query (divide total by number of queries)
    images_per_query = IMAGES_PER_APPARATUS // 3  # We have 3 queries per apparatus
    
    # Download images for each apparatus
    for apparatus_key, config in APPARATUS_CONFIG.items():
        download_apparatus_images(apparatus_key, config, images_per_query)
    
    print("\n" + "="*60)
    print("[SUCCESS] Download Complete!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Review downloaded images and remove any irrelevant ones")
    print("2. Install LabelImg: pip install labelImg")
    print("3. Annotate images: labelImg gym_data\\raw_images")
    print("4. Use class names: Pommel_Horse, Still_Rings, Vault, etc.")
    print("5. Run training script: python train_apparatus_detector.py")
    print("\n[TIP] Aim to keep 100-150 high-quality images per apparatus")


if __name__ == "__main__":
    # Check if icrawler is installed
    try:
        import icrawler
        main()
    except ImportError:
        print("[ERROR] icrawler is not installed")
        print("\nInstall it with:")
        print("  pip install icrawler")
        print("\nThen run this script again.")
