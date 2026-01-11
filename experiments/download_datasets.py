#!/usr/bin/env python3
"""
Download and extract MVTec AD and VisA datasets
"""
import os
import subprocess
import tarfile
from pathlib import Path
import urllib.request
import sys

# Dataset URLs
MVTEC_CATEGORIES = {
    'bottle': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937370-1629958698/bottle.tar.xz',
    'capsule': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937454-1629958872/capsule.tar.xz',
    'cable': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937413-1629958794/cable.tar.xz',
    'hazelnut': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937545-1629959162/hazelnut.tar.xz',
    'leather': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937607-1629959262/leather.tar.xz',
    'screw': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938130-1629960389/screw.tar.xz',
}

VISA_CATEGORIES = {
    # Add VisA URLs here when provided
}

DATA_BASE_PATH = Path(__file__).parent / 'data'
MVTEC_PATH = DATA_BASE_PATH / 'mvtec_ad'
VISA_PATH = DATA_BASE_PATH / 'visa'

def download_file(url, output_path):
    """Download a file with progress"""
    print(f"Downloading {output_path.name}...")
    try:
        urllib.request.urlretrieve(url, output_path, reporthook=lambda a, b, c: print(f"  {a*b//c}%", end='\r'))
        print(f"  ✓ Downloaded {output_path.name}")
        return True
    except Exception as e:
        print(f"  ✗ Error downloading {output_path.name}: {e}")
        return False

def extract_tar_xz(tar_path, extract_path):
    """Extract tar.xz file"""
    print(f"Extracting {tar_path.name}...")
    try:
        with tarfile.open(tar_path, 'r:xz') as tar:
            tar.extractall(path=extract_path)
        print(f"  ✓ Extracted {tar_path.name}")
        tar_path.unlink()  # Remove tar file after extraction
        return True
    except Exception as e:
        print(f"  ✗ Error extracting {tar_path.name}: {e}")
        return False

def download_mvtec_categories():
    """Download and extract MVTec AD categories"""
    print("\n" + "="*70)
    print("DOWNLOADING MVTec AD CATEGORIES")
    print("="*70)
    
    for category, url in MVTEC_CATEGORIES.items():
        print(f"\n[{category.upper()}]")
        tar_path = MVTEC_PATH / f'{category}.tar.xz'
        
        # Download
        if not tar_path.exists():
            if not download_file(url, tar_path):
                continue
        else:
            print(f"  File already exists, skipping download")
        
        # Extract
        if tar_path.exists():
            if not extract_tar_xz(tar_path, MVTEC_PATH):
                continue
            
            # Verify structure
            category_dir = MVTEC_PATH / category
            if category_dir.exists():
                print(f"  ✓ Category folder created at {category_dir}")
                # List structure
                if (category_dir / 'train').exists():
                    train_files = len(list((category_dir / 'train').rglob('*.png')))
                    test_files = len(list((category_dir / 'test').rglob('*.png')))
                    print(f"    - Train images: {train_files}")
                    print(f"    - Test images: {test_files}")

def download_visa_categories():
    """Download and extract VisA categories"""
    if not VISA_CATEGORIES:
        print("\n" + "="*70)
        print("NO VisA CATEGORIES PROVIDED YET")
        print("="*70)
        return
    
    print("\n" + "="*70)
    print("DOWNLOADING VisA CATEGORIES")
    print("="*70)
    
    for category, url in VISA_CATEGORIES.items():
        print(f"\n[{category.upper()}]")
        tar_path = VISA_PATH / f'{category}.tar.xz'
        
        # Download
        if not tar_path.exists():
            if not download_file(url, tar_path):
                continue
        else:
            print(f"  File already exists, skipping download")
        
        # Extract
        if tar_path.exists():
            if not extract_tar_xz(tar_path, VISA_PATH):
                continue
            
            # Verify structure
            category_dir = VISA_PATH / category
            if category_dir.exists():
                print(f"  ✓ Category folder created at {category_dir}")
                if (category_dir / 'train').exists():
                    train_files = len(list((category_dir / 'train').rglob('*.png')))
                    test_files = len(list((category_dir / 'test').rglob('*.png')))
                    print(f"    - Train images: {train_files}")
                    print(f"    - Test images: {test_files}")

def verify_structure():
    """Verify final directory structure"""
    print("\n" + "="*70)
    print("VERIFICATION: DIRECTORY STRUCTURE")
    print("="*70)
    
    print(f"\nMVTec AD Path: {MVTEC_PATH}")
    if MVTEC_PATH.exists():
        mvtec_dirs = [d.name for d in MVTEC_PATH.iterdir() if d.is_dir()]
        print(f"  Categories: {sorted(mvtec_dirs)}")
        print(f"  Total: {len(mvtec_dirs)} categories")
    else:
        print(f"  ✗ Path does not exist")
    
    print(f"\nVisA Path: {VISA_PATH}")
    if VISA_PATH.exists():
        visa_dirs = [d.name for d in VISA_PATH.iterdir() if d.is_dir()]
        if visa_dirs:
            print(f"  Categories: {sorted(visa_dirs)}")
            print(f"  Total: {len(visa_dirs)} categories")
        else:
            print(f"  No categories downloaded yet")
    else:
        print(f"  ✗ Path does not exist")

if __name__ == '__main__':
    print("MVTec AD & VisA Dataset Downloader")
    print("==================================\n")
    
    # Ensure paths exist
    MVTEC_PATH.mkdir(parents=True, exist_ok=True)
    VISA_PATH.mkdir(parents=True, exist_ok=True)
    
    # Download datasets
    download_mvtec_categories()
    download_visa_categories()
    
    # Verify
    verify_structure()
    
    print("\n" + "="*70)
    print("✓ Download and extraction complete!")
    print("="*70)
