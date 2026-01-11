"""
Common dataset utilities for all experiments
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger(__name__)


class MVTecADDataset(Dataset):
    """MVTec Anomaly Detection Dataset."""
    
    def __init__(
        self,
        root_path: str,
        category: str,
        split: str = 'train',  # 'train' or 'test'
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Args:
            root_path: Path to MVTec AD dataset
            category: Product category (e.g., 'bottle', 'cable')
            split: 'train' or 'test'
            image_size: Size to resize images to
            transform: Torchvision transforms
        """
        self.root_path = Path(root_path)
        self.category = category
        self.split = split
        self.image_size = image_size
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
        
        # Load image paths
        self.image_paths = []
        self.labels = []
        self.mask_paths = []
        
        self._load_paths()
    
    def _load_paths(self):
        """Load image paths from directory structure."""
        split_path = self.root_path / self.category / self.split
        
        # Normal samples
        normal_dir = split_path / 'good'
        if normal_dir.exists():
            for img_path in sorted(normal_dir.glob('*.png')):
                self.image_paths.append(img_path)
                self.labels.append(0)  # Normal
                self.mask_paths.append(None)
        
        # Anomalous samples (only in test split)
        if self.split == 'test':
            for anomaly_dir in split_path.iterdir():
                if anomaly_dir.is_dir() and anomaly_dir.name != 'good':
                    for img_path in sorted(anomaly_dir.glob('*.png')):
                        self.image_paths.append(img_path)
                        self.labels.append(1)  # Anomaly
                        
                        # Look for corresponding mask
                        mask_path = self.root_path / self.category / 'ground_truth' / anomaly_dir.name / img_path.stem + '_mask.png'
                        self.mask_paths.append(mask_path if mask_path.exists() else None)
        
        logger.info(f"Loaded {len(self.image_paths)} images for {self.category} ({self.split})")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        """Return image, label, and optional mask."""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Load mask if available
        mask = None
        mask_path = self.mask_paths[idx]
        if mask_path is not None:
            mask = Image.open(mask_path).convert('L')
            mask = transforms.Resize((self.image_size, self.image_size))(mask)
            mask = torch.from_numpy(np.array(mask) / 255.0).float()
        
        return {
            'image': image,
            'label': label,
            'mask': mask,
            'path': str(img_path)
        }


def load_mvtec_dataset(
    root_path: str,
    category: str,
    image_size: int = 224
) -> Tuple[MVTecADDataset, MVTecADDataset]:
    """Load train and test datasets for a category."""
    
    train_dataset = MVTecADDataset(
        root_path=root_path,
        category=category,
        split='train',
        image_size=image_size
    )
    
    test_dataset = MVTecADDataset(
        root_path=root_path,
        category=category,
        split='test',
        image_size=image_size
    )
    
    return train_dataset, test_dataset


def create_dataloaders(
    category: str,
    data_path: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders."""
    
    train_dataset, test_dataset = load_mvtec_dataset(
        root_path=data_path,
        category=category,
        image_size=image_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader
