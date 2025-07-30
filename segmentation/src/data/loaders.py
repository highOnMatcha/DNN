"""
Dataset loading and preprocessing for segmentation tasks.

This module handles loading of PASCAL VOC 2012 and other segmentation datasets,
with support for data augmentation, preprocessing, and efficient data loading.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Dict, List, Tuple, Optional, Callable, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from pathlib import Path

from ..config.settings import PASCAL_VOC_CLASSES, PASCAL_VOC_COLORS


class PascalVOCDataset(Dataset):
    """
    PASCAL VOC 2012 segmentation dataset.
    
    Args:
        root_dir: Root directory containing the PASCAL VOC dataset
        split: Dataset split ('train', 'val', 'trainval')
        transform: Albumentations transform pipeline
        target_size: Target image size (height, width)
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (256, 256)
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        
        # Define paths
        self.images_dir = self.root_dir / "JPEGImages"
        self.masks_dir = self.root_dir / "SegmentationClass"
        self.splits_dir = self.root_dir / "ImageSets" / "Segmentation"
        
        # Load image IDs for the split
        split_file = self.splits_dir / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        
        # Verify that images and masks exist
        self._verify_dataset()
        
        self.classes = PASCAL_VOC_CLASSES
        self.num_classes = len(self.classes)
        
    def _verify_dataset(self):
        """Verify that all required files exist."""
        missing_images = []
        missing_masks = []
        
        for img_id in self.image_ids:
            img_path = self.images_dir / f"{img_id}.jpg"
            mask_path = self.masks_dir / f"{img_id}.png"
            
            if not img_path.exists():
                missing_images.append(str(img_path))
            if not mask_path.exists():
                missing_masks.append(str(mask_path))
        
        if missing_images:
            print(f"Warning: {len(missing_images)} missing images")
        if missing_masks:
            print(f"Warning: {len(missing_masks)} missing masks")
        
        # Remove entries with missing files
        valid_ids = []
        for img_id in self.image_ids:
            img_path = self.images_dir / f"{img_id}.jpg"
            mask_path = self.masks_dir / f"{img_id}.png"
            if img_path.exists() and mask_path.exists():
                valid_ids.append(img_id)
        
        self.image_ids = valid_ids
        print(f"Dataset {self.split}: {len(self.image_ids)} valid samples")
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_id = self.image_ids[idx]
        
        # Load image
        img_path = self.images_dir / f"{img_id}.jpg"
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.masks_dir / f"{img_id}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Handle class 255 (border/ignore) by setting it to 0 (background)
        mask[mask == 255] = 0
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return {
            'image': image,
            'mask': mask.long(),
            'image_id': img_id
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for balanced training.
        
        Returns:
            Tensor of class weights
        """
        class_counts = np.zeros(self.num_classes)
        
        print("Calculating class weights...")
        for idx in range(len(self)):
            sample = self[idx]
            mask = sample['mask'].numpy()
            unique, counts = np.unique(mask, return_counts=True)
            for cls, count in zip(unique, counts):
                if cls < self.num_classes:
                    class_counts[cls] += count
        
        # Calculate weights (inverse frequency)
        total_pixels = class_counts.sum()
        class_weights = total_pixels / (self.num_classes * class_counts + 1e-8)
        
        return torch.FloatTensor(class_weights)


def get_augmentation_pipeline(
    config,
    is_training: bool = True
) -> A.Compose:
    """
    Create augmentation pipeline using Albumentations.
    
    Args:
        config: Configuration object with augmentation parameters
        is_training: Whether this is for training (applies augmentations) or validation
        
    Returns:
        Albumentations compose object
    """
    if not is_training:
        # Validation/test pipeline - only resize and normalize
        return A.Compose([
            A.Resize(height=config.image_size[0], width=config.image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    transforms_list = [
        A.Resize(height=config.image_size[0], width=config.image_size[1])
    ]
    
    if config.use_augmentation:
        # Add augmentations for training
        if config.horizontal_flip_prob > 0:
            transforms_list.append(A.HorizontalFlip(p=config.horizontal_flip_prob))
        
        if config.vertical_flip_prob > 0:
            transforms_list.append(A.VerticalFlip(p=config.vertical_flip_prob))
        
        if config.rotation_limit > 0:
            transforms_list.append(A.Rotate(limit=config.rotation_limit, p=0.5))
        
        if config.brightness_limit > 0 or config.contrast_limit > 0:
            transforms_list.append(A.RandomBrightnessContrast(
                brightness_limit=config.brightness_limit,
                contrast_limit=config.contrast_limit,
                p=0.5
            ))
        
        # Additional augmentations
        transforms_list.extend([
            A.OneOf([
                A.Blur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.1),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=0.1),
        ])
    
    # Always apply normalization and tensor conversion
    transforms_list.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return A.Compose(transforms_list)


def create_dataloaders(
    config,
    train_dataset: Dataset,
    val_dataset: Dataset
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        config: Configuration object
        train_dataset: Training dataset
        val_dataset: Validation dataset
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    return train_loader, val_loader


def download_pascal_voc_2012(data_dir: str) -> str:
    """
    Download and extract PASCAL VOC 2012 dataset.
    
    Args:
        data_dir: Directory to download and extract the dataset
        
    Returns:
        Path to the extracted dataset
    """
    import urllib.request
    import tarfile
    
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_dir = data_dir / "VOCdevkit" / "VOC2012"
    
    if dataset_dir.exists():
        print(f"Dataset already exists at {dataset_dir}")
        return str(dataset_dir)
    
    # Download URLs
    base_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/"
    files_to_download = [
        "VOCtrainval_11-May-2012.tar"
    ]
    
    for filename in files_to_download:
        url = base_url + filename
        filepath = data_dir / filename
        
        if not filepath.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded {filename}")
        
        # Extract
        print(f"Extracting {filename}...")
        with tarfile.open(filepath, 'r') as tar:
            tar.extractall(data_dir)
        print(f"Extracted {filename}")
    
    return str(dataset_dir)


def create_datasets(config) -> Tuple[PascalVOCDataset, PascalVOCDataset]:
    """
    Create training and validation datasets.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    from ..config.settings import get_data_root_dir
    
    # Get data directory
    data_dir = get_data_root_dir()
    
    # Download dataset if needed
    if config.dataset_name == "pascal_voc_2012":
        dataset_path = download_pascal_voc_2012(data_dir)
        
        # Create augmentation pipelines
        train_transform = get_augmentation_pipeline(config, is_training=True)
        val_transform = get_augmentation_pipeline(config, is_training=False)
        
        # Create datasets
        train_dataset = PascalVOCDataset(
            root_dir=dataset_path,
            split="train",
            transform=train_transform,
            target_size=config.image_size
        )
        
        val_dataset = PascalVOCDataset(
            root_dir=dataset_path,
            split="val",
            transform=val_transform,
            target_size=config.image_size
        )
        
        return train_dataset, val_dataset
    
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset_name}")


def visualize_sample(
    dataset: Dataset,
    idx: int,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize a dataset sample with image and mask overlay.
    
    Args:
        dataset: Dataset instance
        idx: Sample index
        save_path: Optional path to save the visualization
    """
    import matplotlib.pyplot as plt
    
    sample = dataset[idx]
    image = sample['image']
    mask = sample['mask']
    
    # Convert tensors to numpy if needed
    if torch.is_tensor(image):
        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        image = image * std[:, None, None] + mean[:, None, None]
        image = torch.clamp(image, 0, 1)
        image = image.permute(1, 2, 0).numpy()
    
    if torch.is_tensor(mask):
        mask = mask.numpy()
    
    # Create color mask
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id in range(len(PASCAL_VOC_COLORS)):
        color_mask[mask == class_id] = PASCAL_VOC_COLORS[class_id]
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='tab20')
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    axes[2].imshow(image)
    axes[2].imshow(color_mask, alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
