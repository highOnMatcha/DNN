"""
Configuration settings for segmentation training and model parameters.

This module provides comprehensive configuration management for image segmentation
training pipelines, including model architecture parameters, training hyperparameters,
dataset configuration, and data augmentation settings.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path


# Path to external model configurations
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "model_configs.json")


def get_models_root_dir() -> str:
    """
    Get the absolute path to the segmentation/models directory.
    
    This ensures all models are stored and loaded from the same location
    regardless of where the script is executed from.
    
    Returns:
        Absolute path to the segmentation/models directory.
    """
    # Get the path to this file (segmentation/src/config/settings.py)
    current_file = Path(__file__).resolve()
    # Go up to segmentation directory: segmentation/src/config -> segmentation/src -> segmentation
    segmentation_root = current_file.parent.parent.parent
    # Create models directory path
    models_dir = segmentation_root / "models"
    # Ensure the directory exists
    models_dir.mkdir(exist_ok=True)
    return str(models_dir)


def get_data_root_dir() -> str:
    """
    Get the absolute path to the segmentation/data directory.
    
    Returns:
        Absolute path to the segmentation/data directory.
    """
    current_file = Path(__file__).resolve()
    segmentation_root = current_file.parent.parent.parent
    data_dir = segmentation_root / "data"
    data_dir.mkdir(exist_ok=True)
    return str(data_dir)


def get_results_root_dir() -> str:
    """
    Get the absolute path to the segmentation/results directory.
    
    Returns:
        Absolute path to the segmentation/results directory.
    """
    current_file = Path(__file__).resolve()
    segmentation_root = current_file.parent.parent.parent
    results_dir = segmentation_root / "results"
    results_dir.mkdir(exist_ok=True)
    return str(results_dir)


def load_model_configs() -> Dict[str, Any]:
    """
    Load model configurations from JSON file.
    
    Returns:
        Dictionary containing model configurations.
    """
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "models": {},
            "datasets": {},
            "training": {}
        }


@dataclass
class SegmentationConfig:
    """
    Configuration class for segmentation models and training.
    
    Attributes:
        # Model Configuration
        name: Model name/identifier
        architecture: Model architecture type (unet, deeplabv3, etc.)
        encoder_name: Encoder backbone (resnet50, efficientnet-b0, etc.)
        encoder_weights: Pre-trained weights (imagenet, None)
        in_channels: Number of input channels
        classes: Number of segmentation classes
        
        # Training Configuration
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        weight_decay: L2 regularization factor
        optimizer: Optimizer type (adam, sgd, adamw)
        scheduler: Learning rate scheduler type
        
        # Data Configuration
        image_size: Input image size (height, width)
        dataset_name: Dataset name (pascal_voc_2012, cityscapes, etc.)
        train_split: Training data split ratio
        val_split: Validation data split ratio
        num_workers: Number of data loading workers
        
        # Augmentation Configuration
        use_augmentation: Whether to use data augmentation
        horizontal_flip_prob: Probability of horizontal flip
        vertical_flip_prob: Probability of vertical flip
        rotation_limit: Maximum rotation angle in degrees
        brightness_limit: Brightness variation limit
        contrast_limit: Contrast variation limit
        
        # Output Configuration
        output_dir: Directory to save models and results
        save_best_only: Whether to save only the best model
        save_frequency: Frequency of model checkpointing
        
        # Logging and Monitoring
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: WandB project name
        log_frequency: Logging frequency (steps)
        
        # Evaluation Configuration
        eval_frequency: Evaluation frequency (epochs)
        metrics: List of metrics to compute
        
        # Device Configuration
        device: Device to use (cuda, cpu, auto)
        mixed_precision: Whether to use mixed precision training
    """
    
    # Model Configuration
    name: str = "unet_resnet50"
    architecture: str = "unet"
    encoder_name: str = "resnet50"
    encoder_weights: str = "imagenet"
    in_channels: int = 3
    classes: int = 21  # PASCAL VOC has 20 classes + background
    
    # Training Configuration
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    
    # Data Configuration
    image_size: Tuple[int, int] = (256, 256)
    dataset_name: str = "pascal_voc_2012"
    train_split: float = 0.8
    val_split: float = 0.2
    num_workers: int = 4
    
    # Augmentation Configuration
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.1
    rotation_limit: int = 15
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    
    # Output Configuration
    output_dir: str = ""
    save_best_only: bool = True
    save_frequency: int = 10
    
    # Logging and Monitoring
    use_wandb: bool = True
    wandb_project: str = "segmentation-bootcamp"
    log_frequency: int = 10
    
    # Evaluation Configuration
    eval_frequency: int = 1
    metrics: List[str] = None
    
    # Device Configuration
    device: str = "auto"
    mixed_precision: bool = True
    
    def __post_init__(self):
        """Post-initialization to set defaults and validate configuration."""
        if self.output_dir == "":
            self.output_dir = os.path.join(get_models_root_dir(), self.name)
        
        if self.metrics is None:
            self.metrics = ["iou", "dice", "pixel_accuracy", "mean_iou"]
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)


# Default configurations for different training scenarios
DEFAULT_CONFIG = SegmentationConfig()

TEST_CONFIG = SegmentationConfig(
    name="unet_test",
    batch_size=2,
    num_epochs=5,
    image_size=(128, 128),
    use_wandb=False,
    mixed_precision=False
)

DEVELOPMENT_CONFIG = SegmentationConfig(
    name="unet_dev",
    batch_size=4,
    num_epochs=20,
    image_size=(256, 256),
    use_wandb=True,
    mixed_precision=True
)

PRODUCTION_CONFIG = SegmentationConfig(
    name="unet_production",
    batch_size=16,
    num_epochs=100,
    image_size=(512, 512),
    encoder_name="efficientnet-b4",
    learning_rate=5e-5,
    use_wandb=True,
    mixed_precision=True
)


def get_config(config_name: str) -> SegmentationConfig:
    """
    Get a configuration by name.
    
    Args:
        config_name: Configuration name (default, test, development, production)
        
    Returns:
        SegmentationConfig instance
    """
    configs = {
        "default": DEFAULT_CONFIG,
        "test": TEST_CONFIG,
        "development": DEVELOPMENT_CONFIG,
        "production": PRODUCTION_CONFIG
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    return configs[config_name]


def list_available_configs() -> List[str]:
    """List all available configuration names."""
    return ["default", "test", "development", "production"]


# PASCAL VOC 2012 class information
PASCAL_VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

PASCAL_VOC_COLORS = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
    [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0],
    [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]
]
