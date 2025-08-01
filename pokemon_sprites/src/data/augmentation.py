"""
Advanced data augmentation for Pokemon sprite generation.

Implements comprehensive augmentation techniques for image-to-image translation tasks.
Based on pix2pix augmentation strategies (Isola et al., 2017) with pixel art optimizations.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageEnhance
import random
import numpy as np
from typing import Tuple, Optional, List, Union, Callable, Dict
import math


class PairedRandomHorizontalFlip:
    """Apply the same random horizontal flip to both input and target images."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, input_img: Image.Image, target_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            return TF.hflip(input_img), TF.hflip(target_img)
        return input_img, target_img


class PairedRandomRotation:
    """Apply the same random rotation to both input and target images."""
    
    def __init__(self, degrees: float = 15):
        self.degrees = degrees
    
    def __call__(self, input_img: Image.Image, target_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        angle = random.uniform(-self.degrees, self.degrees)
        return TF.rotate(input_img, angle), TF.rotate(target_img, angle)


class PairedRandomCrop:
    """Apply the same random crop to both input and target images."""
    
    def __init__(self, size: Union[int, Tuple[int, int]], padding: int = 0):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.padding = padding
    
    def __call__(self, input_img: Image.Image, target_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if self.padding > 0:
            input_img = TF.pad(input_img, self.padding)
            target_img = TF.pad(target_img, self.padding)
        
        # Ensure crop size doesn't exceed image dimensions
        img_width, img_height = input_img.size
        crop_h = min(self.size[0], img_height)
        crop_w = min(self.size[1], img_width)
        
        # If image is smaller than desired crop size, return original images
        if crop_h >= img_height or crop_w >= img_width:
            return input_img, target_img
        
        # Random crop coordinates
        left = random.randint(0, img_width - crop_w)
        top = random.randint(0, img_height - crop_h)
        right = left + crop_w
        bottom = top + crop_h
        
        return input_img.crop((left, top, right, bottom)), target_img.crop((left, top, right, bottom))


class PairedRandomAffine:
    """Apply the same random affine transformation to both images."""
    
    def __init__(self, degrees: float = 0, translate: Optional[Tuple[float, float]] = None,
                 scale: Optional[Tuple[float, float]] = None, shear: Optional[float] = None):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
    
    def __call__(self, input_img: Image.Image, target_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # Get transform parameters
        angle = random.uniform(-self.degrees, self.degrees) if self.degrees > 0 else 0
        
        translate_x = translate_y = 0
        if self.translate:
            max_dx = self.translate[0] * input_img.width
            max_dy = self.translate[1] * input_img.height
            translate_x = random.uniform(-max_dx, max_dx)
            translate_y = random.uniform(-max_dy, max_dy)
        
        scale_factor = 1.0
        if self.scale:
            scale_factor = random.uniform(self.scale[0], self.scale[1])
        
        shear_x = shear_y = 0
        if self.shear:
            shear_x = random.uniform(-self.shear, self.shear)
            shear_y = random.uniform(-self.shear, self.shear)
        
        # Apply same transformation to both images using PIL Image transforms
        # Convert parameters to the format expected by PIL
        transform_params = {
            'angle': angle,
            'translate': (int(translate_x), int(translate_y)),
            'scale': scale_factor,
            'shear': (shear_x, shear_y)
        }
        
        # Use transforms.functional with PIL Images
        input_transformed = TF.affine(input_img, **transform_params)
        target_transformed = TF.affine(target_img, **transform_params)
        
        return input_transformed, target_transformed


class IndependentColorJitter:
    """Apply different color jittering to input and target images."""
    
    def __init__(self, input_params: dict, target_params: dict):
        self.input_jitter = transforms.ColorJitter(**input_params)
        self.target_jitter = transforms.ColorJitter(**target_params)
    
    def __call__(self, input_img: Image.Image, target_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        return self.input_jitter(input_img), self.target_jitter(target_img)


class InputOnlyAugmentation:
    """Apply augmentation only to input image, useful for style variations."""
    
    def __init__(self, augmentations: List[Callable]):
        self.augmentations = transforms.Compose(augmentations)
    
    def __call__(self, input_img: Image.Image, target_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        return self.augmentations(input_img), target_img


class NoiseAugmentation:
    """Add noise to input images to improve robustness."""
    
    def __init__(self, noise_factor: float = 0.1, p: float = 0.3):
        self.noise_factor = noise_factor
        self.p = p
    
    def __call__(self, input_img: Image.Image, target_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            # Convert to numpy for noise addition
            img_array = np.array(input_img)
            noise = np.random.normal(0, self.noise_factor * 255, img_array.shape).astype(np.int16)
            noisy_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            input_img = Image.fromarray(noisy_array)
        
        return input_img, target_img


class BlurAugmentation:
    """Apply random blur to input images."""
    
    def __init__(self, radius_range: Tuple[float, float] = (0.5, 1.5), p: float = 0.2):
        self.radius_range = radius_range
        self.p = p
    
    def __call__(self, input_img: Image.Image, target_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            input_img = input_img.filter(ImageFilter.GaussianBlur(radius=radius))
        
        return input_img, target_img


class CutoutAugmentation:
    """Apply cutout augmentation to input images."""
    
    def __init__(self, cutout_size: int = 16, p: float = 0.3):
        self.cutout_size = cutout_size
        self.p = p
    
    def __call__(self, input_img: Image.Image, target_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            img_array = np.array(input_img)
            h, w = img_array.shape[:2]
            
            # Random position for cutout
            y = random.randint(0, h - self.cutout_size)
            x = random.randint(0, w - self.cutout_size)
            
            # Apply cutout (fill with mean color)
            mean_color = img_array.mean(axis=(0, 1)).astype(np.uint8)
            img_array[y:y+self.cutout_size, x:x+self.cutout_size] = mean_color
            
            input_img = Image.fromarray(img_array)
        
        return input_img, target_img


class MixupAugmentation:
    """Mixup augmentation for both input and target images."""
    
    def __init__(self, alpha: float = 0.2, p: float = 0.1):
        self.alpha = alpha
        self.p = p
        self.dataset = None  # Will be set by the dataset
    
    def set_dataset(self, dataset):
        self.dataset = dataset
    
    def __call__(self, input_img: Image.Image, target_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p and self.dataset is not None:
            # Get random sample from dataset
            rand_idx = random.randint(0, len(self.dataset) - 1)
            rand_input, rand_target = self.dataset.get_raw_sample(rand_idx)
            
            # Resize random images to match current images
            rand_input = rand_input.resize(input_img.size)
            rand_target = rand_target.resize(target_img.size)
            
            # Generate lambda
            lam = np.random.beta(self.alpha, self.alpha)
            
            # Convert to numpy arrays
            input_array = np.array(input_img).astype(np.float32)
            target_array = np.array(target_img).astype(np.float32)
            rand_input_array = np.array(rand_input).astype(np.float32)
            rand_target_array = np.array(rand_target).astype(np.float32)
            
            # Mix images
            mixed_input = (lam * input_array + (1 - lam) * rand_input_array).astype(np.uint8)
            mixed_target = (lam * target_array + (1 - lam) * rand_target_array).astype(np.uint8)
            
            return Image.fromarray(mixed_input), Image.fromarray(mixed_target)
        
        return input_img, target_img


class AdvancedAugmentationPipeline:
    """Advanced augmentation pipeline for Pokemon sprite generation."""
    
    def __init__(self, config: str = "standard", image_size: int = 64, config_dict: Optional[Dict] = None):
        """
        Initialize augmentation pipeline.
        
        Args:
            config: Augmentation configuration level ("light", "standard", "production", "none")
            image_size: Target image size
            config_dict: Optional dictionary with augmentation parameters
        """
        self.image_size = image_size
        self.config = config
        self.config_dict = config_dict
        self.augmentations = self._build_pipeline(config)
    
    def _build_pipeline(self, config: str) -> List:
        """Build augmentation pipeline based on configuration."""
        # Use provided config_dict or fall back to hardcoded values
        if self.config_dict:
            return self._build_from_dict(self.config_dict)
        
        # Fallback to hardcoded configurations
        if config == "light":
            return [
                PairedRandomHorizontalFlip(p=0.5),
                IndependentColorJitter(
                    input_params={"brightness": 0.08, "contrast": 0.08, "saturation": 0.08, "hue": 0.03},
                    target_params={"brightness": 0.02, "contrast": 0.02, "saturation": 0.02, "hue": 0.01}
                ),
                NoiseAugmentation(noise_factor=0.02, p=0.15)
            ]
        
        elif config == "standard":
            return [
                PairedRandomHorizontalFlip(p=0.5),
                PairedRandomRotation(degrees=5),
                IndependentColorJitter(
                    input_params={"brightness": 0.12, "contrast": 0.12, "saturation": 0.12, "hue": 0.05},
                    target_params={"brightness": 0.04, "contrast": 0.04, "saturation": 0.04, "hue": 0.02}
                ),
                NoiseAugmentation(noise_factor=0.04, p=0.2),
                CutoutAugmentation(cutout_size=max(2, self.image_size // 24), p=0.15)
            ]
        
        elif config == "production":
            return [
                PairedRandomHorizontalFlip(p=0.6),
                PairedRandomRotation(degrees=8),
                IndependentColorJitter(
                    input_params={"brightness": 0.15, "contrast": 0.15, "saturation": 0.15, "hue": 0.08},
                    target_params={"brightness": 0.06, "contrast": 0.06, "saturation": 0.06, "hue": 0.03}
                ),
                NoiseAugmentation(noise_factor=0.06, p=0.25),
                CutoutAugmentation(cutout_size=max(3, self.image_size // 20), p=0.2)
            ]
        
        elif config == "none":
            return []
        
        else:
            return self._build_pipeline("standard")
    
    def _build_from_dict(self, config_dict: Dict) -> List:
        """Build augmentation pipeline from configuration dictionary."""
        augmentations = []
        
        # Horizontal flip
        if config_dict.get("horizontal_flip_p", 0) > 0:
            augmentations.append(PairedRandomHorizontalFlip(p=config_dict["horizontal_flip_p"]))
        
        # Rotation
        if config_dict.get("rotation_degrees", 0) > 0:
            augmentations.append(PairedRandomRotation(degrees=config_dict["rotation_degrees"]))
        
        # Color jitter
        color_jitter = config_dict.get("color_jitter")
        if color_jitter:
            augmentations.append(IndependentColorJitter(
                input_params=color_jitter["input"],
                target_params=color_jitter["target"]
            ))
        
        # Noise
        noise_config = config_dict.get("noise")
        if noise_config and noise_config.get("p", 0) > 0:
            augmentations.append(NoiseAugmentation(
                noise_factor=noise_config["factor"],
                p=noise_config["p"]
            ))
        
        # Blur
        blur_config = config_dict.get("blur")
        if blur_config and blur_config.get("p", 0) > 0:
            augmentations.append(BlurAugmentation(
                radius_range=tuple(blur_config["radius_range"]),
                p=blur_config["p"]
            ))
        
        # Cutout
        cutout_config = config_dict.get("cutout")
        if cutout_config and cutout_config.get("p", 0) > 0:
            cutout_size = max(4, self.image_size // cutout_config["size_ratio"])
            augmentations.append(CutoutAugmentation(
                cutout_size=cutout_size,
                p=cutout_config["p"]
            ))
        
        return augmentations
    
    def __call__(self, input_img: Image.Image, target_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Apply augmentation pipeline."""
        for aug in self.augmentations:
            input_img, target_img = aug(input_img, target_img)
        
        return input_img, target_img
    
    def set_dataset(self, dataset):
        """Set dataset reference for augmentations that need it (like Mixup)."""
        for aug in self.augmentations:
            if hasattr(aug, 'set_dataset'):
                aug.set_dataset(dataset)


def get_augmentation_config(level: str = "standard", image_size: int = 64) -> AdvancedAugmentationPipeline:
    """
    Get augmentation configuration optimized for pixel art generation.
    
    Args:
        level: Augmentation level ("light", "standard", "production", "none")
        image_size: Target image size
    
    Returns:
        Configured augmentation pipeline
    """
    # Try to load from JSON config
    try:
        from config.settings import load_model_configs
        config_data = load_model_configs()
        augmentation_configs = config_data.get("augmentation_configs", {})
        config_dict = augmentation_configs.get(level)
        
        if config_dict:
            return AdvancedAugmentationPipeline(config=level, image_size=image_size, config_dict=config_dict)
    except ImportError:
        pass
    
    # Fallback to hardcoded configuration
    return AdvancedAugmentationPipeline(config=level, image_size=image_size)


# Preset configurations for pixel art generation
AUGMENTATION_PRESETS = {
    "test": "light",
    "development": "standard", 
    "production": "production",
    "none": "none"
}
