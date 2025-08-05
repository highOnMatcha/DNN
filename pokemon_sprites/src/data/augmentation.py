"""
Advanced data augmentation for Pokemon sprite generation.

Implements comprehensive augmentation techniques for image-to-image translation
tasks. Based on pix2pix augmentation strategies (Isola et al., 2017) with
pixel art optimizations. Updated for ARGB (RGBA) processing.
"""

import random
from typing import Dict, Optional, Tuple

import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageOps

from core.logging_config import get_logger
logger = get_logger(__name__)


class PairedRandomHorizontalFlip:
    """
    Apply the same random horizontal flip to both input and target images.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, input_img: Image.Image, target_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            return ImageOps.mirror(input_img), ImageOps.mirror(target_img)
        return input_img, target_img


class PairedRandomRotation:
    """Apply the same random rotation to both input and target images."""

    def __init__(self, degrees: float = 15):
        self.degrees = degrees

    def __call__(
        self, input_img: Image.Image, target_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        angle = random.uniform(-self.degrees, self.degrees)
        return input_img.rotate(angle, expand=False), target_img.rotate(
            angle, expand=False
        )


class PairedColorJitter:
    """Apply the same color jitter to both input and target images.
    Updated for pixel art generation where input and target should have
    matching color variations to maintain correspondence."""

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(
        self, input_img: Image.Image, target_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        # Generate the same random parameters for both images
        jitter_transform = transforms.ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue,
        )

        # Handle RGBA images by separating alpha channel
        def apply_jitter_rgba(img, jitter_transform):
            if img.mode == "RGBA":
                # Split RGBA into RGB and Alpha
                rgb_img = img.convert("RGB")
                alpha_channel = img.split()[-1]

                # Apply color jitter to RGB only
                jittered_rgb = jitter_transform(rgb_img)

                # Recombine with original alpha
                jittered_rgba = Image.merge(
                    "RGBA", (*jittered_rgb.split(), alpha_channel)
                )
                return jittered_rgba
            else:
                # For non-RGBA images, apply jitter normally
                return jitter_transform(img)

        # Apply the SAME jitter to both input and target
        return (
            apply_jitter_rgba(input_img, jitter_transform),
            apply_jitter_rgba(target_img, jitter_transform),
        )


class PairedRandomAffine:
    """Apply same random affine transformation to input and target images."""

    def __init__(
        self,
        degrees: float = 0,
        translate: Tuple[float, float] = (0.0, 0.0),
        scale: Tuple[float, float] = (1.0, 1.0),
        p: float = 0.5,
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.p = p

    def __call__(
        self, input_img: Image.Image, target_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            # Generate random parameters
            angle = (
                random.uniform(-self.degrees, self.degrees)
                if self.degrees > 0
                else 0
            )
            (
                random.uniform(-self.translate[0], self.translate[0])
                if self.translate[0] > 0
                else 0
            )
            (
                random.uniform(-self.translate[1], self.translate[1])
                if self.translate[1] > 0
                else 0
            )
            scale_factor = (
                random.uniform(self.scale[0], self.scale[1])
                if self.scale[0] != self.scale[1]
                else 1.0
            )

            # Apply same transformation to both images
            affine_transform = transforms.RandomAffine(
                degrees=0,  # We handle rotation manually for consistency
                translate=(0, 0),
                scale=(scale_factor, scale_factor),
            )

            # Apply transformations
            if angle != 0:
                input_img = input_img.rotate(angle, expand=False)
                target_img = target_img.rotate(angle, expand=False)

            if scale_factor != 1.0:
                input_img = affine_transform(input_img)
                target_img = affine_transform(target_img)

        return input_img, target_img


class PairedCutout:
    """Apply same cutout (random erasing) to input and target images."""

    def __init__(self, size_ratio: int = 32, p: float = 0.1):
        self.size_ratio = size_ratio
        self.p = p

    def __call__(
        self, input_img: Image.Image, target_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            # Convert to numpy for manipulation
            input_array = np.array(input_img)
            target_array = np.array(target_img)

            h, w = input_array.shape[:2]
            cut_size = max(1, min(h, w) // self.size_ratio)

            # Random position (same for both images)
            y = random.randint(0, max(0, h - cut_size))
            x = random.randint(0, max(0, w - cut_size))

            # Apply cutout to both images
            input_array[y : y + cut_size, x : x + cut_size] = 0
            target_array[y : y + cut_size, x : x + cut_size] = 0

            input_img = Image.fromarray(input_array)
            target_img = Image.fromarray(target_array)

        return input_img, target_img

class AdvancedAugmentationPipeline:
    """Advanced augmentation pipeline for Pokemon sprite generation.
    Updated for ARGB (RGBA) processing."""

    def __init__(
        self,
        config: str = "standard",
        image_size: int = 96,
    ):
        """
        Initialize augmentation pipeline.

        Args:
            config: Augmentation configuration level
                ("light", "standard", "production", "none")
            image_size: Target image size
            config_dict: Optional config dictionary override
        """
        self.config = config
        self.image_size = image_size
        logger.info(f"Initializing augmentation pipeline with config: {config}")
        # Build augmentation pipeline based on config
        if config == "none":
            logger.info("No augmentation applied")
            self.transforms = []
        elif config is not None:
            self.transforms = self._build_from_config(config)
        else:
            raise ValueError(f"Unknown augmentation config: {config}")
        
    def _build_from_config(self, config: str) -> list:
        """Build augmentation pipeline from configuration json."""
        #load the json config, example above
        import json
        from pathlib import Path
        config_path = Path(__file__).parent / "../config/model_configs.json"
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        if config not in config_dict["augmentation_configs"]:
            raise ValueError(f"Unknown augmentation config: {config}")
        aug_config = config_dict["augmentation_configs"][config]
        transforms_list = []
        # Add horizontal flip
        if "horizontal_flip_p" in aug_config:
            transforms_list.append(
                PairedRandomHorizontalFlip(aug_config["horizontal_flip_p"])
            )
        # Add rotation
        if "rotation_degrees" in aug_config:
            transforms_list.append(
                PairedRandomRotation(aug_config["rotation_degrees"])
            )
        # Add color jitter
        if "color_jitter" in aug_config:
            transforms_list.append(
                PairedColorJitter(**aug_config["color_jitter"])
            )
        # Add random affine
        if "random_affine" in aug_config:
            transforms_list.append(
                PairedRandomAffine(**aug_config["random_affine"])
            )
        # Add cutout
        if "cutout" in aug_config:
            transforms_list.append(PairedCutout(**aug_config["cutout"]))
        # Return composed transforms
        return transforms_list


    def set_dataset(self, dataset):
        """Set dataset reference for augmentations that need it."""
        # No longer needed since we removed MixupAugmentation

    def __call__(
        self, input_img: Image.Image, target_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Apply augmentation pipeline to image pair."""
        for transform in self.transforms:
            input_img, target_img = transform(input_img, target_img)
        return input_img, target_img


def get_augmentation_config(
    augmentation_level: str, image_size: int = 96
) -> AdvancedAugmentationPipeline:
    """
    Get augmentation configuration for specified level.

    Args:
        augmentation_level: Level of augmentation 
        image_size: Target image size

    Returns:
        Configured augmentation pipeline
    """
    return AdvancedAugmentationPipeline(
        config=augmentation_level, image_size=image_size
    )
