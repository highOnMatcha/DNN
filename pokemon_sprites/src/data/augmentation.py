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
    """Apply the same random affine transformation to both input and target images."""

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
    """Apply the same cutout (random erasing) to both input and target images."""

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


# Remove NoiseAugmentation as it's counterproductive for pixel art
# Remove MixupAugmentation as it's questionable for pixel art generation


class AdvancedAugmentationPipeline:
    """Advanced augmentation pipeline for Pokemon sprite generation.
    Updated for ARGB (RGBA) processing."""

    def __init__(
        self,
        config: str = "standard",
        image_size: int = 64,
        config_dict: Optional[Dict] = None,
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

        # Build augmentation pipeline based on config
        if config_dict is not None:
            # Use config_dict for direct configuration
            self.transforms = self._build_from_config_dict(config_dict)
        elif config == "none":
            self.transforms = []
        elif config == "light":
            self.transforms = [
                PairedRandomHorizontalFlip(p=0.5),
                PairedColorJitter(
                    brightness=0.05,
                    contrast=0.05,
                    saturation=0.05,
                    hue=0.02,
                ),
            ]
        elif config == "standard":
            self.transforms = [
                PairedRandomHorizontalFlip(p=0.6),
                PairedRandomRotation(degrees=5),  # Reduced for pixel art
                PairedColorJitter(
                    brightness=0.08,
                    contrast=0.08,
                    saturation=0.08,
                    hue=0.03,
                ),
                PairedRandomAffine(
                    degrees=0,  # No rotation here since we have dedicated rotation
                    translate=(0.05, 0.05),  # Small translation
                    scale=(0.95, 1.05),  # Small scale variation
                    p=0.3,
                ),
            ]
        elif config == "production":
            self.transforms = [
                PairedRandomHorizontalFlip(p=0.7),
                PairedRandomRotation(degrees=8),  # Moderate for pixel art
                PairedColorJitter(
                    brightness=0.12,
                    contrast=0.12,
                    saturation=0.12,
                    hue=0.05,
                ),
                PairedRandomAffine(
                    degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1), p=0.4
                ),
                PairedCutout(
                    size_ratio=24, p=0.2
                ),  # Applied to both input and target
            ]
        else:
            raise ValueError(f"Unknown augmentation config: {config}")

    def _build_from_config_dict(self, config_dict: Dict):
        """Build augmentation pipeline from configuration dictionary."""
        transforms = []

        # Add horizontal flip
        if config_dict.get("horizontal_flip_p", 0) > 0:
            transforms.append(
                PairedRandomHorizontalFlip(p=config_dict["horizontal_flip_p"])
            )

        # Add rotation
        if config_dict.get("rotation_degrees", 0) > 0:
            transforms.append(
                PairedRandomRotation(degrees=config_dict["rotation_degrees"])
            )

        # Add color jitter (unified format)
        color_jitter = config_dict.get("color_jitter", {})
        if color_jitter:
            # Handle both old format (input/target) and new format (unified)
            if "input" in color_jitter:
                # Old format - use input params for paired jitter
                jitter_params = color_jitter["input"]
            else:
                # New format - direct params
                jitter_params = color_jitter

            if any(
                jitter_params.get(k, 0) > 0
                for k in ["brightness", "contrast", "saturation", "hue"]
            ):
                transforms.append(
                    PairedColorJitter(
                        brightness=jitter_params.get("brightness", 0),
                        contrast=jitter_params.get("contrast", 0),
                        saturation=jitter_params.get("saturation", 0),
                        hue=jitter_params.get("hue", 0),
                    )
                )

        # Add random affine
        affine_config = config_dict.get("random_affine", {})
        if affine_config.get("p", 0) > 0:
            transforms.append(
                PairedRandomAffine(
                    degrees=affine_config.get("degrees", 0),
                    translate=tuple(
                        affine_config.get("translate", [0.0, 0.0])
                    ),
                    scale=tuple(affine_config.get("scale", [1.0, 1.0])),
                    p=affine_config.get("p", 0),
                )
            )

        # Add cutout
        cutout_config = config_dict.get("cutout", {})
        if cutout_config.get("p", 0) > 0:
            transforms.append(
                PairedCutout(
                    size_ratio=cutout_config.get("size_ratio", 32),
                    p=cutout_config.get("p", 0),
                )
            )

        return transforms

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


# Configuration presets
AUGMENTATION_PRESETS = {
    "none": {"config": "none"},
    "light": {"config": "light"},
    "standard": {"config": "standard"},
    "production": {"config": "production"},
}


def get_augmentation_config(
    augmentation_level: str, image_size: int = 64
) -> AdvancedAugmentationPipeline:
    """
    Get augmentation configuration for specified level.

    Args:
        augmentation_level: Level of augmentation ("light", "standard",
                          "production", "none")
        image_size: Target image size

    Returns:
        Configured augmentation pipeline
    """
    return AdvancedAugmentationPipeline(
        config=augmentation_level, image_size=image_size
    )
