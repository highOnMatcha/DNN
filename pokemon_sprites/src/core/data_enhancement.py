"""
Data enhancement utilities for Pokemon sprite generation.

This module provides advanced data augmentation and enhancement techniques
to improve training data diversity and model robustness.
"""

import sys
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from core.logging_config import get_logger

# Add src to path
current_dir = Path(__file__).parent
src_path = str(current_dir.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)


logger = get_logger(__name__)


class DataEnhancer:
    """Provides advanced data enhancement capabilities for Pokemon sprites."""

    def __init__(self, target_size: int = 256):
        """
        Initialize data enhancer.

        Args:
            target_size: Target image size for processing
        """
        self.target_size = target_size

    def color_palette_swap(
        self, image: Image.Image, target_image: Image.Image
    ) -> Image.Image:
        """
        Swap color palette from one Pokemon to another while preserving structure.

        Args:
            image: Source image to modify
            target_image: Image to extract color palette from

        Returns:
            Image with swapped color palette
        """
        try:
            img_arr = np.array(image)
            target_arr = np.array(target_image)

            target_colors = self._extract_dominant_colors(
                target_arr, n_colors=8
            )
            result = self._remap_colors(img_arr, target_colors)

            return Image.fromarray(result)
        except Exception as e:
            logger.warning(f"Color palette swap failed: {e}")
            return image

    def _extract_dominant_colors(
        self, image: np.ndarray, n_colors: int = 8
    ) -> List:
        """
        Extract dominant colors using k-means clustering.

        Args:
            image: Input image as numpy array
            n_colors: Number of dominant colors to extract

        Returns:
            List of dominant colors as RGB tuples
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            logger.warning("Invalid image format for color extraction")
            return self._get_fallback_colors(n_colors)

        pixels = image.reshape(-1, 3)

        # Get unique colors first to determine actual cluster count
        unique_pixels = np.unique(pixels, axis=0)
        actual_clusters = min(len(unique_pixels), n_colors)

        try:
            if actual_clusters <= 1:
                # Return single color or fallback colors for monochrome images
                if len(unique_pixels) == 1:
                    return [tuple(map(int, unique_pixels[0]))]
                else:
                    return self._get_fallback_colors(n_colors)

            kmeans = KMeans(
                n_clusters=actual_clusters, random_state=42, n_init=10
            )
            kmeans.fit(pixels)
            return [
                tuple(map(int, color)) for color in kmeans.cluster_centers_
            ]
        except Exception as e:
            logger.warning(f"K-means color extraction failed: {e}")
            return self._get_fallback_colors(n_colors)

    def _get_fallback_colors(self, n_colors: int) -> List:
        """Get fallback colors when extraction fails."""
        fallback_colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 128, 128),  # Gray
            (255, 128, 0),  # Orange
        ]
        return fallback_colors[:n_colors]

    def _remap_colors(
        self, image: np.ndarray, target_colors: List
    ) -> np.ndarray:
        """
        Remap image colors to target palette.

        Args:
            image: Source image array
            target_colors: Target color palette

        Returns:
            Image with remapped colors
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image

        result = image.copy()

        # Simple color remapping - find regions and replace with target colors
        try:
            for i, target_color in enumerate(
                target_colors[:4]
            ):  # Limit to avoid over-processing
                # Create a mask for similar colors
                reference_color = np.array(
                    [100 + i * 30, 50 + i * 40, 200 - i * 25]
                )
                mask = np.all(
                    np.abs(image - reference_color) < 50,
                    axis=-1,
                )
                result[mask] = target_color
        except Exception as e:
            logger.warning(f"Color remapping failed: {e}")
            return image

        return result

    def enhance_contrast(
        self, image: Image.Image, factor: float = 1.2
    ) -> Image.Image:
        """
        Enhance image contrast.

        Args:
            image: Input image
            factor: Contrast enhancement factor (1.0 = no change)

        Returns:
            Contrast-enhanced image
        """
        try:
            from PIL import ImageEnhance

            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(factor)
        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {e}")
            return image

    def enhance_sharpness(
        self, image: Image.Image, factor: float = 1.5
    ) -> Image.Image:
        """
        Enhance image sharpness for pixel art.

        Args:
            image: Input image
            factor: Sharpness enhancement factor (1.0 = no change)

        Returns:
            Sharpness-enhanced image
        """
        try:
            from PIL import ImageEnhance

            enhancer = ImageEnhance.Sharpness(image)
            return enhancer.enhance(factor)
        except Exception as e:
            logger.warning(f"Sharpness enhancement failed: {e}")
            return image

    def apply_sprite_specific_enhancements(
        self, image: Image.Image
    ) -> Image.Image:
        """
        Apply enhancements specifically designed for sprite generation.

        Args:
            image: Input image

        Returns:
            Enhanced image optimized for sprite generation
        """
        try:
            # Enhance contrast for better feature definition
            image = self.enhance_contrast(image, factor=1.1)

            # Enhance sharpness for pixel art quality
            image = self.enhance_sharpness(image, factor=1.3)

            return image
        except Exception as e:
            logger.warning(f"Sprite enhancement failed: {e}")
            return image

    def generate_color_variations(
        self, image: Image.Image, num_variations: int = 3
    ) -> List[Image.Image]:
        """
        Generate color variations of an input image.

        Args:
            image: Source image
            num_variations: Number of color variations to generate

        Returns:
            List of color-varied images
        """
        variations = []

        try:
            img_arr = np.array(image)

            for i in range(num_variations):
                # Generate different color palettes
                hue_shift = (i + 1) * 60  # Shift hue by 60, 120, 180 degrees
                variation = self._apply_hue_shift(img_arr, hue_shift)
                variations.append(Image.fromarray(variation))

        except Exception as e:
            logger.warning(f"Color variation generation failed: {e}")

        return variations

    def _apply_hue_shift(
        self, image: np.ndarray, hue_shift: int
    ) -> np.ndarray:
        """Apply hue shift to image colors."""
        try:
            import colorsys

            if len(image.shape) != 3 or image.shape[2] != 3:
                return image

            # Convert RGB to HSV, shift hue, convert back
            shifted = image.copy().astype(float) / 255.0

            for i in range(shifted.shape[0]):
                for j in range(shifted.shape[1]):
                    r, g, b = shifted[i, j]
                    h, s, v = colorsys.rgb_to_hsv(r, g, b)
                    h = (h + hue_shift / 360.0) % 1.0
                    r, g, b = colorsys.hsv_to_rgb(h, s, v)
                    shifted[i, j] = [r, g, b]

            return (shifted * 255).astype(np.uint8)

        except Exception as e:
            logger.warning(f"Hue shift failed: {e}")
            return image
