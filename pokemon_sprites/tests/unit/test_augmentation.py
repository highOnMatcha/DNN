"""
Unit tests for ARGB data augmentation module.

Tests the ARGB-focused augmentation pipeline for Pokemon sprite generation.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from data.augmentation import (
    AdvancedAugmentationPipeline,
    PairedColorJitter,
    PairedCutout,
    PairedRandomAffine,
    PairedRandomHorizontalFlip,
    PairedRandomRotation,
    get_augmentation_config,
)

try:
    from core.logging_config import get_logger

    logger = get_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


class TestARGBPairedTransforms(unittest.TestCase):
    """Test paired transforms for ARGB images."""

    def setUp(self):
        """Set up test environment with ARGB images."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.input_image = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
        self.target_image = Image.new("RGBA", (64, 64), (0, 255, 0, 255))

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_paired_random_horizontal_flip_argb(self):
        """Test PairedRandomHorizontalFlip with ARGB images."""
        flip_transform = PairedRandomHorizontalFlip(p=1.0)
        flipped_input, flipped_target = flip_transform(
            self.input_image, self.target_image
        )

        self.assertIsInstance(flipped_input, Image.Image)
        self.assertIsInstance(flipped_target, Image.Image)
        self.assertEqual(flipped_input.mode, "RGBA")
        self.assertEqual(flipped_target.mode, "RGBA")

    def test_paired_random_rotation_argb(self):
        """Test PairedRandomRotation with ARGB images."""
        rotation_transform = PairedRandomRotation(degrees=30)
        rotated_input, rotated_target = rotation_transform(
            self.input_image, self.target_image
        )

        self.assertIsInstance(rotated_input, Image.Image)
        self.assertIsInstance(rotated_target, Image.Image)
        self.assertEqual(rotated_input.mode, "RGBA")
        self.assertEqual(rotated_target.mode, "RGBA")

    def test_paired_color_jitter_argb(self):
        """Test PairedColorJitter with ARGB images."""
        color_jitter = PairedColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )
        jittered_input, jittered_target = color_jitter(
            self.input_image, self.target_image
        )

        self.assertIsInstance(jittered_input, Image.Image)
        self.assertIsInstance(jittered_target, Image.Image)
        self.assertEqual(jittered_input.mode, "RGBA")
        self.assertEqual(jittered_target.mode, "RGBA")

    def test_paired_random_affine_argb(self):
        """Test PairedRandomAffine with ARGB images."""
        affine_transform = PairedRandomAffine(
            degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)
        )
        transformed_input, transformed_target = affine_transform(
            self.input_image, self.target_image
        )

        self.assertIsInstance(transformed_input, Image.Image)
        self.assertIsInstance(transformed_target, Image.Image)
        self.assertEqual(transformed_input.mode, "RGBA")
        self.assertEqual(transformed_target.mode, "RGBA")

    def test_paired_cutout_argb(self):
        """Test PairedCutout with ARGB images."""
        cutout_transform = PairedCutout(size_ratio=16, p=1.0)
        cutout_input, cutout_target = cutout_transform(
            self.input_image, self.target_image
        )

        self.assertIsInstance(cutout_input, Image.Image)
        self.assertIsInstance(cutout_target, Image.Image)
        self.assertEqual(cutout_input.mode, "RGBA")
        self.assertEqual(cutout_target.mode, "RGBA")

    def test_advanced_augmentation_pipeline_argb(self):
        """Test AdvancedAugmentationPipeline with ARGB images."""
        config_dict = {
            "horizontal_flip_p": 0.5,
            "rotation_degrees": 15,
            "color_jitter": {
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.1,
            },
            "cutout": {"size_ratio": 16, "p": 0.2},
        }

        pipeline = AdvancedAugmentationPipeline(
            "standard", config_dict=config_dict
        )
        aug_input, aug_target = pipeline(self.input_image, self.target_image)

        self.assertIsInstance(aug_input, Image.Image)
        self.assertIsInstance(aug_target, Image.Image)
        self.assertEqual(aug_input.mode, "RGBA")
        self.assertEqual(aug_target.mode, "RGBA")

    def test_get_augmentation_config(self):
        """Test get_augmentation_config function."""
        config = get_augmentation_config("standard")
        self.assertIsInstance(config, AdvancedAugmentationPipeline)
        self.assertEqual(config.config, "standard")

    def test_argb_alpha_channel_preservation(self):
        """Test that alpha channel is preserved through augmentations."""
        test_image = Image.new("RGBA", (32, 32))
        pixels = []
        for y in range(32):
            for x in range(32):
                alpha = int(255 * (x + y) / 62)
                pixels.append((255, 0, 0, alpha))
        test_image.putdata(pixels)

        target_image = Image.new("RGBA", (32, 32), (0, 255, 0, 255))

        flip_transform = PairedRandomHorizontalFlip(p=1.0)
        flipped_input, flipped_target = flip_transform(
            test_image, target_image
        )

        original_alpha = np.array(test_image)[:, :, 3]
        flipped_alpha = np.array(flipped_input)[:, :, 3]

        np.testing.assert_array_equal(original_alpha, np.fliplr(flipped_alpha))


if __name__ == "__main__":
    unittest.main()
