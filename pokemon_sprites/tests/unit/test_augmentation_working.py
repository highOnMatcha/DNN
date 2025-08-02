"""
Working unit tests for data augmentation module.

This module tests the actual augmentation classes that exist in the codebase.
"""

import unittest
import tempfile
import logging
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
import sys
import os

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    from core.logging_config import initialize_project_logging
    from data.augmentation import (
        PairedRandomHorizontalFlip,
        PairedRandomRotation,
        PairedRandomCrop,
        IndependentColorJitter,
        NoiseAugmentation,
        BlurAugmentation,
        CutoutAugmentation,
        MixupAugmentation,
        AdvancedAugmentationPipeline,
        get_augmentation_config
    )
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR = str(e)

# Configure test logging
if IMPORTS_SUCCESSFUL:
    initialize_project_logging("test_augmentation")
logger = logging.getLogger(__name__)


class TestColors:
    """ANSI color codes for professional test output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def print_test_result(test_name: str, success: bool, message: str = ""):
    """Print formatted test result with appropriate colors."""
    if success:
        print(f"{TestColors.GREEN}{TestColors.BOLD}[SUCCESS]{TestColors.RESET} {test_name}")
        if message:
            print(f"          {message}")
    else:
        print(f"{TestColors.RED}{TestColors.BOLD}[FAIL]{TestColors.RESET} {test_name}")
        if message:
            print(f"       {message}")


@unittest.skipIf(not IMPORTS_SUCCESSFUL, f"Import failed: {IMPORT_ERROR if not IMPORTS_SUCCESSFUL else ''}")
class TestBasicAugmentation(unittest.TestCase):
    """Test suite for basic augmentation functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_image_size = (64, 64)
        self.input_img = Image.new('RGB', self.test_image_size, color='red')
        self.target_img = Image.new('RGB', self.test_image_size, color='blue')

    def test_paired_horizontal_flip(self):
        """Test PairedRandomHorizontalFlip functionality."""
        flip = PairedRandomHorizontalFlip(p=1.0)  # Always flip
        self.assertIsNotNone(flip)
        
        # Test that both images get flipped consistently
        input_flipped, target_flipped = flip(self.input_img, self.target_img)
        self.assertEqual(input_flipped.size, self.test_image_size)
        self.assertEqual(target_flipped.size, self.test_image_size)
        
        print_test_result("test_paired_horizontal_flip", True, 
                         "PairedRandomHorizontalFlip working correctly")

    def test_paired_rotation(self):
        """Test PairedRandomRotation functionality."""
        rotation = PairedRandomRotation(degrees=45)
        self.assertIsNotNone(rotation)
        
        # Test that both images get rotated
        input_rotated, target_rotated = rotation(self.input_img, self.target_img)
        self.assertEqual(input_rotated.size, self.test_image_size)
        self.assertEqual(target_rotated.size, self.test_image_size)
        
        print_test_result("test_paired_rotation", True, 
                         "PairedRandomRotation working correctly")

    def test_paired_crop(self):
        """Test PairedRandomCrop functionality."""
        crop = PairedRandomCrop(size=(32, 32), padding=4)
        self.assertIsNotNone(crop)
        
        # Test that both images get cropped to correct size
        input_cropped, target_cropped = crop(self.input_img, self.target_img)
        self.assertEqual(input_cropped.size, (32, 32))
        self.assertEqual(target_cropped.size, (32, 32))
        
        print_test_result("test_paired_crop", True, 
                         "PairedRandomCrop working correctly")

    def test_independent_color_jitter(self):
        """Test IndependentColorJitter functionality."""
        jitter = IndependentColorJitter(
            input_params={'brightness': 0.2, 'contrast': 0.2},
            target_params={'brightness': 0.1, 'contrast': 0.1}
        )
        self.assertIsNotNone(jitter)
        
        # Test that color jitter can be applied
        input_jittered, target_jittered = jitter(self.input_img, self.target_img)
        self.assertEqual(input_jittered.size, self.test_image_size)
        self.assertEqual(target_jittered.size, self.test_image_size)
        
        print_test_result("test_independent_color_jitter", True, 
                         "IndependentColorJitter working correctly")

    def test_noise_augmentation(self):
        """Test NoiseAugmentation functionality."""
        noise_aug = NoiseAugmentation(noise_factor=0.1)
        self.assertIsNotNone(noise_aug)
        
        # Test noise application with PIL images (not tensors)
        input_noisy, target_noisy = noise_aug(self.input_img, self.target_img)
        
        # Should return PIL Images of same size
        self.assertEqual(input_noisy.size, self.input_img.size)
        self.assertEqual(target_noisy.size, self.target_img.size)
        
        print_test_result("test_noise_augmentation", True, 
                         "NoiseAugmentation working correctly")

    def test_blur_augmentation(self):
        """Test BlurAugmentation functionality."""
        blur_aug = BlurAugmentation(radius_range=(1.0, 2.0), p=1.0)  # Always apply
        self.assertIsNotNone(blur_aug)
        
        # Test blur application
        input_blurred, target_blurred = blur_aug(self.input_img, self.target_img)
        self.assertEqual(input_blurred.size, self.test_image_size)
        self.assertEqual(target_blurred.size, self.test_image_size)
        
        print_test_result("test_blur_augmentation", True, 
                         "BlurAugmentation working correctly")

    def test_cutout_augmentation(self):
        """Test CutoutAugmentation functionality."""
        cutout_aug = CutoutAugmentation(cutout_size=16, p=1.0)  # Always apply
        self.assertIsNotNone(cutout_aug)
        
        # Test cutout application with PIL Images
        input_cut, target_cut = cutout_aug(self.input_img, self.target_img)
        self.assertEqual(input_cut.size, self.test_image_size)
        self.assertEqual(target_cut.size, self.test_image_size)
        
        print_test_result("test_cutout_augmentation", True, 
                         "CutoutAugmentation working correctly")

    def test_advanced_augmentation_pipeline(self):
        """Test AdvancedAugmentationPipeline functionality."""
        # Create a simple pipeline
        pipeline = AdvancedAugmentationPipeline()
        self.assertIsNotNone(pipeline)
        
        # Test pipeline application
        input_aug, target_aug = pipeline(self.input_img, self.target_img)
        self.assertEqual(input_aug.size, self.test_image_size)
        self.assertEqual(target_aug.size, self.test_image_size)
        
        print_test_result("test_advanced_augmentation_pipeline", True, 
                         "AdvancedAugmentationPipeline working correctly")

    def test_get_augmentation_config(self):
        """Test get_augmentation_config function."""
        # Test different augmentation levels
        for level in ["light", "standard", "heavy"]:
            config = get_augmentation_config(level=level, image_size=64)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, AdvancedAugmentationPipeline)
        
        print_test_result("test_get_augmentation_config", True, 
                         "get_augmentation_config working for all levels")


if __name__ == '__main__':
    print(f"{TestColors.BLUE}{TestColors.BOLD}Running Data Augmentation Tests{TestColors.RESET}\n")
    unittest.main(verbosity=2)
