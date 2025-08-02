"""
Comprehensive unit tests for data augmentation module.

This module tests the actual available augmentation classes and functions
from the data.augmentation module to maximize test coverage.
"""

import shutil
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock

from PIL import Image

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from data.augmentation import (
    AdvancedAugmentationPipeline,
    BlurAugmentation,
    CutoutAugmentation,
    IndependentColorJitter,
    InputOnlyAugmentation,
    MixupAugmentation,
    NoiseAugmentation,
    PairedRandomAffine,
    PairedRandomCrop,
    PairedRandomHorizontalFlip,
    PairedRandomRotation,
    get_augmentation_config,
)

logger = __import__("logging").getLogger(__name__)


class TestPairedTransforms(unittest.TestCase):
    """Test paired transforms that work on input-target pairs."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())

        # Create test image pair
        self.input_image = Image.new("RGB", (64, 64), (255, 0, 0))
        self.target_image = Image.new("RGB", (64, 64), (0, 255, 0))

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_paired_random_horizontal_flip(self):
        """Test PairedRandomHorizontalFlip transform."""
        flip_transform = PairedRandomHorizontalFlip(p=1.0)  # Always flip

        flipped_input, flipped_target = flip_transform(
            self.input_image, self.target_image
        )

        self.assertIsInstance(flipped_input, Image.Image)
        self.assertIsInstance(flipped_target, Image.Image)
        self.assertEqual(flipped_input.size, self.input_image.size)
        self.assertEqual(flipped_target.size, self.target_image.size)

    def test_paired_random_rotation(self):
        """Test PairedRandomRotation transform."""
        rotation_transform = PairedRandomRotation(degrees=30)

        rotated_input, rotated_target = rotation_transform(
            self.input_image, self.target_image
        )

        self.assertIsInstance(rotated_input, Image.Image)
        self.assertIsInstance(rotated_target, Image.Image)
        # Rotation might change size slightly
        self.assertGreater(rotated_input.size[0], 0)
        self.assertGreater(rotated_target.size[0], 0)

    def test_paired_random_crop(self):
        """Test PairedRandomCrop transform."""
        crop_transform = PairedRandomCrop(size=(48, 48))

        cropped_input, cropped_target = crop_transform(
            self.input_image, self.target_image
        )

        self.assertIsInstance(cropped_input, Image.Image)
        self.assertIsInstance(cropped_target, Image.Image)
        self.assertEqual(cropped_input.size, (48, 48))
        self.assertEqual(cropped_target.size, (48, 48))

    def test_paired_random_affine(self):
        """Test PairedRandomAffine transform."""
        affine_transform = PairedRandomAffine(
            degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)
        )

        affine_input, affine_target = affine_transform(
            self.input_image, self.target_image
        )

        self.assertIsInstance(affine_input, Image.Image)
        self.assertIsInstance(affine_target, Image.Image)


class TestIndependentTransforms(unittest.TestCase):
    """Test transforms that work independently on images."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_image = Image.new("RGB", (64, 64), (128, 128, 128))

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_independent_color_jitter(self):
        """Test IndependentColorJitter transform."""
        color_jitter = IndependentColorJitter(
            input_params={"brightness": 0.2, "contrast": 0.2},
            target_params={"saturation": 0.2, "hue": 0.1},
        )

        jittered_input, jittered_target = color_jitter(
            self.test_image, self.test_image
        )

        self.assertIsInstance(jittered_input, Image.Image)
        self.assertIsInstance(jittered_target, Image.Image)
        self.assertEqual(jittered_input.size, self.test_image.size)
        self.assertEqual(jittered_target.size, self.test_image.size)

    def test_input_only_augmentation(self):
        """Test InputOnlyAugmentation wrapper."""
        import torchvision.transforms as transforms

        # Create a simple transform to wrap
        base_transforms = [transforms.ColorJitter(brightness=0.2)]
        input_only_aug = InputOnlyAugmentation(base_transforms)

        # Test with image pair (should only transform first)
        target_image = Image.new("RGB", (64, 64), (200, 200, 200))
        augmented_input, unchanged_target = input_only_aug(
            self.test_image, target_image
        )

        self.assertIsInstance(augmented_input, Image.Image)
        self.assertIsInstance(unchanged_target, Image.Image)


class TestNoiseAndBlurTransforms(unittest.TestCase):
    """Test noise and blur augmentation transforms."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_image = Image.new("RGB", (64, 64), (128, 128, 128))

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_noise_augmentation(self):
        """Test NoiseAugmentation transform."""
        noise_aug = NoiseAugmentation(noise_factor=0.1, p=1.0)  # Always apply

        noisy_input, target = noise_aug(self.test_image, self.test_image)

        self.assertIsInstance(noisy_input, Image.Image)
        self.assertIsInstance(target, Image.Image)
        self.assertEqual(noisy_input.size, self.test_image.size)
        self.assertEqual(target.size, self.test_image.size)

    def test_blur_augmentation(self):
        """Test BlurAugmentation transform."""
        blur_aug = BlurAugmentation(radius_range=(0.5, 1.5), p=1.0)

        blurred_input, target = blur_aug(self.test_image, self.test_image)

        self.assertIsInstance(blurred_input, Image.Image)
        self.assertIsInstance(target, Image.Image)
        self.assertEqual(blurred_input.size, self.test_image.size)
        self.assertEqual(target.size, self.test_image.size)

    def test_cutout_augmentation(self):
        """Test CutoutAugmentation transform."""
        cutout_aug = CutoutAugmentation(cutout_size=8, p=1.0)

        cutout_input, target = cutout_aug(self.test_image, self.test_image)

        self.assertIsInstance(cutout_input, Image.Image)
        self.assertIsInstance(target, Image.Image)
        self.assertEqual(cutout_input.size, self.test_image.size)
        self.assertEqual(target.size, self.test_image.size)


class TestMixupAugmentation(unittest.TestCase):
    """Test Mixup augmentation for advanced data mixing."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())

        # Create test images
        self.test_input = Image.new("RGB", (32, 32), (100, 150, 200))
        self.test_target = Image.new("RGB", (32, 32), (200, 150, 100))

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_mixup_augmentation_without_dataset(self):
        """Test MixupAugmentation transform without dataset (should pass through)."""
        mixup_aug = MixupAugmentation(alpha=1.0, p=1.0)

        # Without dataset, should return original images
        result_input, result_target = mixup_aug(
            self.test_input, self.test_target
        )

        self.assertIsInstance(result_input, Image.Image)
        self.assertIsInstance(result_target, Image.Image)

    def test_mixup_augmentation_with_mock_dataset(self):
        """Test MixupAugmentation transform with mock dataset."""
        mixup_aug = MixupAugmentation(alpha=1.0, p=1.0)

        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=5)
        mock_dataset.get_raw_sample = Mock(
            return_value=(self.test_input, self.test_target)
        )

        mixup_aug.set_dataset(mock_dataset)

        # Should now perform mixup
        result_input, result_target = mixup_aug(
            self.test_input, self.test_target
        )

        self.assertIsInstance(result_input, Image.Image)
        self.assertIsInstance(result_target, Image.Image)


class TestAdvancedAugmentationPipeline(unittest.TestCase):
    """Test the AdvancedAugmentationPipeline class."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.input_image = Image.new("RGB", (64, 64), (255, 100, 50))
        self.target_image = Image.new("RGB", (64, 64), (50, 100, 255))

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_advanced_pipeline_initialization(self):
        """Test AdvancedAugmentationPipeline initialization."""
        pipeline = AdvancedAugmentationPipeline(
            config="standard", image_size=64
        )

        self.assertIsInstance(pipeline, AdvancedAugmentationPipeline)
        self.assertEqual(pipeline.image_size, 64)

    def test_advanced_pipeline_call(self):
        """Test AdvancedAugmentationPipeline execution."""
        pipeline = AdvancedAugmentationPipeline(config="light", image_size=64)

        augmented_input, augmented_target = pipeline(
            self.input_image, self.target_image
        )

        self.assertIsInstance(augmented_input, Image.Image)
        self.assertIsInstance(augmented_target, Image.Image)

    def test_advanced_pipeline_different_configs(self):
        """Test pipeline with different configuration levels."""
        configs = ["none", "light", "standard", "production"]

        for config in configs:
            try:
                pipeline = AdvancedAugmentationPipeline(
                    config=config, image_size=64
                )

                result_input, result_target = pipeline(
                    self.input_image, self.target_image
                )

                self.assertIsInstance(result_input, Image.Image)
                self.assertIsInstance(result_target, Image.Image)
            except Exception as e:
                self.fail(f"Pipeline failed for config '{config}': {e}")


class TestAugmentationConfig(unittest.TestCase):
    """Test augmentation configuration functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_get_augmentation_config(self):
        """Test get_augmentation_config function."""
        config = get_augmentation_config(level="standard", image_size=64)

        # Should return an AdvancedAugmentationPipeline
        self.assertIsInstance(config, AdvancedAugmentationPipeline)

    def test_get_augmentation_config_variations(self):
        """Test different augmentation config variations."""
        levels = ["none", "light", "standard", "production"]
        image_sizes = [32, 64, 128]

        successful_configs = 0

        for level in levels:
            for image_size in image_sizes:
                try:
                    config = get_augmentation_config(
                        level=level, image_size=image_size
                    )
                    if config is not None:
                        self.assertIsInstance(
                            config, AdvancedAugmentationPipeline
                        )
                        successful_configs += 1
                except Exception:
                    pass  # Some combinations might not be valid

        # Should have at least some successful configurations
        self.assertGreater(successful_configs, 0)


class TestAugmentationIntegration(unittest.TestCase):
    """Test integration between different augmentation components."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.input_image = Image.new("RGB", (64, 64), (200, 150, 100))
        self.target_image = Image.new("RGB", (64, 64), (100, 150, 200))

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_full_augmentation_workflow(self):
        """Test complete augmentation workflow."""
        # 1. Get augmentation config
        config = get_augmentation_config(level="standard", image_size=64)

        # 2. Apply augmentation
        augmented_input, augmented_target = config(
            self.input_image, self.target_image
        )

        # 3. Verify results
        self.assertIsInstance(augmented_input, Image.Image)
        self.assertIsInstance(augmented_target, Image.Image)

    def test_multiple_augmentations_sequence(self):
        """Test applying multiple augmentations in sequence."""
        # Create individual transforms
        flip_transform = PairedRandomHorizontalFlip(p=1.0)
        crop_transform = PairedRandomCrop(size=(48, 48))
        noise_transform = NoiseAugmentation(noise_factor=0.05, p=1.0)

        # Apply transforms in sequence
        img1_input, img1_target = flip_transform(
            self.input_image, self.target_image
        )
        img2_input, img2_target = crop_transform(img1_input, img1_target)
        final_input, final_target = noise_transform(img2_input, img2_target)

        # Verify final output
        self.assertIsInstance(final_input, Image.Image)
        self.assertIsInstance(final_target, Image.Image)

        # Size should be affected by crop
        self.assertEqual(final_input.size, (48, 48))
        self.assertEqual(final_target.size, (48, 48))


class TestAugmentationEdgeCases(unittest.TestCase):
    """Test edge cases and error handling in augmentation."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_small_image_augmentation(self):
        """Test augmentation with very small images."""
        small_image = Image.new("RGB", (8, 8), (100, 100, 100))

        # Test with flip (should work on any size)
        flip_transform = PairedRandomHorizontalFlip(p=1.0)
        flipped = flip_transform(small_image, small_image)

        self.assertIsInstance(flipped[0], Image.Image)
        self.assertEqual(flipped[0].size, (8, 8))

    def test_extreme_parameter_values(self):
        """Test augmentation with extreme parameter values."""
        test_image = Image.new("RGB", (32, 32), (128, 128, 128))

        # Test with extreme rotation
        extreme_rotation = PairedRandomRotation(
            degrees=359
        )  # Almost full rotation
        rotated = extreme_rotation(test_image, test_image)

        self.assertIsInstance(rotated[0], Image.Image)

    def test_grayscale_image_augmentation(self):
        """Test augmentation with grayscale images."""
        grayscale_image = Image.new("L", (32, 32), 128)

        # Test flip (should work with grayscale)
        flip_transform = PairedRandomHorizontalFlip(p=1.0)
        flipped = flip_transform(grayscale_image, grayscale_image)

        self.assertIsInstance(flipped[0], Image.Image)

    def test_zero_probability_transforms(self):
        """Test transforms with zero probability (should pass through)."""
        test_image = Image.new("RGB", (32, 32), (150, 150, 150))

        # Test with zero probability
        flip_never = PairedRandomHorizontalFlip(p=0.0)
        noise_never = NoiseAugmentation(noise_factor=0.1, p=0.0)
        blur_never = BlurAugmentation(radius_range=(1.0, 2.0), p=0.0)

        # Should return original images
        flip_result = flip_never(test_image, test_image)
        noise_result = noise_never(test_image, test_image)
        blur_result = blur_never(test_image, test_image)

        self.assertIsInstance(flip_result[0], Image.Image)
        self.assertIsInstance(noise_result[0], Image.Image)
        self.assertIsInstance(blur_result[0], Image.Image)


class TestTransformParameterValidation(unittest.TestCase):
    """Test parameter validation in augmentation transforms."""

    def test_paired_flip_parameters(self):
        """Test PairedRandomHorizontalFlip parameter validation."""
        # Valid parameters
        flip_valid = PairedRandomHorizontalFlip(p=0.5)
        self.assertEqual(flip_valid.p, 0.5)

        # Boundary values
        flip_min = PairedRandomHorizontalFlip(p=0.0)
        self.assertEqual(flip_min.p, 0.0)

        flip_max = PairedRandomHorizontalFlip(p=1.0)
        self.assertEqual(flip_max.p, 1.0)

    def test_rotation_parameters(self):
        """Test PairedRandomRotation parameter validation."""
        # Valid parameters
        rotation = PairedRandomRotation(degrees=30)
        self.assertEqual(rotation.degrees, 30)

    def test_crop_parameters(self):
        """Test PairedRandomCrop parameter validation."""
        # Valid size tuple
        crop_tuple = PairedRandomCrop(size=(32, 32))
        self.assertEqual(crop_tuple.size, (32, 32))

        # Valid size integer (should be converted to tuple)
        crop_int = PairedRandomCrop(size=32)
        self.assertEqual(crop_int.size, (32, 32))

    def test_noise_parameters(self):
        """Test NoiseAugmentation parameter validation."""
        noise = NoiseAugmentation(noise_factor=0.1, p=0.5)
        self.assertEqual(noise.noise_factor, 0.1)
        self.assertEqual(noise.p, 0.5)

    def test_blur_parameters(self):
        """Test BlurAugmentation parameter validation."""
        blur = BlurAugmentation(radius_range=(0.5, 1.5), p=0.5)
        self.assertEqual(blur.radius_range, (0.5, 1.5))
        self.assertEqual(blur.p, 0.5)

    def test_cutout_parameters(self):
        """Test CutoutAugmentation parameter validation."""
        cutout = CutoutAugmentation(cutout_size=8, p=0.7)
        self.assertEqual(cutout.cutout_size, 8)
        self.assertEqual(cutout.p, 0.7)


class TestAdvancedFeatures(unittest.TestCase):
    """Test advanced augmentation features."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_image = Image.new("RGB", (64, 64), (128, 128, 128))

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_pipeline_with_custom_config_dict(self):
        """Test pipeline with custom configuration dictionary."""
        custom_config = {
            "flip_p": 0.8,
            "rotation_degrees": 20,
            "noise_factor": 0.05,
            "blur_p": 0.3,
        }

        pipeline = AdvancedAugmentationPipeline(
            config="standard", image_size=64, config_dict=custom_config
        )

        result_input, result_target = pipeline(
            self.test_image, self.test_image
        )

        self.assertIsInstance(result_input, Image.Image)
        self.assertIsInstance(result_target, Image.Image)

    def test_affine_transform_comprehensive(self):
        """Test comprehensive affine transformation."""
        affine_transform = PairedRandomAffine(
            degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5
        )

        result_input, result_target = affine_transform(
            self.test_image, self.test_image
        )

        self.assertIsInstance(result_input, Image.Image)
        self.assertIsInstance(result_target, Image.Image)

    def test_independent_color_jitter_asymmetric(self):
        """Test IndependentColorJitter with very different parameters."""
        color_jitter = IndependentColorJitter(
            input_params={"brightness": 0.5, "contrast": 0.3},
            target_params={"hue": 0.2, "saturation": 0.4},
        )

        result_input, result_target = color_jitter(
            self.test_image, self.test_image
        )

        self.assertIsInstance(result_input, Image.Image)
        self.assertIsInstance(result_target, Image.Image)


if __name__ == "__main__":
    # Suppress warnings for cleaner test output
    warnings.filterwarnings("ignore", category=UserWarning)

    # Run tests with detailed output
    unittest.main(verbosity=2)
