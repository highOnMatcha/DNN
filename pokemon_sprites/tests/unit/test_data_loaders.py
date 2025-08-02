"""
Unit tests for the data loaders module.

This module contains comprehensive unit tests for all functions in the
src/data/loaders.py module, ensuring robust data loading and processing
functionality with maximum code coverage.
"""

import json
import logging
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import numpy as np
import requests
from PIL import Image

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.logging_config import initialize_project_logging
from data.loaders import (
    calculate_image_stats,
    create_train_val_split,
    download_pokemon_data_with_cache,
    find_valid_pairs,
    process_image_pairs,
    resize_with_padding,
    save_dataset_metadata,
)

# Import test utilities
from tests import TestDataFactory

# Configure test logging
initialize_project_logging("test_data_loaders")
logger = logging.getLogger(__name__)


class TestColors:
    """ANSI color codes for professional test output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_test_result(test_name: str, success: bool, message: str = ""):
    """Print formatted test result with appropriate colors."""
    if success:
        print(
            f"{TestColors.GREEN}{TestColors.BOLD}[SUCCESS]{TestColors.RESET} {test_name}"
        )
        if message:
            print(f"          {message}")
    else:
        print(
            f"{TestColors.RED}{TestColors.BOLD}[FAIL]{TestColors.RESET} {test_name}"
        )
        if message:
            print(f"       {message}")


class TestDownloadPokemonDataWithCache(unittest.TestCase):
    """Test suite for download_pokemon_data_with_cache function."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.pokemon_ids = [1, 2, 3]
        logger.info(f"Created test directory: {self.test_dir}")

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            logger.info(f"Cleaned up test directory: {self.test_dir}")

    @patch("data.loaders.requests.get")
    def test_successful_download(self, mock_get):
        """Test successful download of Pokemon data."""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"fake_image_data"
        mock_get.return_value = mock_response

        downloaded, cached, failed = download_pokemon_data_with_cache(
            self.test_dir, self.pokemon_ids, "black-white"
        )

        self.assertEqual(downloaded, 3)
        self.assertEqual(cached, 0)
        self.assertEqual(len(failed), 0)

        # Verify files were created
        for pokemon_id in self.pokemon_ids:
            file_path = self.test_dir / f"pokemon_{pokemon_id:04d}.png"
            self.assertTrue(file_path.exists())

        print_test_result(
            "test_successful_download",
            True,
            f"Downloaded {downloaded} files successfully",
        )

    @patch("data.loaders.requests.get")
    def test_cached_files(self, mock_get):
        """Test that existing files are detected as cached."""
        # Create existing files
        for pokemon_id in self.pokemon_ids:
            file_path = self.test_dir / f"pokemon_{pokemon_id:04d}.png"
            file_path.write_bytes(b"existing_data")

        downloaded, cached, failed = download_pokemon_data_with_cache(
            self.test_dir, self.pokemon_ids, "black-white"
        )

        self.assertEqual(downloaded, 0)
        self.assertEqual(cached, 3)
        self.assertEqual(len(failed), 0)
        # Ensure requests.get was not called
        mock_get.assert_not_called()

        print_test_result(
            "test_cached_files",
            True,
            f"Detected {cached} cached files correctly",
        )

    @patch("data.loaders.requests.get")
    def test_failed_downloads(self, mock_get):
        """Test handling of failed downloads."""
        # Mock failed HTTP response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        downloaded, cached, failed = download_pokemon_data_with_cache(
            self.test_dir, self.pokemon_ids, "black-white"
        )

        self.assertEqual(downloaded, 0)
        self.assertEqual(cached, 0)
        self.assertEqual(len(failed), 3)
        self.assertEqual(failed, self.pokemon_ids)

        print_test_result(
            "test_failed_downloads",
            True,
            f"Handled {len(failed)} failed downloads correctly",
        )

    @patch("data.loaders.requests.get")
    def test_network_timeout(self, mock_get):
        """Test handling of network timeouts."""
        mock_get.side_effect = requests.exceptions.Timeout()

        downloaded, cached, failed = download_pokemon_data_with_cache(
            self.test_dir, self.pokemon_ids, "black-white"
        )

        self.assertEqual(downloaded, 0)
        self.assertEqual(cached, 0)
        self.assertEqual(len(failed), 3)

        print_test_result(
            "test_network_timeout", True, "Handled network timeouts correctly"
        )

    def test_sprite_type_urls(self):
        """Test correct URL generation for different sprite types."""
        with patch("data.loaders.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b"fake_image_data"
            mock_get.return_value = mock_response

            # Test black-white sprite type
            download_pokemon_data_with_cache(self.test_dir, [1], "black-white")
            expected_url_bw = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-v/black-white/1.png"

            # Test artwork sprite type
            download_pokemon_data_with_cache(self.test_dir, [2], "artwork")
            expected_url_artwork = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/2.png"

            # Check that both URLs were called
            expected_calls = [
                call(expected_url_bw, timeout=10),
                call(expected_url_artwork, timeout=10),
            ]
            mock_get.assert_has_calls(expected_calls)

        print_test_result(
            "test_sprite_type_urls",
            True,
            "URL generation for different sprite types verified",
        )


class TestFindValidPairs(unittest.TestCase):
    """Test suite for find_valid_pairs function."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.sprites_dir = self.test_dir / "sprites"
        self.artwork_dir = self.test_dir / "artwork"
        self.sprites_dir.mkdir()
        self.artwork_dir.mkdir()

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def _create_test_image(self, path: Path, size: tuple = (64, 64)):
        """Helper method to create a test image using shared utility."""
        img = TestDataFactory.create_test_image(size, "red")
        img.save(path)

    def test_valid_pairs_found(self):
        """Test finding valid sprite-artwork pairs."""
        # Create matching sprite and artwork files
        self._create_test_image(self.sprites_dir / "pokemon_0001.png")
        self._create_test_image(self.artwork_dir / "pokemon_0001_artwork.png")
        self._create_test_image(self.sprites_dir / "pokemon_0002_bw.png")
        self._create_test_image(self.artwork_dir / "pokemon_0002_artwork.png")

        pairs = find_valid_pairs(self.sprites_dir, self.artwork_dir)

        self.assertEqual(len(pairs), 2)
        self.assertIn("0001", [pair["pokemon_id"] for pair in pairs])
        self.assertIn("0002", [pair["pokemon_id"] for pair in pairs])

        print_test_result(
            "test_valid_pairs_found",
            True,
            f"Found {len(pairs)} valid pairs correctly",
        )

    def test_missing_artwork(self):
        """Test handling of sprites without matching artwork."""
        # Create sprite without matching artwork
        self._create_test_image(self.sprites_dir / "pokemon_0001.png")
        # No corresponding artwork file

        pairs = find_valid_pairs(self.sprites_dir, self.artwork_dir)

        self.assertEqual(len(pairs), 0)
        print_test_result(
            "test_missing_artwork",
            True,
            "Correctly handled missing artwork files",
        )

    def test_invalid_image_files(self):
        """Test handling of corrupted or invalid image files."""
        # Create invalid image files
        sprite_path = self.sprites_dir / "pokemon_0001.png"
        artwork_path = self.artwork_dir / "pokemon_0001_artwork.png"

        sprite_path.write_text("not an image")  # Invalid content
        artwork_path.write_text("not an image")  # Invalid content

        pairs = find_valid_pairs(self.sprites_dir, self.artwork_dir)

        self.assertEqual(len(pairs), 0)
        print_test_result(
            "test_invalid_image_files",
            True,
            "Correctly handled invalid image files",
        )

    def test_nonexistent_directories(self):
        """Test handling of nonexistent directories."""
        nonexistent_dir = self.test_dir / "nonexistent"

        pairs = find_valid_pairs(nonexistent_dir, self.artwork_dir)
        self.assertEqual(len(pairs), 0)

        pairs = find_valid_pairs(self.sprites_dir, nonexistent_dir)
        self.assertEqual(len(pairs), 0)

        print_test_result(
            "test_nonexistent_directories",
            True,
            "Correctly handled nonexistent directories",
        )

    def test_filename_formats(self):
        """Test different sprite filename formats."""
        # Test various naming conventions
        test_files = [
            ("pokemon_0001.png", "0001"),
            ("pokemon_0002_bw.png", "0002"),
            ("invalid_format.png", None),  # Should be skipped
            ("pokemon_.png", None),  # Should be skipped
        ]

        for sprite_name, expected_id in test_files:
            if expected_id:
                self._create_test_image(self.sprites_dir / sprite_name)
                self._create_test_image(
                    self.artwork_dir / f"pokemon_{expected_id}_artwork.png"
                )
            else:
                self._create_test_image(self.sprites_dir / sprite_name)

        pairs = find_valid_pairs(self.sprites_dir, self.artwork_dir)
        found_ids = [pair["pokemon_id"] for pair in pairs]

        self.assertEqual(len(pairs), 2)  # Only valid formats
        self.assertIn("0001", found_ids)
        self.assertIn("0002", found_ids)

        print_test_result(
            "test_filename_formats",
            True,
            "Correctly parsed different filename formats",
        )


class TestResizeWithPadding(unittest.TestCase):
    """Test suite for resize_with_padding function."""

    def test_resize_smaller_image(self):
        """Test resizing an image smaller than target size."""
        # Create a small test image
        img = Image.new("RGB", (50, 50), color="blue")
        target_size = (100, 100)

        result = resize_with_padding(img, target_size)

        self.assertEqual(result.size, target_size)
        self.assertEqual(result.mode, "RGB")

        print_test_result(
            "test_resize_smaller_image",
            True,
            f"Resized {img.size} to {target_size} with padding",
        )

    def test_resize_larger_image(self):
        """Test resizing an image larger than target size."""
        # Create a large test image
        img = Image.new("RGB", (200, 200), color="green")
        target_size = (100, 100)

        result = resize_with_padding(img, target_size)

        self.assertEqual(result.size, target_size)
        self.assertEqual(result.mode, "RGB")

        print_test_result(
            "test_resize_larger_image",
            True,
            f"Resized {img.size} to {target_size} with thumbnail",
        )

    def test_resize_different_aspect_ratio(self):
        """Test resizing an image with different aspect ratio."""
        # Create a rectangular test image
        img = Image.new("RGB", (200, 100), color="yellow")
        target_size = (150, 150)

        result = resize_with_padding(img, target_size)

        self.assertEqual(result.size, target_size)
        self.assertEqual(result.mode, "RGB")

        print_test_result(
            "test_resize_different_aspect_ratio",
            True,
            "Correctly handled different aspect ratio",
        )

    def test_square_target_size(self):
        """Test resizing to square target size."""
        img = Image.new("RGB", (80, 120), color="purple")
        target_size = (100, 100)

        result = resize_with_padding(img, target_size)

        self.assertEqual(result.size, target_size)
        # Check that the image is centered by examining padding
        img_array = np.array(result)
        # The original image should be centered with white padding
        self.assertTrue(np.any(img_array == 255))  # White padding present

        print_test_result(
            "test_square_target_size",
            True,
            "Correctly created square image with padding",
        )


class TestProcessImagePairs(unittest.TestCase):
    """Test suite for process_image_pairs function."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.test_dir / "output"

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def _create_test_pair(self, pokemon_id: str):
        """Helper method to create a test image pair."""
        sprite_path = self.test_dir / f"sprite_{pokemon_id}.png"
        artwork_path = self.test_dir / f"artwork_{pokemon_id}.png"

        Image.new("RGB", (64, 64), color="red").save(sprite_path)
        Image.new("RGB", (128, 128), color="blue").save(artwork_path)

        return {
            "pokemon_id": pokemon_id,
            "sprite_path": sprite_path,
            "artwork_path": artwork_path,
        }

    def test_successful_processing(self):
        """Test successful processing of image pairs."""
        pairs = [
            self._create_test_pair("0001"),
            self._create_test_pair("0002"),
        ]

        successful_pairs = process_image_pairs(pairs, self.output_dir)

        self.assertEqual(successful_pairs, 2)

        # Check output directories were created
        input_dir = self.output_dir / "input_artwork"
        target_dir = self.output_dir / "target_sprites"
        self.assertTrue(input_dir.exists())
        self.assertTrue(target_dir.exists())

        # Check processed images exist
        for pair in pairs:
            input_file = input_dir / f"pokemon_{pair['pokemon_id']}.png"
            target_file = target_dir / f"pokemon_{pair['pokemon_id']}.png"
            self.assertTrue(input_file.exists())
            self.assertTrue(target_file.exists())

        print_test_result(
            "test_successful_processing",
            True,
            f"Successfully processed {successful_pairs} image pairs",
        )

    def test_corrupted_image_handling(self):
        """Test handling of corrupted image files."""
        # Create a pair with corrupted artwork
        sprite_path = self.test_dir / "sprite_0001.png"
        artwork_path = self.test_dir / "artwork_0001.png"

        Image.new("RGB", (64, 64), color="red").save(sprite_path)
        artwork_path.write_text("corrupted")  # Invalid image data

        pairs = [
            {
                "pokemon_id": "0001",
                "sprite_path": sprite_path,
                "artwork_path": artwork_path,
            }
        ]

        successful_pairs = process_image_pairs(pairs, self.output_dir)

        self.assertEqual(successful_pairs, 0)
        print_test_result(
            "test_corrupted_image_handling",
            True,
            "Correctly handled corrupted image files",
        )

    def test_custom_target_size(self):
        """Test processing with custom target size."""
        pairs = [self._create_test_pair("0001")]
        target_size = (128, 128)

        successful_pairs = process_image_pairs(
            pairs, self.output_dir, target_size
        )

        self.assertEqual(successful_pairs, 1)

        # Verify output image size
        input_file = self.output_dir / "input_artwork" / "pokemon_0001.png"
        with Image.open(input_file) as img:
            self.assertEqual(img.size, target_size)

        print_test_result(
            "test_custom_target_size",
            True,
            f"Correctly processed with target size {target_size}",
        )


class TestCreateTrainValSplit(unittest.TestCase):
    """Test suite for create_train_val_split function."""

    def test_default_split(self):
        """Test default train/validation split."""
        pairs = [{"pokemon_id": f"{i:04d}"} for i in range(100)]

        train_pairs, val_pairs = create_train_val_split(pairs)

        self.assertEqual(len(train_pairs), 85)  # 85% of 100
        self.assertEqual(len(val_pairs), 15)  # 15% of 100
        self.assertEqual(len(train_pairs) + len(val_pairs), len(pairs))

        print_test_result(
            "test_default_split",
            True,
            f"Split 100 pairs into {len(train_pairs)} train, {len(val_pairs)} val",
        )

    def test_custom_split(self):
        """Test custom train/validation split ratio."""
        pairs = [{"pokemon_id": f"{i:04d}"} for i in range(50)]

        train_pairs, val_pairs = create_train_val_split(
            pairs, validation_split=0.2
        )

        self.assertEqual(len(train_pairs), 40)  # 80% of 50
        self.assertEqual(len(val_pairs), 10)  # 20% of 50

        print_test_result(
            "test_custom_split",
            True,
            f"Custom split: {len(train_pairs)} train, {len(val_pairs)} val",
        )

    def test_small_dataset_split(self):
        """Test split with very small dataset."""
        pairs = [{"pokemon_id": "0001"}, {"pokemon_id": "0002"}]

        train_pairs, val_pairs = create_train_val_split(pairs)

        self.assertEqual(len(train_pairs) + len(val_pairs), 2)
        # At least one item should be in training set
        self.assertGreaterEqual(len(train_pairs), 1)

        print_test_result(
            "test_small_dataset_split",
            True,
            "Correctly handled small dataset split",
        )


class TestSaveDatasetMetadata(unittest.TestCase):
    """Test suite for save_dataset_metadata function."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_save_metadata(self):
        """Test saving dataset metadata to JSON file."""
        metadata = {
            "total_pairs": 100,
            "train_pairs": 85,
            "val_pairs": 15,
            "image_size": [256, 256],
            "created_at": "2025-08-02",
        }

        save_dataset_metadata(self.test_dir, metadata)

        metadata_file = self.test_dir / "dataset_info.json"
        self.assertTrue(metadata_file.exists())

        # Verify content
        with open(metadata_file, "r") as f:
            loaded_metadata = json.load(f)

        self.assertEqual(loaded_metadata, metadata)
        print_test_result(
            "test_save_metadata",
            True,
            "Successfully saved and verified metadata",
        )

    def test_empty_metadata(self):
        """Test saving empty metadata."""
        metadata = {}

        save_dataset_metadata(self.test_dir, metadata)

        metadata_file = self.test_dir / "dataset_info.json"
        self.assertTrue(metadata_file.exists())

        with open(metadata_file, "r") as f:
            loaded_metadata = json.load(f)

        self.assertEqual(loaded_metadata, {})
        print_test_result(
            "test_empty_metadata", True, "Successfully handled empty metadata"
        )


class TestCalculateImageStats(unittest.TestCase):
    """Test suite for calculate_image_stats function."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def _create_test_images(self, count: int):
        """Helper method to create test images."""
        image_files = []
        for i in range(count):
            img_path = self.test_dir / f"test_{i}.png"
            # Create images with different colors for variation
            color = (i * 30 % 255, (i * 60) % 255, (i * 90) % 255)
            img = Image.new("RGB", (64, 64), color=color)
            img.save(img_path)
            image_files.append(img_path)
        return image_files

    def test_calculate_stats_multiple_images(self):
        """Test calculating statistics for multiple images."""
        image_files = self._create_test_images(10)

        stats = calculate_image_stats(image_files)

        # Verify all expected keys are present
        self.assertIn("brightness", stats)
        self.assertIn("contrast", stats)
        self.assertIn("color_complexity", stats)
        self.assertIn("sample_size", stats)

        # Verify nested structure
        self.assertIn("mean", stats["brightness"])
        self.assertIn("std", stats["brightness"])

        self.assertEqual(stats["sample_size"], 10)
        print_test_result(
            "test_calculate_stats_multiple_images",
            True,
            f"Calculated stats for {stats['sample_size']} images",
        )

    def test_calculate_stats_large_sample(self):
        """Test statistics calculation with large sample (should be limited)."""
        image_files = self._create_test_images(60)  # More than 50

        stats = calculate_image_stats(image_files)

        # Should be limited to 50 samples
        self.assertEqual(stats["sample_size"], 50)
        print_test_result(
            "test_calculate_stats_large_sample",
            True,
            f"Correctly limited to {stats['sample_size']} samples",
        )

    def test_calculate_stats_empty_list(self):
        """Test statistics calculation with empty image list."""
        stats = calculate_image_stats([])

        self.assertEqual(stats["brightness"]["mean"], 0)
        self.assertEqual(stats["contrast"]["mean"], 0)
        self.assertEqual(stats["color_complexity"]["mean"], 0)
        self.assertEqual(stats["sample_size"], 0)

        print_test_result(
            "test_calculate_stats_empty_list",
            True,
            "Correctly handled empty image list",
        )

    def test_calculate_stats_corrupted_images(self):
        """Test statistics calculation with some corrupted images."""
        # Create mix of valid and corrupted images
        valid_files = self._create_test_images(3)
        corrupted_file = self.test_dir / "corrupted.png"
        corrupted_file.write_text("not an image")

        all_files = valid_files + [corrupted_file]

        stats = calculate_image_stats(all_files)

        # Should only process valid images
        self.assertEqual(stats["sample_size"], 3)
        print_test_result(
            "test_calculate_stats_corrupted_images",
            True,
            "Correctly skipped corrupted images",
        )


if __name__ == "__main__":
    print(f"\n{TestColors.BLUE}{TestColors.BOLD}{'='*70}{TestColors.RESET}")
    print(
        f"{TestColors.BLUE}{TestColors.BOLD}Running Unit Tests for Data Loaders Module{TestColors.RESET}"
    )
    print(f"{TestColors.BLUE}{TestColors.BOLD}{'='*70}{TestColors.RESET}\n")

    # Run tests with detailed output
    unittest.main(verbosity=2, exit=False)

    print(f"\n{TestColors.BLUE}{TestColors.BOLD}{'='*70}{TestColors.RESET}")
    print(
        f"{TestColors.BLUE}{TestColors.BOLD}Unit Tests Completed{TestColors.RESET}"
    )
    print(f"{TestColors.BLUE}{TestColors.BOLD}{'='*70}{TestColors.RESET}")
