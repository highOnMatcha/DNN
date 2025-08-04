"""
Unit tests for the training sprite_generator module.

Tests for sprite generation utilities used after training completion,
including missing sprite detection and generation workflows.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.logging_config import get_logger
from training.sprite_generator import (
    create_comparison_grid,
    find_missing_sprites,
    generate_missing_sprites,
    log_generated_sprites_to_wandb,
)

logger = get_logger(__name__)


class TestSpriteGeneratorUtilities(unittest.TestCase):
    """Test cases for sprite generation utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.artwork_dir = Path(self.temp_dir) / "artwork"
        self.sprite_dir = Path(self.temp_dir) / "sprites"

        # Create directories
        self.artwork_dir.mkdir(parents=True)
        self.sprite_dir.mkdir(parents=True)

        # Create test artwork files
        for i in [1, 2, 3, 5, 7]:  # Missing 4, 6, 8
            artwork_path = self.artwork_dir / f"pokemon_{i:03d}_artwork.png"
            test_image = Image.new("RGB", (256, 256), (255, 0, 0))
            test_image.save(artwork_path)

        # Create some existing sprites
        for i in [1, 2, 5]:  # Missing sprites for 3, 7
            sprite_path = self.sprite_dir / f"pokemon_{i:03d}_bw.png"
            test_image = Image.new("RGB", (96, 96), (0, 255, 0))
            test_image.save(sprite_path)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_find_missing_sprites(self):
        """Test finding missing sprites."""
        logger.info("[TEST] Testing find_missing_sprites function")

        try:
            missing = find_missing_sprites(self.artwork_dir, self.sprite_dir)

            # Should find artwork files that don't have corresponding sprites
            self.assertIsInstance(missing, list)

            # Check that we found the expected missing sprites
            missing_ids = []
            for artwork_path in missing:
                filename = artwork_path.stem
                if "_artwork" in filename:
                    pokemon_id = filename.split("_")[1]
                    missing_ids.append(int(pokemon_id))

            # Should include IDs 3 and 7 (have artwork but no sprites)
            self.assertIn(3, missing_ids)
            self.assertIn(7, missing_ids)

            logger.info(f"[SUCCESS] Found {len(missing)} missing sprites")

        except Exception as e:
            logger.error(f"[FAIL] find_missing_sprites failed: {e}")
            self.fail(f"find_missing_sprites failed: {e}")

    def test_find_missing_sprites_nonexistent_dirs(self):
        """Test find_missing_sprites with nonexistent directories."""
        logger.info(
            "[TEST] Testing find_missing_sprites with nonexistent directories"
        )

        try:
            nonexistent_dir = Path(self.temp_dir) / "nonexistent"
            missing = find_missing_sprites(nonexistent_dir, self.sprite_dir)
            self.assertEqual(missing, [])

            missing = find_missing_sprites(self.artwork_dir, nonexistent_dir)
            self.assertEqual(missing, [])

            logger.info("[SUCCESS] Handled nonexistent directories correctly")

        except Exception as e:
            logger.error(f"[FAIL] Nonexistent directory handling failed: {e}")
            self.fail(f"Nonexistent directory handling failed: {e}")

    @patch("training.sprite_generator.logger")
    def test_find_missing_sprites_logging(self, mock_logger):
        """Test logging in find_missing_sprites."""
        logger.info("[TEST] Testing find_missing_sprites logging")

        try:
            find_missing_sprites(self.artwork_dir, self.sprite_dir)

            # Verify that appropriate logging occurred
            # The function should log information about missing sprites
            self.assertTrue(
                mock_logger.info.called or mock_logger.warning.called
            )

            logger.info("[SUCCESS] Logging behavior verified")

        except Exception as e:
            logger.error(f"[FAIL] Logging test failed: {e}")
            self.fail(f"Logging test failed: {e}")

    @patch("training.sprite_generator.wandb")
    def test_log_generated_sprites_to_wandb(self, mock_wandb):
        """Test logging generation results to wandb."""
        logger.info("[TEST] Testing log_generated_sprites_to_wandb function")

        try:
            # Mock wandb setup
            mock_wandb_run = Mock()
            mock_wandb_run.log = Mock()
            mock_wandb.Image = Mock(return_value="mock_image")

            # Create a test output directory with sprites
            output_dir = Path(self.temp_dir) / "output"
            output_dir.mkdir(exist_ok=True)
            sprite_file = output_dir / "pokemon_001_bw.png"
            test_image = Image.new("RGB", (96, 96), (0, 255, 0))
            test_image.save(sprite_file)

            log_generated_sprites_to_wandb(mock_wandb_run, output_dir)

            # Verify wandb logging was called
            mock_wandb_run.log.assert_called()

            logger.info("[SUCCESS] log_generated_sprites_to_wandb test passed")

        except Exception as e:
            logger.error(f"[FAIL] log_generated_sprites_to_wandb failed: {e}")
            self.fail(f"log_generated_sprites_to_wandb failed: {e}")

    @patch("training.sprite_generator.Image")
    def test_create_comparison_grid(self, mock_image_module):
        """Test sprite comparison grid creation."""
        logger.info("[TEST] Testing create_comparison_grid function")

        try:
            # Mock PIL Image operations
            mock_img = Mock()
            mock_img.size = (256, 256)
            mock_img.resize = Mock(return_value=mock_img)
            mock_image_module.open = Mock(return_value=mock_img)
            mock_image_module.new = Mock(return_value=mock_img)

            # Test file paths
            artwork_paths = [
                self.artwork_dir / "pokemon_001_artwork.png",
                self.artwork_dir / "pokemon_002_artwork.png",
            ]
            sprite_paths = [
                self.sprite_dir / "pokemon_001_bw.png",
                self.sprite_dir / "pokemon_002_bw.png",
            ]

            output_path = Path(self.temp_dir) / "comparison_grid.png"

            result = create_comparison_grid(
                artwork_paths, sprite_paths, output_path
            )

            # Function might return None or the path - both are valid
            self.assertTrue(result is None or result == output_path)

            logger.info("[SUCCESS] create_comparison_grid test passed")

        except Exception as e:
            logger.error(f"[FAIL] create_comparison_grid failed: {e}")
            self.fail(f"create_comparison_grid failed: {e}")


class TestGenerateMissingSprites(unittest.TestCase):
    """Test cases for generate_missing_sprites function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.artwork_dir = Path(self.temp_dir) / "artwork"
        self.sprite_dir = Path(self.temp_dir) / "sprites"
        self.output_dir = Path(self.temp_dir) / "output"

        # Create directories
        for dir_path in [self.artwork_dir, self.sprite_dir, self.output_dir]:
            dir_path.mkdir(parents=True)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_missing_sprites(self):
        """Test generate_missing_sprites function."""
        logger.info("[TEST] Testing generate_missing_sprites function")

        try:
            # Create test artwork file
            artwork_path = self.artwork_dir / "pokemon_001_artwork.png"
            test_image = Image.new("RGB", (256, 256), (255, 0, 0))
            test_image.save(artwork_path)

            # Mock trainer
            mock_trainer = Mock()
            mock_trainer.generate_sprite = Mock(
                return_value={"base": Image.new("RGB", (96, 96), (0, 255, 0))}
            )

            # Test generation
            result_dir = generate_missing_sprites(
                trainer=mock_trainer,
                missing_artwork=[artwork_path],
                output_dir=self.output_dir,
            )

            # Verify results
            self.assertIsInstance(result_dir, Path)
            self.assertEqual(result_dir, self.output_dir)

            logger.info("[SUCCESS] generate_missing_sprites test passed")

        except Exception as e:
            logger.error(f"[FAIL] generate_missing_sprites failed: {e}")
            self.fail(f"generate_missing_sprites failed: {e}")

    def test_generate_missing_sprites_error_handling(self):
        """Test error handling in generate_missing_sprites."""
        logger.info("[TEST] Testing generate_missing_sprites error handling")

        try:
            # Test with None trainer
            result_dir = generate_missing_sprites(
                trainer=None, missing_artwork=[], output_dir=self.output_dir
            )

            # Should handle error gracefully and return output dir
            self.assertIsInstance(result_dir, Path)
            self.assertEqual(result_dir, self.output_dir)

            logger.info("[SUCCESS] Error handling test passed")

        except Exception as e:
            logger.error(f"[FAIL] Error handling test failed: {e}")
            self.fail(f"Error handling test failed: {e}")


class TestSpriteGeneratorIntegration(unittest.TestCase):
    """Integration tests for sprite generator module."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_missing_sprite_workflow(self):
        """Test the complete missing sprite workflow."""
        logger.info("[TEST] Testing full missing sprite workflow")

        try:
            # Setup directory structure
            artwork_dir = Path(self.temp_dir) / "artwork"
            sprite_dir = Path(self.temp_dir) / "sprites"
            output_dir = Path(self.temp_dir) / "output"

            for dir_path in [artwork_dir, sprite_dir, output_dir]:
                dir_path.mkdir(parents=True)

            # Create test artwork (but no corresponding sprites)
            for i in [1, 2, 3]:
                artwork_path = artwork_dir / f"pokemon_{i:03d}_artwork.png"
                test_image = Image.new("RGB", (256, 256), (255, 0, 0))
                test_image.save(artwork_path)

            # Step 1: Find missing sprites
            missing = find_missing_sprites(artwork_dir, sprite_dir)
            self.assertEqual(len(missing), 3)  # All are missing

            # Step 2: Verify missing sprites found successfully
            # (In a real workflow, these would be processed by generation pipeline)

            # Step 3: Log results (without actual wandb)
            with patch("training.sprite_generator.wandb") as mock_wandb:
                mock_wandb_run = Mock()
                mock_wandb_run.log = Mock()
                mock_wandb.Image = Mock(return_value="mock_image")

                # Create a sprite file for logging
                sprite_file = output_dir / "pokemon_001_bw.png"
                test_image = Image.new("RGB", (96, 96), (0, 255, 0))
                test_image.save(sprite_file)

                log_generated_sprites_to_wandb(mock_wandb_run, output_dir)
                mock_wandb_run.log.assert_called()

            logger.info("[SUCCESS] Full workflow test passed")

        except Exception as e:
            logger.error(f"[FAIL] Full workflow test failed: {e}")
            self.fail(f"Full workflow test failed: {e}")

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        logger.info("[TEST] Testing edge cases")

        try:
            # Test with empty directories
            empty_dir = Path(self.temp_dir) / "empty"
            empty_dir.mkdir()

            missing = find_missing_sprites(empty_dir, empty_dir)
            self.assertEqual(missing, [])

            # Test with malformed filenames
            malformed_dir = Path(self.temp_dir) / "malformed"
            malformed_dir.mkdir()

            # Create files with unexpected naming
            test_image = Image.new("RGB", (64, 64), (0, 0, 255))
            weird_file = malformed_dir / "not_pokemon_format.png"
            test_image.save(weird_file)

            missing = find_missing_sprites(malformed_dir, empty_dir)
            # Should handle malformed filenames gracefully
            self.assertIsInstance(missing, list)

            logger.info("[SUCCESS] Edge case tests passed")

        except Exception as e:
            logger.error(f"[FAIL] Edge case tests failed: {e}")
            self.fail(f"Edge case tests failed: {e}")


if __name__ == "__main__":
    unittest.main()
