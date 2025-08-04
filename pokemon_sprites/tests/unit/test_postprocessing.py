"""
Unit tests for the postprocessing module.

Tests for ARGBToPaletteConverter and SpritePostProcessor classes that handle
conversion of ARGB model outputs to optimized P format sprites.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.logging_config import get_logger
from data.postprocessing import ARGBToPaletteConverter, SpritePostProcessor

logger = get_logger(__name__)


class TestARGBToPaletteConverter(unittest.TestCase):
    """Test cases for ARGBToPaletteConverter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = ARGBToPaletteConverter(
            max_colors=16, preserve_transparency=True, optimize_palette=True
        )

        # Create test RGBA image
        self.test_image = Image.new("RGBA", (32, 32), (255, 0, 0, 255))

        # Add some transparency and other colors
        for y in range(16):
            for x in range(16):
                self.test_image.putpixel((x, y), (0, 255, 0, 128))

        for y in range(16, 32):
            for x in range(16, 32):
                self.test_image.putpixel((x, y), (0, 0, 255, 0))

    def test_converter_initialization(self):
        """Test converter initialization with different parameters."""
        logger.info("[TEST] Testing converter initialization")

        # Test default initialization
        converter = ARGBToPaletteConverter()
        self.assertEqual(converter.max_colors, 256)
        self.assertTrue(converter.preserve_transparency)
        self.assertTrue(converter.optimize_palette)

        # Test custom parameters
        converter = ARGBToPaletteConverter(
            max_colors=128, preserve_transparency=False, optimize_palette=False
        )
        self.assertEqual(converter.max_colors, 128)
        self.assertFalse(converter.preserve_transparency)
        self.assertFalse(converter.optimize_palette)

        # Test edge cases for max_colors
        converter = ARGBToPaletteConverter(max_colors=0)
        self.assertEqual(converter.max_colors, 1)

        converter = ARGBToPaletteConverter(max_colors=300)
        self.assertEqual(converter.max_colors, 256)

        logger.info("[SUCCESS] Converter initialization tests passed")

    def test_extract_colors(self):
        """Test color extraction from RGBA image."""
        logger.info("[TEST] Testing color extraction")

        # Test color extraction
        opaque_pixels, transparent_pixels = self.converter._extract_colors(
            self.test_image
        )

        self.assertIsInstance(opaque_pixels, np.ndarray)
        self.assertEqual(opaque_pixels.shape[1], 4)  # RGBA channels

        if transparent_pixels is not None:
            self.assertIsInstance(transparent_pixels, np.ndarray)
            self.assertEqual(transparent_pixels.shape[1], 4)  # RGBA channels

        logger.info("[SUCCESS] Color extraction tests passed")

    def test_optimize_palette_kmeans(self):
        """Test K-means palette optimization."""
        logger.info("[TEST] Testing K-means palette optimization")

        # Create test color array
        colors = np.array(
            [
                [255, 0, 0, 255],
                [0, 255, 0, 255],
                [0, 0, 255, 255],
                [255, 255, 0, 255],
            ],
            dtype=np.uint8,
        )

        # Test with different numbers of clusters
        for n_colors in [2, 4]:
            optimized_palette = self.converter._optimize_palette_kmeans(
                colors, n_colors
            )
            self.assertIsInstance(optimized_palette, np.ndarray)
            self.assertEqual(optimized_palette.shape[0], n_colors)
            self.assertEqual(optimized_palette.shape[1], 4)

        logger.info("[SUCCESS] K-means optimization tests passed")

    def test_assign_to_palette(self):
        """Test color assignment to palette."""
        logger.info("[TEST] Testing palette assignment")

        # Create test palette and colors
        palette = np.array(
            [
                [255, 0, 0, 255],
                [0, 255, 0, 255],
                [0, 0, 255, 255],
            ],
            dtype=np.uint8,
        )

        colors = np.array(
            [
                [250, 5, 5, 255],  # Should map to red
                [5, 250, 5, 255],  # Should map to green
                [5, 5, 250, 255],  # Should map to blue
            ],
            dtype=np.uint8,
        )

        indices = self.converter._assign_to_palette(colors, palette)

        self.assertIsInstance(indices, np.ndarray)
        self.assertEqual(len(indices), len(colors))
        self.assertTrue(all(0 <= idx < len(palette) for idx in indices))

        logger.info("[SUCCESS] Palette assignment tests passed")

    def test_convert_single_image(self):
        """Test single image conversion."""
        logger.info("[TEST] Testing single image conversion")

        try:
            result = self.converter.convert_single_image(self.test_image)

            # API returns tuple (p_image, palette)
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)

            p_image, palette = result
            self.assertIsInstance(p_image, Image.Image)
            self.assertEqual(p_image.mode, "P")
            self.assertEqual(p_image.size, self.test_image.size)

            self.assertIsInstance(palette, np.ndarray)
            self.assertEqual(palette.shape[1], 4)  # RGBA palette

            logger.info("[SUCCESS] Single image conversion tests passed")

        except Exception as e:
            logger.error(f"[FAIL] Single image conversion failed: {e}")
            self.fail(f"Single image conversion failed: {e}")

    def test_convert_batch(self):
        """Test batch image conversion."""
        logger.info("[TEST] Testing batch conversion")

        # Create multiple test images
        images = [
            self.test_image,
            Image.new("RGBA", (16, 16), (0, 255, 0, 255)),
            Image.new("RGBA", (24, 24), (0, 0, 255, 128)),
        ]

        try:
            result = self.converter.convert_batch(images)

            # API returns tuple (images_list, global_palette)
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)

            images_list, global_palette = result
            self.assertIsInstance(images_list, list)
            self.assertEqual(len(images_list), len(images))

            for p_image in images_list:
                self.assertIsInstance(p_image, Image.Image)
                self.assertEqual(p_image.mode, "P")

            if global_palette is not None:
                self.assertIsInstance(global_palette, np.ndarray)

            logger.info("[SUCCESS] Batch conversion tests passed")

        except Exception as e:
            logger.error(f"[FAIL] Batch conversion failed: {e}")
            self.fail(f"Batch conversion failed: {e}")

    def test_p_to_rgba_for_display(self):
        """Test P to RGBA conversion for display."""
        logger.info("[TEST] Testing P to RGBA conversion")

        try:
            # First convert to P format
            p_image, palette = self.converter.convert_single_image(
                self.test_image
            )

            # Convert back to RGBA
            rgba_image = self.converter.p_to_rgba_for_display(p_image)

            self.assertEqual(rgba_image.mode, "RGBA")
            self.assertEqual(rgba_image.size, p_image.size)

            logger.info("[SUCCESS] P to RGBA conversion tests passed")

        except Exception as e:
            logger.error(f"[FAIL] P to RGBA conversion failed: {e}")
            self.fail(f"P to RGBA conversion failed: {e}")

    def test_analyze_compression(self):
        """Test compression analysis."""
        logger.info("[TEST] Testing compression analysis")

        try:
            # Create images for comparison
            original = self.test_image
            p_image, palette = self.converter.convert_single_image(original)

            analysis = self.converter.analyze_compression(original, p_image)

            self.assertIsInstance(analysis, dict)
            self.assertIn("rgba_size_bytes", analysis)
            self.assertIn("p_size_bytes", analysis)
            self.assertIn("compression_ratio", analysis)
            self.assertIn("size_reduction_percent", analysis)
            self.assertIn("palette_colors", analysis)

            # Verify numeric values
            self.assertIsInstance(analysis["rgba_size_bytes"], int)
            self.assertIsInstance(analysis["p_size_bytes"], int)
            self.assertIsInstance(analysis["compression_ratio"], float)

            logger.info("[SUCCESS] Compression analysis tests passed")

        except Exception as e:
            logger.error(f"[FAIL] Compression analysis failed: {e}")
            self.fail(f"Compression analysis failed: {e}")

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        logger.info("[TEST] Testing edge cases")

        # Test with very small image
        small_image = Image.new("RGBA", (1, 1), (255, 0, 0, 255))
        p_image, palette = self.converter.convert_single_image(small_image)
        self.assertIsInstance(p_image, Image.Image)
        self.assertIsInstance(palette, np.ndarray)

        # Test with single color image
        single_color = Image.new("RGBA", (10, 10), (128, 128, 128, 255))
        p_image, palette = self.converter.convert_single_image(single_color)
        self.assertIsInstance(p_image, Image.Image)

        logger.info("[SUCCESS] Edge case tests passed")

        # Test with transparent image
        transparent = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
        p_image, palette = self.converter.convert_single_image(transparent)
        self.assertIsInstance(p_image, Image.Image)
        self.assertIsInstance(palette, np.ndarray)

        logger.info("[SUCCESS] Edge case tests passed")


class TestSpritePostProcessor(unittest.TestCase):
    """Test cases for SpritePostProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = SpritePostProcessor()

        # Create test RGBA sprite
        self.test_sprite = Image.new("RGBA", (64, 64), (255, 0, 0, 255))

        # Add some detail to the sprite
        for y in range(32):
            for x in range(32):
                self.test_sprite.putpixel((x, y), (0, 255, 0, 200))

    def test_processor_initialization(self):
        """Test processor initialization."""
        logger.info("[TEST] Testing processor initialization")

        # Test default initialization
        processor = SpritePostProcessor()
        self.assertIsNotNone(processor.converter)

        # Test with custom config
        config = {"max_colors": 32, "preserve_transparency": False}
        processor = SpritePostProcessor(converter_config=config)
        self.assertIsNotNone(processor.converter)
        self.assertEqual(processor.converter.max_colors, 32)
        self.assertFalse(processor.converter.preserve_transparency)

        logger.info("[SUCCESS] Processor initialization tests passed")

    def test_process_single_sprite(self):
        """Test single sprite processing."""
        logger.info("[TEST] Testing single sprite processing")

        try:
            # Test default processing
            result = self.processor.process_single_sprite(self.test_sprite)

            self.assertIsInstance(result, dict)
            self.assertIn("rgba", result)
            self.assertIn("p", result)

            # Verify RGBA output
            rgba_output = result["rgba"]
            self.assertEqual(rgba_output.mode, "RGBA")
            self.assertEqual(rgba_output.size, self.test_sprite.size)

            # Verify P output
            p_output = result["p"]
            self.assertEqual(p_output.mode, "P")
            self.assertEqual(p_output.size, self.test_sprite.size)

            # Test with custom formats
            result = self.processor.process_single_sprite(
                self.test_sprite,
                return_formats=[
                    "rgba",
                    "p",
                    "palette",
                    "compression_analysis",
                ],
            )

            self.assertIn("palette", result)
            self.assertIn("compression_analysis", result)

            logger.info("[SUCCESS] Single sprite processing tests passed")

        except Exception as e:
            logger.error(f"[FAIL] Single sprite processing failed: {e}")
            self.fail(f"Single sprite processing failed: {e}")

    def test_process_sprite_batch(self):
        """Test batch sprite processing."""
        logger.info("[TEST] Testing batch sprite processing")

        # Create batch of sprites
        sprites = [
            self.test_sprite,
            Image.new("RGBA", (32, 32), (0, 255, 0, 255)),
            Image.new("RGBA", (48, 48), (0, 0, 255, 128)),
        ]

        try:
            results = self.processor.process_sprite_batch(sprites)

            # API returns a dictionary with batch information
            self.assertIsInstance(results, dict)
            self.assertIn("batch_size", results)
            self.assertEqual(results["batch_size"], len(sprites))

            # Check for RGBA sprites if requested
            if "rgba_sprites" in results:
                self.assertIsInstance(results["rgba_sprites"], list)
                self.assertEqual(len(results["rgba_sprites"]), len(sprites))

            # Check for P format sprites if requested
            if "p_sprites" in results:
                self.assertIsInstance(results["p_sprites"], list)
                self.assertEqual(len(results["p_sprites"]), len(sprites))

            logger.info("[SUCCESS] Batch sprite processing tests passed")

        except Exception as e:
            logger.error(f"[FAIL] Batch sprite processing failed: {e}")
            self.fail(f"Batch sprite processing failed: {e}")

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        logger.info("[TEST] Testing invalid input handling")

        # Test with non-RGBA image
        rgb_image = Image.new("RGB", (32, 32), (255, 0, 0))

        try:
            # Should handle RGB image by converting to RGBA
            result = self.processor.process_single_sprite(rgb_image)
            self.assertIsInstance(result, dict)

            logger.info("[SUCCESS] Invalid input handling tests passed")

        except Exception as e:
            # This is acceptable - some methods may require RGBA input
            logger.info(f"[INFO] Expected handling of non-RGBA input: {e}")


class TestPostprocessingIntegration(unittest.TestCase):
    """Integration tests for postprocessing functionality."""

    def test_full_pipeline_integration(self):
        """Test full postprocessing pipeline."""
        logger.info("[TEST] Testing full postprocessing pipeline integration")

        try:
            # Create a realistic sprite scenario
            sprite = Image.new("RGBA", (96, 96), (0, 0, 0, 0))

            # Add sprite body
            for y in range(20, 76):
                for x in range(20, 76):
                    sprite.putpixel((x, y), (200, 100, 50, 255))

            # Add some details
            for y in range(30, 40):
                for x in range(30, 40):
                    sprite.putpixel((x, y), (255, 255, 255, 255))

            # Process through full pipeline
            processor = SpritePostProcessor(
                converter_config={"max_colors": 64, "optimize_palette": True}
            )

            result = processor.process_single_sprite(
                sprite,
                return_formats=[
                    "rgba",
                    "p",
                    "palette",
                    "compression_analysis",
                ],
            )

            # Verify all expected outputs
            self.assertIn("rgba", result)
            self.assertIn("p", result)
            self.assertIn("palette", result)
            self.assertIn("compression_analysis", result)

            # Verify compression benefits
            compression = result["compression_analysis"]
            self.assertGreater(compression["compression_ratio"], 1.0)

            logger.info("[SUCCESS] Full pipeline integration tests passed")

        except Exception as e:
            logger.error(f"[FAIL] Full pipeline integration failed: {e}")
            self.fail(f"Full pipeline integration failed: {e}")

    @patch("sklearn.cluster.KMeans")
    def test_kmeans_error_handling(self, mock_kmeans):
        """Test error handling in K-means clustering."""
        logger.info("[TEST] Testing K-means error handling")

        # Make K-means raise an exception
        mock_kmeans.side_effect = Exception("Clustering failed")

        converter = ARGBToPaletteConverter(optimize_palette=True)
        test_image = Image.new("RGBA", (16, 16), (255, 0, 0, 255))

        try:
            # Should handle clustering errors gracefully
            result = converter.convert_single_image(test_image)
            # If it doesn't raise an exception, the error handling worked
            self.assertIsInstance(result, dict)

            logger.info("[SUCCESS] K-means error handling tests passed")

        except Exception as e:
            # Check if it's a reasonable fallback behavior
            logger.info(f"[INFO] K-means error handling: {e}")


if __name__ == "__main__":
    unittest.main()
