"""
Test suite for LearningRateFinder component.
"""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# [INITIALIZATION] Add src directory to Python path for module imports
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from core.logging_config import get_logger

    logger = get_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


class MockDiscriminator(nn.Module):
    """Mock discriminator that accepts two inputs like Pix2PixDiscriminator."""

    def __init__(self):
        super().__init__()
        # Process concatenated input (4+4=8 channels for ARGB) -> 1 output
        self.model = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1),  # 8 channels input for ARGB
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_img, target_img):
        # Concatenate inputs along channel dimension like real discriminator
        combined = torch.cat([input_img, target_img], dim=1)
        return self.model(combined)


class MockGenerator(nn.Module):
    """Mock generator that outputs images of the right size."""

    def __init__(self):
        super().__init__()
        # Simple conv layers that maintain spatial dimensions for ARGB
        self.model = nn.Sequential(
            nn.Conv2d(
                4, 4, 1, 1, 0
            ),  # Convert 4 ARGB channels to 4 ARGB channels
        )

    def forward(self, x):
        return self.model(x)


class TestLearningRateFinderBasic(unittest.TestCase):
    """Basic tests for LearningRateFinder."""

    def setUp(self):
        """Set up test environment."""
        self.generator = MockGenerator()
        self.discriminator = MockDiscriminator()

    def test_lr_finder_initialization(self):
        """Test LearningRateFinder initialization."""
        logger.info("[TEST] Testing LearningRateFinder initialization")

        try:
            from optimizers.lr_finder import LearningRateFinder

            lr_finder = LearningRateFinder(
                self.generator, self.discriminator, device="cpu"
            )

            # Verify initialization
            self.assertEqual(lr_finder.device, "cpu")
            self.assertIsNotNone(lr_finder.generator)
            self.assertIsNotNone(lr_finder.discriminator)
            self.assertIsInstance(lr_finder.history, dict)
            self.assertIn("lr", lr_finder.history)
            self.assertIn("loss", lr_finder.history)

            logger.info("[SUCCESS] LearningRateFinder initialization works")
        except Exception as e:
            logger.error(
                f"[FAIL] LearningRateFinder initialization failed: {e}"
            )
            self.fail(f"LearningRateFinder initialization failed: {e}")

    def test_create_synthetic_dataloader(self):
        """Test the synthetic dataloader creation."""
        logger.info("[TEST] Testing synthetic dataloader creation")

        try:
            from optimizers.lr_finder import create_synthetic_dataloader

            dataloader = create_synthetic_dataloader(
                batch_size=2, num_samples=10
            )

            # Verify dataloader structure
            self.assertIsInstance(dataloader, DataLoader)

            # Test getting a batch
            artwork, sprite = next(iter(dataloader))
            self.assertEqual(artwork.shape[0], 2)  # batch_size
            self.assertEqual(artwork.shape[1], 4)  # ARGB input channels
            self.assertEqual(sprite.shape[1], 4)  # ARGB output channels

            logger.info("[SUCCESS] Synthetic dataloader creation works")
        except Exception as e:
            logger.error(f"[FAIL] Synthetic dataloader creation failed: {e}")
            self.fail(f"Synthetic dataloader creation failed: {e}")

    def test_lr_finder_find_optimal_lr(self):
        """Test the find_optimal_lr method."""
        logger.info("[TEST] Testing find_optimal_lr method")

        try:
            from optimizers.lr_finder import (
                LearningRateFinder,
                create_synthetic_dataloader,
            )

            lr_finder = LearningRateFinder(
                self.generator, self.discriminator, device="cpu"
            )

            # Create small dataloader for testing
            dataloader = create_synthetic_dataloader(
                batch_size=2, num_samples=8
            )

            # Test with conservative parameters
            results = lr_finder.find_optimal_lr(
                dataloader,
                start_lr=1e-5,
                end_lr=1e-3,
                num_iterations=5,  # Very small for testing
                lambda_l1=10.0,
            )

            # Check result structure
            self.assertIsInstance(results, dict)
            self.assertIn("optimal_lr", results)
            self.assertIn("min_lr", results)
            self.assertIn("max_lr", results)
            self.assertIn("analysis_method", results)

            # Check that learning rates are reasonable
            self.assertGreater(results["optimal_lr"], 0)
            self.assertGreater(results["min_lr"], 0)
            self.assertGreater(results["max_lr"], 0)

            # Check history was recorded
            self.assertGreater(len(lr_finder.history["lr"]), 0)
            self.assertGreater(len(lr_finder.history["loss"]), 0)

            logger.info("[SUCCESS] find_optimal_lr method works")
        except Exception as e:
            logger.error(f"[FAIL] find_optimal_lr method failed: {e}")
            self.fail(f"find_optimal_lr method failed: {e}")

    def test_analyze_results_method(self):
        """Test the _analyze_results method."""
        logger.info("[TEST] Testing _analyze_results method")

        try:
            from optimizers.lr_finder import LearningRateFinder

            lr_finder = LearningRateFinder(
                self.generator, self.discriminator, device="cpu"
            )

            # Test with mock data
            lrs = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
            losses = [10.0, 8.0, 5.0, 7.0, 15.0]  # minimum at lr = 1e-4

            results = lr_finder._analyze_results(lrs, losses)

            # Check result structure
            self.assertIsInstance(results, dict)
            self.assertIn("optimal_lr", results)
            self.assertIn("min_lr", results)
            self.assertIn("max_lr", results)
            self.assertIn("min_loss_lr", results)
            self.assertIn("analysis_method", results)

            # Check analysis logic
            self.assertEqual(
                results["min_loss_lr"], 1e-4
            )  # Should find minimum
            self.assertEqual(
                results["optimal_lr"], 1e-4 / 3.0
            )  # Smith's method
            self.assertEqual(results["analysis_method"], "smith_et_al_2017")

            logger.info("[SUCCESS] _analyze_results method works")
        except Exception as e:
            logger.error(f"[FAIL] _analyze_results method failed: {e}")
            self.fail(f"_analyze_results method failed: {e}")

    def test_analyze_results_fallback(self):
        """Test the _analyze_results fallback for insufficient data."""
        logger.info("[TEST] Testing _analyze_results fallback")

        try:
            from optimizers.lr_finder import LearningRateFinder

            lr_finder = LearningRateFinder(
                self.generator, self.discriminator, device="cpu"
            )

            # Test with insufficient data
            lrs = [1e-5, 5e-5]  # Less than 5 points
            losses = [10.0, 8.0]

            results = lr_finder._analyze_results(lrs, losses)

            # Check fallback behavior
            self.assertIsInstance(results, dict)
            self.assertEqual(results["optimal_lr"], 1e-4)  # Fallback value
            self.assertEqual(results["analysis_method"], "fallback")

            logger.info("[SUCCESS] _analyze_results fallback works")
        except Exception as e:
            logger.error(f"[FAIL] _analyze_results fallback failed: {e}")
            self.fail(f"_analyze_results fallback failed: {e}")



