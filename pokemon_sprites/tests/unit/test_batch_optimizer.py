"""
Simple focused test suite for BatchSizeOptimizer component.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

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


class TestBatchSizeOptimizerBasic(unittest.TestCase):
    """Basic tests for BatchSizeOptimizer."""

    def setUp(self):
        """Set up test environment."""
        self.generator = MockGenerator()
        self.discriminator = MockDiscriminator()

    def test_batch_optimizer_initialization(self):
        """Test BatchSizeOptimizer initialization."""
        logger.info("[TEST] Testing BatchSizeOptimizer initialization")

        try:
            from optimizers.batch_optimizer import BatchSizeOptimizer

            optimizer = BatchSizeOptimizer(
                self.generator, self.discriminator, device="cpu"
            )

            # Verify initialization
            self.assertEqual(optimizer.device, "cpu")
            self.assertIsNotNone(optimizer.generator)
            self.assertIsNotNone(optimizer.discriminator)

            logger.info("[SUCCESS] BatchSizeOptimizer initialization works")
        except Exception as e:
            logger.error(
                f"[FAIL] BatchSizeOptimizer initialization failed: {e}"
            )
            self.fail(f"BatchSizeOptimizer initialization failed: {e}")

    def test_test_single_batch_size_method(self):
        """Test the _test_single_batch_size method."""
        logger.info("[TEST] Testing _test_single_batch_size method")

        try:
            from optimizers.batch_optimizer import BatchSizeOptimizer

            optimizer = BatchSizeOptimizer(
                self.generator, self.discriminator, device="cpu"
            )

            # Patch the CUDA issue in the original code for testing
            with patch("torch.cuda.reset_peak_memory_stats"):
                with patch("torch.cuda.max_memory_allocated") as mock_max:
                    mock_max.return_value = 100 * 1024 * 1024  # 100MB

                    # Test with small sizes for ARGB
                    success, memory_used, training_time = (
                        optimizer._test_single_batch_size(
                            batch_size=1,
                            input_size=(4, 32, 32),  # ARGB input
                            output_size=(4, 32, 32),  # ARGB output
                        )
                    )

                    # Should succeed and return valid results
                    self.assertTrue(success)
                    self.assertIsInstance(memory_used, (int, float))
                    self.assertIsInstance(training_time, (int, float))
                    self.assertGreaterEqual(memory_used, 0)
                    self.assertGreaterEqual(training_time, 0)

            logger.info("[SUCCESS] _test_single_batch_size method works")
        except Exception as e:
            logger.error(f"[FAIL] _test_single_batch_size method failed: {e}")
            self.fail(f"_test_single_batch_size method failed: {e}")

    def test_find_optimal_batch_size_method(self):
        """Test the find_optimal_batch_size method."""
        logger.info("[TEST] Testing find_optimal_batch_size method")

        try:
            from optimizers.batch_optimizer import BatchSizeOptimizer

            optimizer = BatchSizeOptimizer(
                self.generator, self.discriminator, device="cpu"
            )

            # Test with conservative parameters
            results = optimizer.find_optimal_batch_size(
                initial_max_batch_size=64,  # Test larger batch sizes for 16GB GPU optimization
                input_size=(3, 32, 32),
                output_size=(4, 32, 32),
            )

            # Check result structure
            self.assertIsInstance(results, dict)
            self.assertIn("recommended_batch_size", results)
            self.assertIn("optimal_batch_size", results)
            self.assertIn("max_stable_batch_size", results)

            # Check that batch sizes are reasonable
            self.assertGreater(results["recommended_batch_size"], 0)
            self.assertGreater(results["optimal_batch_size"], 0)
            self.assertGreater(results["max_stable_batch_size"], 0)

            logger.info("[SUCCESS] find_optimal_batch_size method works")
        except Exception as e:
            logger.error(f"[FAIL] find_optimal_batch_size method failed: {e}")
            self.fail(f"find_optimal_batch_size method failed: {e}")


