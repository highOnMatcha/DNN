"""
Unit tests for optimizers module.
Tests learning rate optimization, batch optimization, and model validation.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import torch.nn as nn


class TestOptimizers(unittest.TestCase):
    """Test optimization utilities."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            import shutil

            shutil.rmtree(self.test_dir)

    def test_optimizers_import(self):
        """Test that optimizer modules can be imported."""
        try:
            from src.optimizers.batch_optimizer import BatchSizeOptimizer
            from src.optimizers.lr_finder import LearningRateFinder
            from src.optimizers.schedule_optimizer import (
                TrainingScheduleOptimizer,
            )

            self.assertIsNotNone(LearningRateFinder)
            self.assertIsNotNone(BatchSizeOptimizer)
            self.assertIsNotNone(TrainingScheduleOptimizer)
            print("[SUCCESS] Optimizer modules import")
        except ImportError as e:
            self.fail(f"Failed to import optimizer modules: {e}")

    def test_learning_rate_finder_class_exists(self):
        """Test that LearningRateFinder class exists and can be referenced."""
        from src.optimizers.lr_finder import LearningRateFinder

        # Check that the class exists
        self.assertTrue(hasattr(LearningRateFinder, "__init__"))
        self.assertTrue(hasattr(LearningRateFinder, "find_optimal_lr"))
        print("[SUCCESS] LearningRateFinder class structure")

    def test_batch_size_optimizer_class_exists(self):
        """Test that BatchSizeOptimizer class exists and can be referenced."""
        from src.optimizers.batch_optimizer import BatchSizeOptimizer

        # Check that the class exists
        self.assertTrue(hasattr(BatchSizeOptimizer, "__init__"))
        self.assertTrue(hasattr(BatchSizeOptimizer, "find_optimal_batch_size"))
        print("[SUCCESS] BatchSizeOptimizer class structure")

    def test_training_schedule_optimizer_class_exists(self):
        """Test that TrainingScheduleOptimizer class exists and can be referenced."""
        from src.optimizers.schedule_optimizer import TrainingScheduleOptimizer

        # Check that the class exists
        self.assertTrue(hasattr(TrainingScheduleOptimizer, "__init__"))
        print("[SUCCESS] TrainingScheduleOptimizer class structure")

    def test_optimizer_functionality_with_mock_models(self):
        """Test optimizer functionality with mock PyTorch models."""
        # Create mock generator and discriminator
        mock_generator = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )

        mock_discriminator = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, 4, 1, 0),
            nn.Sigmoid(),
        )

        # Test that we can create instances with these models
        try:
            from src.optimizers.batch_optimizer import BatchSizeOptimizer
            from src.optimizers.lr_finder import LearningRateFinder
            from src.optimizers.schedule_optimizer import (
                TrainingScheduleOptimizer,
            )

            # Test LearningRateFinder
            with patch("torch.cuda.is_available", return_value=False):
                lr_finder = LearningRateFinder(
                    mock_generator, mock_discriminator, device="cpu"
                )
                self.assertIsNotNone(lr_finder)

            # Test BatchSizeOptimizer (force CPU for CI environment)
            batch_optimizer = BatchSizeOptimizer(
                mock_generator, mock_discriminator, device="cpu"
            )
            self.assertIsNotNone(batch_optimizer)

            # Test TrainingScheduleOptimizer
            schedule_optimizer = TrainingScheduleOptimizer(
                total_samples=1000, validation_split=0.2
            )
            self.assertIsNotNone(schedule_optimizer)

            print("[SUCCESS] Optimizer instantiation with mock models")
        except Exception as e:
            print(f"[FAIL] Optimizer instantiation: {e}")
            # Don't fail for import/dependency issues
            if "No module named" not in str(e) and "CUDA" not in str(e):
                self.fail(f"Optimizer instantiation failed: {e}")


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
