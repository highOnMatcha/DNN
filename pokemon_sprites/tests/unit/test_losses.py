"""
Unit tests for loss functions.

This module provides comprehensive tests for all loss function implementations
to ensure correct functionality, error handling, and performance.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from src.losses import (
    AntiBlurLoss,
    CombinedAntiBlurLoss,
    ImprovedPerceptualLoss,
    PixelArtLoss,
    create_loss,
    get_available_losses,
)


class TestLossFunctions(unittest.TestCase):
    """Test cases for individual loss functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.batch_size = 2
        self.channels = 3
        self.height = 64
        self.width = 64

        # Create test tensors
        self.generated = torch.randn(
            self.batch_size,
            self.channels,
            self.height,
            self.width,
            device=self.device,
            requires_grad=True,
        )
        self.target = torch.randn(
            self.batch_size,
            self.channels,
            self.height,
            self.width,
            device=self.device,
        )

    def test_anti_blur_loss_forward(self):
        """Test AntiBlurLoss forward pass."""
        loss_fn = AntiBlurLoss(alpha=1.0, beta=0.5)

        # Test forward pass
        loss = loss_fn(self.generated, self.target)

        # Assertions
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar loss
        self.assertGreater(loss.item(), 0)  # Positive loss

        # Test gradient computation
        loss.backward()
        self.assertIsNotNone(self.generated.grad)

    def test_pixel_art_loss_forward(self):
        """Test PixelArtLoss forward pass."""
        loss_fn = PixelArtLoss(sharpness_weight=2.0, color_weight=1.0)

        # Test forward pass
        loss = loss_fn(self.generated, self.target)

        # Assertions
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0)

        # Test gradient computation
        loss.backward()
        self.assertIsNotNone(self.generated.grad)

    @patch("torchvision.models.vgg16")
    def test_perceptual_loss_forward(self, mock_vgg):
        """Test ImprovedPerceptualLoss forward pass."""
        # Mock VGG model
        mock_features = MagicMock()
        mock_children = [nn.ReLU() for _ in range(25)]  # Mock VGG layers
        mock_features.children.return_value = mock_children
        mock_vgg.return_value.features = mock_features

        loss_fn = ImprovedPerceptualLoss()

        # Test forward pass
        loss = loss_fn(self.generated, self.target)

        # Assertions
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)

    def test_combined_loss_forward(self):
        """Test CombinedAntiBlurLoss forward pass."""
        with patch("torchvision.models.vgg16"):
            loss_fn = CombinedAntiBlurLoss(
                l1_weight=100.0,
                anti_blur_weight=10.0,
                pixel_art_weight=5.0,
                perceptual_weight=1.0,
            )

            # Test forward pass
            total_loss, components = loss_fn(self.generated, self.target)

            # Assertions
            self.assertIsInstance(total_loss, torch.Tensor)
            self.assertIsInstance(components, dict)
            self.assertEqual(total_loss.dim(), 0)

            # Check all components are present
            expected_keys = [
                "l1_loss",
                "anti_blur_loss",
                "pixel_art_loss",
                "perceptual_loss",
                "total_loss",
            ]
            for key in expected_keys:
                self.assertIn(key, components)
                self.assertIsInstance(components[key], float)

    def test_input_validation(self):
        """Test input validation for loss functions."""
        loss_fn = AntiBlurLoss()

        # Test shape mismatch
        wrong_shape = torch.randn(1, 3, 32, 32)
        with self.assertRaises(ValueError):
            loss_fn(self.generated, wrong_shape)

        # Test wrong dimensions
        wrong_dims = torch.randn(3, 64, 64)  # Missing batch dimension
        with self.assertRaises(ValueError):
            loss_fn(wrong_dims, self.target)

    def test_edge_cases(self):
        """Test edge cases for loss functions."""
        loss_fn = AntiBlurLoss()

        # Test with identical inputs
        identical = self.target.clone()
        loss = loss_fn(identical, self.target)
        self.assertGreaterEqual(loss.item(), 0)  # Should be close to zero

        # Test with extreme values
        extreme_gen = torch.ones_like(self.target)
        extreme_target = torch.zeros_like(self.target)
        loss = loss_fn(extreme_gen, extreme_target)
        self.assertGreater(loss.item(), 0)


class TestLossFactory(unittest.TestCase):
    """Test cases for loss factory functions."""

    def test_create_loss_valid(self):
        """Test creating losses with valid names."""
        # Test non-VGG losses
        simple_losses = ["anti_blur", "pixel_art"]
        for loss_name in simple_losses:
            loss = create_loss(loss_name)
            self.assertIsNotNone(loss)

        # Test VGG-based losses with mocking
        with patch("torchvision.models.vgg16"):
            for loss_name in ["perceptual", "combined"]:
                loss = create_loss(loss_name)
                self.assertIsNotNone(loss)

    def test_create_loss_invalid(self):
        """Test creating loss with invalid name."""
        with self.assertRaises(ValueError):
            create_loss("invalid_loss_name")

    def test_create_loss_with_params(self):
        """Test creating loss with custom parameters."""
        loss = create_loss("anti_blur", alpha=2.0, beta=1.0)
        self.assertEqual(loss.alpha, 2.0)
        self.assertEqual(loss.beta, 1.0)

    def test_get_available_losses(self):
        """Test getting available loss names."""
        losses = get_available_losses()
        self.assertIsInstance(losses, list)
        self.assertGreater(len(losses), 0)

        # Check expected loss types are present
        expected = ["anti_blur", "pixel_art", "perceptual", "combined"]
        for expected_loss in expected:
            self.assertIn(expected_loss, losses)


class TestLossIntegration(unittest.TestCase):
    """Integration tests for loss functions."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.device = torch.device("cpu")
        self.generated = torch.randn(1, 3, 64, 64, requires_grad=True)
        self.target = torch.randn(1, 3, 64, 64)

    def test_loss_backward_compatibility(self):
        """Test that loss functions maintain backward compatibility."""
        # Test that we can import losses both ways
        from src.losses import AntiBlurLoss as NewAntiBlur
        from src.losses.anti_blur_loss import AntiBlurLoss as DirectAntiBlur

        self.assertEqual(NewAntiBlur, DirectAntiBlur)

    @patch("torchvision.models.vgg16")
    def test_training_loop_simulation(self, mock_vgg):
        """Simulate a basic training loop with loss functions."""
        # Mock VGG for perceptual loss
        mock_features = MagicMock()
        mock_vgg.return_value.features = mock_features

        loss_fn = create_loss("combined")
        optimizer = torch.optim.Adam([self.generated], lr=0.001)

        # Simulate training steps
        for step in range(3):
            optimizer.zero_grad()
            total_loss, components = loss_fn(self.generated, self.target)
            total_loss.backward()
            optimizer.step()

            # Verify loss computation worked
            self.assertIsInstance(total_loss.item(), float)
            self.assertIsInstance(components, dict)


if __name__ == "__main__":
    unittest.main()
