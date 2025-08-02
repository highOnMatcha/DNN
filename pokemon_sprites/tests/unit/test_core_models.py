"""
Unit tests for core.models module.
Tests model architectures and neural network components.
"""

import unittest

import torch
import torch.nn as nn

from src.core.models import (
    AttentionBlock,
    ConvBlock,
    Pix2PixDiscriminator,
    Pix2PixGenerator,
    ResBlock,
)


class TestModelArchitectures(unittest.TestCase):
    """Test neural network model architectures."""

    def setUp(self):
        """Set up test environment."""
        self.input_channels = 3
        self.output_channels = 3
        self.batch_size = 1
        self.image_size = 256

    def test_pix2pix_generator_creation(self):
        """Test Pix2PixGenerator model creation."""
        model = Pix2PixGenerator(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
        )

        self.assertIsInstance(model, nn.Module)
        print("[SUCCESS] Pix2PixGenerator model creation")

    def test_pix2pix_generator_forward_pass(self):
        """Test Pix2PixGenerator forward pass."""
        model = Pix2PixGenerator(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
        )

        # Create test input
        test_input = torch.randn(
            self.batch_size,
            self.input_channels,
            self.image_size,
            self.image_size,
        )

        with torch.no_grad():
            output = model(test_input)

        expected_shape = (
            self.batch_size,
            self.output_channels,
            self.image_size,
            self.image_size,
        )
        self.assertEqual(output.shape, expected_shape)
        print("[SUCCESS] Pix2PixGenerator forward pass")

    def test_pix2pix_discriminator_creation(self):
        """Test Pix2PixDiscriminator model creation."""
        model = Pix2PixDiscriminator(
            input_channels=self.input_channels + self.output_channels
        )

        self.assertIsInstance(model, nn.Module)
        print("[SUCCESS] Pix2PixDiscriminator model creation")

    def test_pix2pix_discriminator_forward_pass(self):
        """Test Pix2PixDiscriminator forward pass."""
        model = Pix2PixDiscriminator(
            input_channels=self.input_channels + self.output_channels
        )

        # Create separate input and target images
        input_img = torch.randn(
            self.batch_size,
            self.input_channels,
            self.image_size,
            self.image_size,
        )

        target_img = torch.randn(
            self.batch_size,
            self.output_channels,
            self.image_size,
            self.image_size,
        )

        with torch.no_grad():
            output = model(input_img, target_img)

        # Should output a prediction
        self.assertEqual(len(output.shape), 4)  # Should be 4D tensor
        self.assertEqual(output.shape[0], self.batch_size)
        print("[SUCCESS] Pix2PixDiscriminator forward pass")

    def test_res_block_creation(self):
        """Test ResBlock creation."""
        channels = 64
        block = ResBlock(channels)

        self.assertIsInstance(block, nn.Module)
        print("[SUCCESS] ResBlock creation")

    def test_res_block_forward_pass(self):
        """Test ResBlock forward pass."""
        channels = 64
        block = ResBlock(channels)

        test_input = torch.randn(self.batch_size, channels, 32, 32)

        with torch.no_grad():
            output = block(test_input)

        self.assertEqual(output.shape, test_input.shape)
        print("[SUCCESS] ResBlock forward pass")

    def test_conv_block_creation(self):
        """Test ConvBlock creation."""
        in_channels = 64
        out_channels = 128
        block = ConvBlock(in_channels, out_channels)

        self.assertIsInstance(block, nn.Module)
        print("[SUCCESS] ConvBlock creation")

    def test_conv_block_forward_pass(self):
        """Test ConvBlock forward pass."""
        in_channels = 64
        out_channels = 128
        block = ConvBlock(in_channels, out_channels)

        test_input = torch.randn(self.batch_size, in_channels, 32, 32)

        with torch.no_grad():
            output = block(test_input)

        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], out_channels)
        print("[SUCCESS] ConvBlock forward pass")

    def test_attention_block_creation(self):
        """Test AttentionBlock creation."""
        channels = 64
        block = AttentionBlock(channels)

        self.assertIsInstance(block, nn.Module)
        print("[SUCCESS] AttentionBlock creation")

    def test_attention_block_forward_pass(self):
        """Test AttentionBlock forward pass."""
        channels = 64
        block = AttentionBlock(channels)

        test_input = torch.randn(self.batch_size, channels, 32, 32)

        with torch.no_grad():
            output = block(test_input)

        self.assertEqual(output.shape, test_input.shape)
        print("[SUCCESS] AttentionBlock forward pass")

    def test_model_parameter_counts(self):
        """Test model parameter counting."""
        generator = Pix2PixGenerator(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
        )

        discriminator = Pix2PixDiscriminator(
            input_channels=self.input_channels + self.output_channels
        )

        gen_params = sum(p.numel() for p in generator.parameters())
        disc_params = sum(p.numel() for p in discriminator.parameters())

        self.assertGreater(gen_params, 0)
        self.assertGreater(disc_params, 0)
        print("[SUCCESS] Model parameter counting")


if __name__ == "__main__":
    unittest.main()
