"""
Unit tests for the core models module.

This module contains comprehensive unit tests for all model architectures
in the src/core/models.py module, ensuring robust model functionality
with maximum code coverage.
"""

import unittest

import torch
import torch.nn as nn

from src.core.models import (
    AttentionBlock,
    ConvBlock,
    CycleGANDiscriminator,
    CycleGANGenerator,
    Pix2PixDiscriminator,
    Pix2PixGenerator,
    ResBlock,
    UNet,
    count_parameters,
    create_model,
)


class TestModelArchitectures(unittest.TestCase):
    """Test neural network model architectures."""

    def setUp(self):
        """Set up test environment."""
        self.input_channels = 3
        self.output_channels = 3
        self.batch_size = 2
        self.image_size = 64  # Smaller size for faster testing

    def test_pix2pix_generator_creation(self):
        """Test Pix2PixGenerator model creation."""
        model = Pix2PixGenerator(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            ngf=32,  # Smaller for testing
            n_blocks=3,
        )

        self.assertIsInstance(model, nn.Module)
        print("[SUCCESS] Pix2PixGenerator model creation")

    def test_pix2pix_generator_forward_pass(self):
        """Test Pix2PixGenerator forward pass."""
        model = Pix2PixGenerator(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            ngf=32,
            n_blocks=3,
        )

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
            input_channels=self.input_channels + self.output_channels,
            ndf=32,
            n_layers=2,
        )

        self.assertIsInstance(model, nn.Module)
        print("[SUCCESS] Pix2PixDiscriminator model creation")

    def test_pix2pix_discriminator_forward_pass(self):
        """Test Pix2PixDiscriminator forward pass."""
        model = Pix2PixDiscriminator(
            input_channels=self.input_channels + self.output_channels,
            ndf=32,
            n_layers=2,
        )

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

        self.assertEqual(len(output.shape), 4)
        self.assertEqual(output.shape[0], self.batch_size)
        print("[SUCCESS] Pix2PixDiscriminator forward pass")

    def test_cyclegan_generator_creation(self):
        """Test CycleGANGenerator model creation."""
        model = CycleGANGenerator(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            ngf=32,
            n_blocks=3,
        )

        self.assertIsInstance(model, nn.Module)
        print("[SUCCESS] CycleGANGenerator model creation")

    def test_cyclegan_generator_forward_pass(self):
        """Test CycleGANGenerator forward pass."""
        model = CycleGANGenerator(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            ngf=32,
            n_blocks=3,
        )

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
        print("[SUCCESS] CycleGANGenerator forward pass")

    def test_cyclegan_discriminator_creation(self):
        """Test CycleGANDiscriminator model creation."""
        model = CycleGANDiscriminator(
            input_channels=self.input_channels, ndf=32, n_layers=2
        )

        self.assertIsInstance(model, nn.Module)
        print("[SUCCESS] CycleGANDiscriminator model creation")

    def test_cyclegan_discriminator_forward_pass(self):
        """Test CycleGANDiscriminator forward pass."""
        model = CycleGANDiscriminator(
            input_channels=self.input_channels, ndf=32, n_layers=2
        )

        test_input = torch.randn(
            self.batch_size,
            self.input_channels,
            self.image_size,
            self.image_size,
        )

        with torch.no_grad():
            output = model(test_input)

        self.assertEqual(len(output.shape), 4)
        self.assertEqual(output.shape[0], self.batch_size)
        print("[SUCCESS] CycleGANDiscriminator forward pass")

    def test_unet_creation(self):
        """Test UNet model creation."""
        model = UNet(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            features=[32, 64, 128],  # Smaller for testing
            dropout=0.1,
        )

        self.assertIsInstance(model, nn.Module)
        print("[SUCCESS] UNet model creation")

    def test_unet_forward_pass(self):
        """Test UNet forward pass."""
        model = UNet(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            features=[32, 64, 128],
            dropout=0.1,
        )

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
        print("[SUCCESS] UNet forward pass")

    def test_conv_block_creation(self):
        """Test ConvBlock creation."""
        block = ConvBlock(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        self.assertIsInstance(block, nn.Module)
        print("[SUCCESS] ConvBlock creation")

    def test_conv_block_forward_pass(self):
        """Test ConvBlock forward pass."""
        block = ConvBlock(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        test_input = torch.randn(self.batch_size, 32, 32, 32)

        with torch.no_grad():
            output = block(test_input)

        self.assertEqual(output.shape, (self.batch_size, 64, 32, 32))
        print("[SUCCESS] ConvBlock forward pass")

    def test_res_block_creation(self):
        """Test ResBlock creation."""
        block = ResBlock(channels=64)

        self.assertIsInstance(block, nn.Module)
        print("[SUCCESS] ResBlock creation")

    def test_res_block_forward_pass(self):
        """Test ResBlock forward pass."""
        block = ResBlock(channels=64)

        test_input = torch.randn(self.batch_size, 64, 32, 32)

        with torch.no_grad():
            output = block(test_input)

        self.assertEqual(output.shape, test_input.shape)
        print("[SUCCESS] ResBlock forward pass")

    def test_attention_block_creation(self):
        """Test AttentionBlock creation."""
        block = AttentionBlock(channels=64)

        self.assertIsInstance(block, nn.Module)
        print("[SUCCESS] AttentionBlock creation")

    def test_attention_block_forward_pass(self):
        """Test AttentionBlock forward pass."""
        block = AttentionBlock(channels=64)

        test_input = torch.randn(self.batch_size, 64, 32, 32)

        with torch.no_grad():
            output = block(test_input)

        self.assertEqual(output.shape, test_input.shape)
        print("[SUCCESS] AttentionBlock forward pass")

    def test_model_factory_unet(self):
        """Test model factory for UNet."""
        config = {
            "architecture": "unet",
            "parameters": {
                "input_channels": 3,
                "output_channels": 3,
                "features": [32, 64, 128],
            },
        }

        model = create_model(config)

        self.assertIsInstance(model, UNet)
        print("[SUCCESS] Model factory UNet creation")

    def test_model_factory_pix2pix(self):
        """Test model factory for Pix2Pix."""
        config = {
            "architecture": "pix2pix",
            "parameters": {
                "generator": {
                    "input_channels": 3,
                    "output_channels": 3,
                    "ngf": 32,
                },
                "discriminator": {"input_channels": 6, "ndf": 32},
            },
        }

        models = create_model(config)

        self.assertIsInstance(models, dict)
        self.assertTrue("generator" in models)
        self.assertTrue("discriminator" in models)
        self.assertIsInstance(models["generator"], Pix2PixGenerator)
        self.assertIsInstance(models["discriminator"], Pix2PixDiscriminator)
        print("[SUCCESS] Model factory Pix2Pix creation")

    def test_count_parameters(self):
        """Test parameter counting functionality."""
        model = Pix2PixGenerator(
            input_channels=3, output_channels=3, ngf=32, n_blocks=3
        )

        param_count = count_parameters(model)

        self.assertIsInstance(param_count, int)
        self.assertGreater(param_count, 0)
        print("[SUCCESS] Parameter counting")

    def test_count_parameters_dict(self):
        """Test parameter counting for model dictionary."""
        config = {
            "architecture": "pix2pix",
            "parameters": {
                "generator": {"ngf": 32},
                "discriminator": {"ndf": 32},
            },
        }

        models = create_model(config)
        param_count = count_parameters(models)

        self.assertIsInstance(param_count, int)
        self.assertGreater(param_count, 0)
        print("[SUCCESS] Parameter counting for model dictionary")

    def test_model_device_compatibility(self):
        """Test model device compatibility."""
        model = UNet(input_channels=3, output_channels=3, features=[32, 64])

        # Test CPU
        device = torch.device("cpu")
        model = model.to(device)

        torch.randn(1, 3, 64, 64).to(device)
