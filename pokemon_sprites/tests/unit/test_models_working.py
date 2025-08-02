"""
Working unit tests for models module.

This module tests the actual model classes that exist in the codebase.
"""

import unittest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    from core.logging_config import initialize_project_logging
    from core.models import (
        UNet,
        Pix2PixGenerator,
        Pix2PixDiscriminator,
        ConvBlock,
        ResBlock,
        AttentionBlock,
        SelfAttention,
        create_model
    )
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR = str(e)

# Configure test logging
if IMPORTS_SUCCESSFUL:
    initialize_project_logging("test_models")


class TestColors:
    """ANSI color codes for professional test output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def print_test_result(test_name: str, success: bool, message: str = ""):
    """Print formatted test result with appropriate colors."""
    if success:
        print(f"{TestColors.GREEN}{TestColors.BOLD}[SUCCESS]{TestColors.RESET} {test_name}")
        if message:
            print(f"          {message}")
    else:
        print(f"{TestColors.RED}{TestColors.BOLD}[FAIL]{TestColors.RESET} {test_name}")
        if message:
            print(f"       {message}")


@unittest.skipIf(not IMPORTS_SUCCESSFUL, f"Import failed: {IMPORT_ERROR if not IMPORTS_SUCCESSFUL else ''}")
class TestBasicModels(unittest.TestCase):
    """Test suite for basic model functionality."""

    def setUp(self):
        """Set up test environment."""
        self.batch_size = 4
        self.channels = 3
        self.height = 64
        self.width = 64
        self.input_tensor = torch.randn(self.batch_size, self.channels, self.height, self.width)

    def test_unet_model(self):
        """Test UNet model creation and forward pass."""
        model = UNet(input_channels=3, output_channels=3, features=[32, 64, 128])
        self.assertIsNotNone(model)
        
        # Test forward pass
        x = torch.randn(2, 3, 64, 64)
        output = model(x)
        
        self.assertEqual(output.shape, (2, 3, 64, 64))
        self.assertFalse(torch.isnan(output).any())
        
        print_test_result("test_unet_model", True, 
                         f"UNet output shape: {output.shape}")

    def test_pix2pix_generator(self):
        """Test Pix2Pix generator model."""
        model = Pix2PixGenerator(input_channels=3, output_channels=3, ngf=64)
        self.assertIsNotNone(model)
        
        # Test forward pass
        x = torch.randn(2, 3, 256, 256)
        output = model(x)
        
        self.assertEqual(output.shape, (2, 3, 256, 256))
        self.assertFalse(torch.isnan(output).any())
        
        print_test_result("test_pix2pix_generator", True, 
                         f"Pix2PixGenerator output shape: {output.shape}")

    def test_pix2pix_discriminator(self):
        """Test Pix2Pix discriminator model."""
        model = Pix2PixDiscriminator(input_channels=6, ndf=64)  # 6 channels for input+target
        self.assertIsNotNone(model)
        
        # Test forward pass with separate input and target images
        input_img = torch.randn(2, 3, 256, 256)
        target_img = torch.randn(2, 3, 256, 256) 
        output = model(input_img, target_img)
        
        # Output should be a patch-wise prediction
        self.assertTrue(len(output.shape) == 4)  # Batch, channels, height, width
        self.assertFalse(torch.isnan(output).any())
        
        print_test_result("test_pix2pix_discriminator", True, 
                         f"Pix2PixDiscriminator output shape: {output.shape}")

    def test_conv_block(self):
        """Test ConvBlock module."""
        block = ConvBlock(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.assertIsNotNone(block)
        
        # Test forward pass
        with torch.no_grad():
            output = block(self.input_tensor)
            
        expected_shape = (self.batch_size, 64, self.height, self.width)
        self.assertEqual(output.shape, expected_shape)
        self.assertFalse(torch.isnan(output).any())
        
        print_test_result("test_conv_block", True, 
                         f"ConvBlock output shape: {output.shape}")

    def test_res_block(self):
        """Test ResBlock module."""
        block = ResBlock(channels=64)
        self.assertIsNotNone(block)
        
        # Create input with correct number of channels
        input_tensor = torch.randn(self.batch_size, 64, self.height, self.width)
        
        # Test forward pass
        with torch.no_grad():
            output = block(input_tensor)
            
        self.assertEqual(output.shape, input_tensor.shape)
        self.assertFalse(torch.isnan(output).any())
        
        print_test_result("test_res_block", True, 
                         f"ResBlock output shape: {output.shape}")

    def test_attention_block(self):
        """Test AttentionBlock module."""
        block = AttentionBlock(channels=64)
        self.assertIsNotNone(block)
        
        # Create input with correct number of channels
        input_tensor = torch.randn(self.batch_size, 64, self.height, self.width)
        
        # Test forward pass
        with torch.no_grad():
            output = block(input_tensor)
            
        self.assertEqual(output.shape, input_tensor.shape)
        self.assertFalse(torch.isnan(output).any())
        
        print_test_result("test_attention_block", True, 
                         f"AttentionBlock output shape: {output.shape}")

    def test_self_attention(self):
        """Test SelfAttention module."""
        attention = SelfAttention(channels=64)
        self.assertIsNotNone(attention)
        
        # Create input with correct number of channels
        input_tensor = torch.randn(self.batch_size, 64, self.height, self.width)
        
        # Test forward pass
        with torch.no_grad():
            output = attention(input_tensor)
            
        self.assertEqual(output.shape, input_tensor.shape)
        self.assertFalse(torch.isnan(output).any())
        
        print_test_result("test_self_attention", True, 
                         f"SelfAttention output shape: {output.shape}")

    def test_model_parameter_count(self):
        """Test that models have reasonable parameter counts."""
        models_to_test = [
            ("UNet", UNet(input_channels=3, output_channels=3, features=[32, 64, 128])),
            ("Pix2PixGenerator", Pix2PixGenerator(input_channels=3, output_channels=3, ngf=64)),
            ("Pix2PixDiscriminator", Pix2PixDiscriminator(input_channels=6, ndf=64)),
        ]
        
        for name, model in models_to_test:
            param_count = sum(p.numel() for p in model.parameters())
            self.assertGreater(param_count, 1000)  # Should have at least 1K parameters
            self.assertLess(param_count, 1e8)      # Should be less than 100M parameters
        
        print_test_result("test_model_parameter_count", True, 
                         "All models have reasonable parameter counts")


if __name__ == '__main__':
    print(f"{TestColors.BLUE}{TestColors.BOLD}Running Model Tests{TestColors.RESET}\n")
    unittest.main(verbosity=2)
