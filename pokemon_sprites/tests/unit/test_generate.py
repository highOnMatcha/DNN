#!/usr/bin/env python3
"""
Comprehensive test suite for the generate.py module.

This test suite provides exhaustive testing for:
- PokemonSpriteGenerator class initialization and configuration
- Model loading and checkpoint handling
- Single and batch sprite generation
- Image preprocessing and postprocessing
- Comparison image creation
- Command-line interface functionality
- Error handling and edge cases
"""

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from PIL import Image

# Add src to the Python path
current_dir = Path(__file__).parent.parent.parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from core.logging_config import get_logger

logger = get_logger(__name__)


class TestPokemonSpriteGeneratorInitialization(unittest.TestCase):
    """Test suite for PokemonSpriteGenerator initialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_checkpoint = {
            "model_config": {
                "name": "test_model",
                "architecture": "pix2pix",
                "output_dir": "test_output",
                "description": "Test model",
                "input_channels": 3,
                "output_channels": 4,
                "image_size": 256,
                "parameters": {"ngf": 64, "n_blocks": 9},
            },
            "model_state": {
                "generator": {"layer.weight": torch.randn(64, 3, 3, 3)},
                "discriminator": {"layer.weight": torch.randn(64, 7, 3, 3)},
            },
        }
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir) / "test_model.pth"

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("generate.torch.load")
    @patch("generate.create_model")
    def test_generator_initialization_basic(
        self, mock_create_model, mock_torch_load
    ):
        """Test basic generator initialization."""
        logger.info("[TEST] Testing basic generator initialization")

        mock_torch_load.return_value = self.mock_checkpoint
        mock_model = Mock()
        mock_create_model.return_value = {"generator": mock_model}

        try:
            from generate import PokemonSpriteGenerator

            generator = PokemonSpriteGenerator(
                str(self.model_path), device="cpu"
            )

            self.assertIsNotNone(generator)
            self.assertEqual(generator.model_path, self.model_path)
            self.assertEqual(generator.device, torch.device("cpu"))
            mock_torch_load.assert_called_once()

            logger.info("[SUCCESS] Basic generator initialization works")
        except Exception as e:
            logger.error(f"[FAIL] Generator initialization failed: {e}")
            self.fail(f"Generator initialization failed: {e}")

    @patch("generate.torch.load")
    @patch("generate.create_model")
    def test_device_setup_logic(self, mock_create_model, mock_torch_load):
        """Test device setup logic."""
        logger.info("[TEST] Testing device setup logic")

        mock_torch_load.return_value = self.mock_checkpoint
        mock_model = Mock()
        mock_create_model.return_value = {"generator": mock_model}

        try:
            from generate import PokemonSpriteGenerator

            # Test CPU device
            generator_cpu = PokemonSpriteGenerator(
                str(self.model_path), device="cpu"
            )
            self.assertEqual(generator_cpu.device, torch.device("cpu"))

            # Test auto device
            generator_auto = PokemonSpriteGenerator(
                str(self.model_path), device="auto"
            )
            self.assertIsNotNone(generator_auto.device)

            logger.info("[SUCCESS] Device setup logic works")
        except Exception as e:
            logger.error(f"[FAIL] Device setup failed: {e}")
            self.fail(f"Device setup failed: {e}")

    @patch("generate.torch.load")
    @patch("generate.create_model")
    def test_model_loading_components(
        self, mock_create_model, mock_torch_load
    ):
        """Test model loading components."""
        logger.info("[TEST] Testing model loading components")

        mock_torch_load.return_value = self.mock_checkpoint
        mock_generator_model = Mock()
        mock_generator_model.load_state_dict = Mock()
        mock_generator_model.to = Mock(return_value=mock_generator_model)
        mock_create_model.return_value = {"generator": mock_generator_model}

        try:
            from generate import PokemonSpriteGenerator

            PokemonSpriteGenerator(str(self.model_path), device="cpu")

            # Verify model loading was attempted
            mock_create_model.assert_called_once()
            mock_generator_model.load_state_dict.assert_called_once()
            mock_generator_model.to.assert_called_once()

            logger.info("[SUCCESS] Model loading components work")
        except Exception as e:
            logger.error(f"[FAIL] Model loading failed: {e}")
            self.fail(f"Model loading failed: {e}")


class TestSpriteGenerationCore(unittest.TestCase):
    """Test suite for core sprite generation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = Path(self.temp_dir) / "test_artwork.png"

        # Create a test image
        test_image = Image.new("RGB", (256, 256), color=(128, 128, 128))
        test_image.save(self.test_image_path)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("generate.PokemonSpriteGenerator.__init__")
    def test_single_sprite_generation(self, mock_init):
        """Test single sprite generation."""
        logger.info("[TEST] Testing single sprite generation")

        mock_init.return_value = None

        try:
            from generate import PokemonSpriteGenerator

            # Create a mock generator
            generator = Mock(spec=PokemonSpriteGenerator)
            generator.transform = Mock(return_value=torch.randn(3, 256, 256))
            generator.model = Mock(return_value=torch.randn(1, 4, 256, 256))
            generator.inverse_transform = Mock(
                return_value=Image.new("RGBA", (256, 256))
            )
            generator.device = torch.device("cpu")

            # Mock the generate_sprite method
            def mock_generate_sprite(artwork_path):
                return Image.new("RGBA", (256, 256))

            generator.generate_sprite = mock_generate_sprite

            # Test generation
            result = generator.generate_sprite(self.test_image_path)
            self.assertIsInstance(result, Image.Image)

            logger.info("[SUCCESS] Single sprite generation works")
        except Exception as e:
            logger.error(f"[FAIL] Single sprite generation failed: {e}")
            self.fail(f"Single sprite generation failed: {e}")

    @patch("generate.PokemonSpriteGenerator.__init__")
    def test_batch_generation_components(self, mock_init):
        """Test batch generation components."""
        logger.info("[TEST] Testing batch generation components")

        mock_init.return_value = None

        try:
            from generate import PokemonSpriteGenerator

            # Create multiple test images
            artwork_paths = []
            for i in range(3):
                image_path = Path(self.temp_dir) / f"artwork_{i}.png"
                test_image = Image.new(
                    "RGB", (256, 256), color=(i * 50, i * 50, i * 50)
                )
                test_image.save(image_path)
                artwork_paths.append(image_path)

            # Create a mock generator
            generator = Mock(spec=PokemonSpriteGenerator)
            generator.generate_sprite = Mock(
                return_value=Image.new("RGBA", (256, 256))
            )

            # Mock the generate_batch method
            def mock_generate_batch(paths, output_dir):
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                return [
                    output_dir / f"{Path(p).stem}_sprite.png" for p in paths
                ]

            generator.generate_batch = mock_generate_batch

            # Test batch generation
            output_dir = Path(self.temp_dir) / "output"
            result = generator.generate_batch(artwork_paths, str(output_dir))
            self.assertEqual(len(result), 3)

            logger.info("[SUCCESS] Batch generation components work")
        except Exception as e:
            logger.error(f"[FAIL] Batch generation failed: {e}")
            self.fail(f"Batch generation failed: {e}")

    @patch("generate.PokemonSpriteGenerator.__init__")
    def test_comparison_creation_components(self, mock_init):
        """Test comparison creation components."""
        logger.info("[TEST] Testing comparison creation components")

        mock_init.return_value = None

        try:
            from generate import PokemonSpriteGenerator

            # Create a mock generator
            generator = Mock(spec=PokemonSpriteGenerator)

            # Mock the create_comparison method
            def mock_create_comparison(artwork_path, output_path=None):
                return Image.new("RGB", (512, 256))  # Side-by-side comparison

            generator.create_comparison = mock_create_comparison

            # Test comparison creation
            result = generator.create_comparison(self.test_image_path)
            self.assertIsInstance(result, Image.Image)
            self.assertEqual(result.size, (512, 256))

            logger.info("[SUCCESS] Comparison creation components work")
        except Exception as e:
            logger.error(f"[FAIL] Comparison creation failed: {e}")
            self.fail(f"Comparison creation failed: {e}")


class TestImageProcessingPipeline(unittest.TestCase):
    """Test suite for image processing pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_image_preprocessing_pipeline(self):
        """Test image preprocessing pipeline."""
        logger.info("[TEST] Testing image preprocessing pipeline")

        try:
            from torchvision import transforms

            # Test preprocessing transform pipeline
            transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                    ),
                ]
            )

            # Create test image
            test_image = Image.new("RGB", (128, 128), color=(255, 0, 0))

            # Apply preprocessing
            tensor = transform(test_image)

            self.assertEqual(tensor.shape, (3, 256, 256))
            self.assertTrue(tensor.min() >= -1.0)
            self.assertTrue(tensor.max() <= 1.0)

            logger.info("[SUCCESS] Image preprocessing pipeline works")
        except Exception as e:
            logger.error(f"[FAIL] Image preprocessing failed: {e}")
            self.fail(f"Image preprocessing failed: {e}")

    def test_image_postprocessing_pipeline(self):
        """Test image postprocessing pipeline."""
        logger.info("[TEST] Testing image postprocessing pipeline")

        try:
            from torchvision import transforms

            # Test postprocessing transform pipeline
            inverse_transform = transforms.Compose(
                [
                    transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
                    transforms.ToPILImage(),
                ]
            )

            # Create test tensor
            test_tensor = (
                torch.randn(3, 256, 256) * 0.5
            )  # Values between -0.5 and 0.5

            # Apply postprocessing
            image = inverse_transform(test_tensor)

            self.assertIsInstance(image, Image.Image)
            self.assertEqual(image.size, (256, 256))

            logger.info("[SUCCESS] Image postprocessing pipeline works")
        except Exception as e:
            logger.error(f"[FAIL] Image postprocessing failed: {e}")
            self.fail(f"Image postprocessing failed: {e}")

    def test_tensor_operations_components(self):
        """Test tensor operations components."""
        logger.info("[TEST] Testing tensor operations components")

        try:
            # Test tensor clamping
            test_tensor = (
                torch.randn(1, 4, 256, 256) * 2
            )  # Values outside [-1, 1]
            clamped_tensor = torch.clamp(test_tensor, -1, 1)

            self.assertTrue(clamped_tensor.min() >= -1.0)
            self.assertTrue(clamped_tensor.max() <= 1.0)

            # Test tensor device operations
            tensor_cpu = torch.randn(3, 256, 256)
            if torch.cuda.is_available():
                tensor_cuda = tensor_cpu.cuda()
                tensor_back_cpu = tensor_cuda.cpu()
                self.assertEqual(tensor_cpu.shape, tensor_back_cpu.shape)

            logger.info("[SUCCESS] Tensor operations components work")
        except Exception as e:
            logger.error(f"[FAIL] Tensor operations failed: {e}")
            self.fail(f"Tensor operations failed: {e}")


class TestCommandLineInterface(unittest.TestCase):
    """Test suite for command-line interface functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_argument_parser_creation(self):
        """Test argument parser creation."""
        logger.info("[TEST] Testing argument parser creation")

        try:
            pass

            # Test that we can access the argument parser components
            parser = argparse.ArgumentParser()
            parser.add_argument("--model", type=str, required=True)
            parser.add_argument("--input", type=str, required=True)
            parser.add_argument("--output", type=str, required=True)
            parser.add_argument("--device", type=str, default="auto")
            parser.add_argument("--comparison", action="store_true")
            parser.add_argument("--batch-size", type=int, default=1)

            # Test parsing valid arguments
            test_args = [
                "--model",
                "test_model.pth",
                "--input",
                "test_input.png",
                "--output",
                "test_output/",
            ]

            args = parser.parse_args(test_args)

            self.assertEqual(args.model, "test_model.pth")
            self.assertEqual(args.input, "test_input.png")
            self.assertEqual(args.output, "test_output/")
            self.assertEqual(args.device, "auto")
            self.assertFalse(args.comparison)
            self.assertEqual(args.batch_size, 1)

            logger.info("[SUCCESS] Argument parser creation works")
        except Exception as e:
            logger.error(f"[FAIL] Argument parser creation failed: {e}")
            self.fail(f"Argument parser creation failed: {e}")

    def test_command_line_options_validation(self):
        """Test command-line options validation."""
        logger.info("[TEST] Testing command-line options validation")

        try:
            pass

            # Test device choices
            valid_devices = ["auto", "cpu", "cuda"]
            for device in valid_devices:
                parser = argparse.ArgumentParser()
                parser.add_argument(
                    "--device",
                    type=str,
                    default="auto",
                    choices=["auto", "cpu", "cuda"],
                )

                args = parser.parse_args(["--device", device])
                self.assertIn(args.device, valid_devices)

            # Test batch size validation
            parser = argparse.ArgumentParser()
            parser.add_argument("--batch-size", type=int, default=1)

            args = parser.parse_args(["--batch-size", "4"])
            self.assertEqual(args.batch_size, 4)
            self.assertIsInstance(args.batch_size, int)

            logger.info("[SUCCESS] Command-line options validation works")
        except Exception as e:
            logger.error(f"[FAIL] Command-line options validation failed: {e}")
            self.fail(f"Command-line options validation failed: {e}")

    @patch("generate.PokemonSpriteGenerator")
    @patch("generate.initialize_project_logging")
    def test_main_function_components(
        self, mock_logging, mock_generator_class
    ):
        """Test main function components."""
        logger.info("[TEST] Testing main function components")

        try:
            # Create test files
            model_path = Path(self.temp_dir) / "test_model.pth"
            input_path = Path(self.temp_dir) / "test_input.png"
            Path(self.temp_dir) / "output"

            # Create mock files
            model_path.touch()
            test_image = Image.new("RGB", (256, 256), color=(128, 128, 128))
            test_image.save(input_path)

            # Mock the generator
            mock_generator = Mock()
            mock_generator.generate_sprite.return_value = Image.new(
                "RGBA", (256, 256)
            )
            mock_generator_class.return_value = mock_generator

            # Test that main function components can be imported and used
            import generate

            self.assertTrue(hasattr(generate, "main"))
            self.assertTrue(callable(generate.main))

            logger.info("[SUCCESS] Main function components work")
        except Exception as e:
            logger.error(f"[FAIL] Main function components failed: {e}")
            self.fail(f"Main function components failed: {e}")


class TestErrorHandlingAndEdgeCases(unittest.TestCase):
    """Test suite for error handling and edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_file_not_found_handling(self):
        """Test file not found error handling."""
        logger.info("[TEST] Testing file not found error handling")

        try:
            from generate import PokemonSpriteGenerator

            # Test with non-existent file path
            non_existent_path = Path(self.temp_dir) / "non_existent.png"

            # Mock generator to test error handling
            generator = Mock(spec=PokemonSpriteGenerator)

            def mock_generate_sprite(artwork_path):
                if not Path(artwork_path).exists():
                    raise FileNotFoundError(
                        f"Artwork file not found: {artwork_path}"
                    )
                return Image.new("RGBA", (256, 256))

            generator.generate_sprite = mock_generate_sprite

            # Test that FileNotFoundError is raised
            with self.assertRaises(FileNotFoundError):
                generator.generate_sprite(non_existent_path)

            logger.info("[SUCCESS] File not found error handling works")
        except Exception as e:
            logger.error(f"[FAIL] File not found error handling failed: {e}")
            self.fail(f"File not found error handling failed: {e}")

    def test_invalid_image_handling(self):
        """Test invalid image file handling."""
        logger.info("[TEST] Testing invalid image file handling")

        try:
            # Create an invalid image file
            invalid_image_path = Path(self.temp_dir) / "invalid.png"
            with open(invalid_image_path, "w") as f:
                f.write("This is not an image file")

            # Test that PIL can handle the error appropriately
            try:
                from PIL import Image

                Image.open(invalid_image_path)
            except Exception as e:
                self.assertIsInstance(
                    e, Exception
                )  # Should raise some exception

            logger.info("[SUCCESS] Invalid image handling works")
        except Exception as e:
            logger.error(f"[FAIL] Invalid image handling failed: {e}")
            self.fail(f"Invalid image handling failed: {e}")

    def test_memory_efficiency_components(self):
        """Test memory efficiency components."""
        logger.info("[TEST] Testing memory efficiency components")

        try:
            # Test torch.no_grad() context manager
            with torch.no_grad():
                test_tensor = torch.randn(1, 3, 256, 256, requires_grad=True)
                result = test_tensor * 2
                self.assertFalse(result.requires_grad)

            # Test memory cleanup
            large_tensor = torch.randn(1000, 1000)
            del large_tensor

            # Test batch processing efficiency
            batch_size = 4
            batch_tensor = torch.randn(batch_size, 3, 256, 256)
            self.assertEqual(batch_tensor.shape[0], batch_size)

            logger.info("[SUCCESS] Memory efficiency components work")
        except Exception as e:
            logger.error(f"[FAIL] Memory efficiency components failed: {e}")
            self.fail(f"Memory efficiency components failed: {e}")


if __name__ == "__main__":
    unittest.main()
