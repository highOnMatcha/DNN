"""
Simple unit tests for configuration settings module.

This test suite validates basic configuration functionality that exists
in the actual codebase.
"""

import unittest

from config.settings import (
    ModelConfig,
    TrainingConfig,
    get_available_training_configs,
    get_data_root_dir,
    get_models_root_dir,
    get_training_config,
    list_available_models,
)

# Import test configuration (sets up path)
from tests.test_config import TestColors, print_test_result


class TestBasicConfiguration(unittest.TestCase):
    """Test suite for basic configuration functionality."""

    def test_get_models_root_dir(self):
        """Test models root directory retrieval."""
        models_dir = get_models_root_dir()

        # Should return a valid path
        self.assertIsInstance(models_dir, str)
        self.assertTrue(len(models_dir) > 0)
        print_test_result("Models root directory", True, f"Path: {models_dir}")

    def test_get_data_root_dir(self):
        """Test data root directory retrieval."""
        data_dir = get_data_root_dir()

        # Should return a valid path
        self.assertIsInstance(data_dir, str)
        self.assertTrue(len(data_dir) > 0)
        print_test_result("Data root directory", True, f"Path: {data_dir}")

    def test_training_config_creation(self):
        """Test TrainingConfig instantiation."""
        config = TrainingConfig()

        # Should have basic attributes
        self.assertTrue(hasattr(config, "epochs"))
        self.assertTrue(hasattr(config, "batch_size"))
        self.assertTrue(hasattr(config, "learning_rate"))

        # Basic validation
        self.assertIsInstance(config.epochs, int)
        self.assertIsInstance(config.batch_size, int)
        self.assertIsInstance(config.learning_rate, float)

        print_test_result(
            "TrainingConfig creation",
            True,
            f"epochs: {config.epochs}, batch_size: {config.batch_size}",
        )

    def test_model_config_creation(self):
        """Test ModelConfig instantiation."""
        config = ModelConfig(
            name="test", architecture="pix2pix", output_dir="/tmp/test"
        )

        # Should have basic attributes
        self.assertEqual(config.name, "test")
        self.assertEqual(config.architecture, "pix2pix")
        self.assertEqual(config.output_dir, "/tmp/test")

        print_test_result(
            "ModelConfig creation",
            True,
            f"name: {config.name}, arch: {config.architecture}",
        )

    def test_get_training_config(self):
        """Test training config retrieval."""
        config = get_training_config("development")

        # Should return a TrainingConfig object
        self.assertIsInstance(config, TrainingConfig)
        self.assertTrue(hasattr(config, "epochs"))
        self.assertTrue(hasattr(config, "batch_size"))

        print_test_result(
            "Training config retrieval", True, "Config type: development"
        )

    def test_list_available_models(self):
        """Test listing available models."""
        models = list_available_models()

        # Should return a dictionary
        self.assertIsInstance(models, dict)

        print_test_result(
            "Available models list",
            True,
            f"Model categories: {list(models.keys())}",
        )

    def test_get_available_training_configs(self):
        """Test getting available training configs."""
        configs = get_available_training_configs()

        # Should return a list
        self.assertIsInstance(configs, list)

        print_test_result(
            "Available training configs", True, f"Configs: {configs}"
        )


if __name__ == "__main__":
    print(
        f"{TestColors.BLUE}{TestColors.BOLD}Running Simple Configuration Tests{TestColors.RESET}\n"
    )
    unittest.main(verbosity=2)
