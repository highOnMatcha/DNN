"""
Unit tests for training/config_loader.py module.

This module provides testing coverage for configuration loading,
validation, and model information retrieval functionality.
"""

import sys
import unittest
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.logging_config import get_logger
from training.config_loader import ConfigurationLoader

logger = get_logger(__name__)


class TestConfigurationLoaderBasic(unittest.TestCase):
    """Test basic ConfigurationLoader functionality."""

    def setUp(self):
        """Set up test environment."""
        self.config_loader = ConfigurationLoader()

    def test_configuration_loader_initialization(self):
        """Test ConfigurationLoader initialization."""
        self.assertIsInstance(self.config_loader, ConfigurationLoader)
        self.assertEqual(len(self.config_loader.supported_models), 4)
        self.assertEqual(len(self.config_loader.supported_configs), 3)
        logger.info("ConfigurationLoader initialization verified")

    def test_get_data_root_dir(self):
        """Test data root directory retrieval from config loader."""
        root_dir = self.config_loader.get_data_root_dir()
        self.assertIsInstance(root_dir, Path)
        logger.info("Data root directory retrieval verified")

    def test_get_available_models(self):
        """Test available models retrieval."""
        models = self.config_loader.get_available_models()
        self.assertIsInstance(models, list)
        self.assertIn("lightweight-baseline", models)
        self.assertIn("sprite-optimized", models)
        self.assertIn("transformer-enhanced", models)
        logger.info("Available models retrieval verified")

    def test_get_available_configs(self):
        """Test available configs retrieval."""
        configs = self.config_loader.get_available_configs()
        self.assertIsInstance(configs, list)
        self.assertIn("test", configs)
        self.assertIn("development", configs)
        self.assertIn("production", configs)
        logger.info("Available configs retrieval verified")

    def test_get_model_info_basic(self):
        """Test basic model info retrieval."""
        model_info = self.config_loader.get_model_info("lightweight-baseline")
        self.assertIsInstance(model_info, dict)
        self.assertIn("description", model_info)
        self.assertIn("architecture", model_info)
        logger.info("Model info retrieval verified")

    def test_get_model_info_empty_string(self):
        """Test model info retrieval with empty string."""
        model_info = self.config_loader.get_model_info("")
        self.assertIsInstance(model_info, dict)
        self.assertEqual(model_info, {})
        logger.info("Empty string model info handling verified")

    def test_get_model_info_none(self):
        """Test model info retrieval with None."""
        model_info = self.config_loader.get_model_info(None)
        self.assertIsInstance(model_info, dict)
        self.assertEqual(model_info, {})
        logger.info("None model info handling verified")

    def test_get_model_info_nonexistent(self):
        """Test model info retrieval with nonexistent model."""
        model_info = self.config_loader.get_model_info("nonexistent-model")
        self.assertIsInstance(model_info, dict)
        self.assertEqual(model_info, {})
        logger.info("Nonexistent model info handling verified")


if __name__ == "__main__":
    unittest.main(verbosity=2)
