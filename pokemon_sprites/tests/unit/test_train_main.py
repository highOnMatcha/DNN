"""
Comprehensive unit tests for train.py main module functionality.

This module provides thorough testing coverage for the main training script components,
including argument validation, logging initialization, and pipeline execution workflows.
"""

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import train
from core.logging_config import get_logger

logger = get_logger(__name__)


class TestTrainMainFunctions(unittest.TestCase):
    """Test main functions in train.py module."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_setup_argument_parser_basic(self):
        """Test argument parser setup with basic functionality."""
        parser = train.setup_argument_parser()

        # Verify parser type
        self.assertIsInstance(parser, argparse.ArgumentParser)

        # Test parsing valid arguments
        test_args = [
            "--model",
            "lightweight-baseline",
            "--config",
            "development",
        ]
        args = parser.parse_args(test_args)

        self.assertEqual(args.model, "lightweight-baseline")
        self.assertEqual(args.config, "development")
        logger.info("Argument parser basic functionality validated")

    def test_setup_argument_parser_all_options(self):
        """Test argument parser with all available options."""
        parser = train.setup_argument_parser()

        # Test with all possible arguments
        test_args = [
            "--model",
            "sprite-optimized",
            "--config",
            "production",
            "--max-samples",
            "500",
            "--augmentation",
            "moderate",
            "--wandb",
            "--generate",
            "--max-generate",
            "100",
            "--log-level",
            "DEBUG",
            "--experiment-name",
            "test_experiment",
        ]

        args = parser.parse_args(test_args)

        self.assertEqual(args.model, "sprite-optimized")
        self.assertEqual(args.config, "production")
        self.assertEqual(args.max_samples, 500)
        self.assertEqual(args.augmentation, "moderate")
        self.assertTrue(args.wandb)
        self.assertTrue(args.generate)
        self.assertEqual(args.max_generate, 100)
        self.assertEqual(args.log_level, "DEBUG")
        self.assertEqual(args.experiment_name, "test_experiment")
        logger.info("All argument parser options validated")

    @patch("train.initialize_project_logging")
    @patch("train.log_system_info")
    def test_initialize_experiment_logging(self, mock_log_sys, mock_init_log):
        """Test experiment logging initialization."""
        # Create mock args
        mock_args = Mock()
        mock_args.log_level = "INFO"
        mock_args.experiment_name = "test_exp"
        mock_args.model = "lightweight-baseline"
        mock_args.config = "development"
        mock_args.max_samples = 100
        mock_args.wandb = True
        mock_args.generate = False

        # Test experiment logging initialization
        experiment_id = train.initialize_experiment_logging(mock_args)

        # Verify experiment ID format
        self.assertIsInstance(experiment_id, str)
        self.assertIn("lightweight-baseline", experiment_id)
        self.assertIn("development", experiment_id)

        # Verify logging functions were called
        mock_init_log.assert_called_once()
        mock_log_sys.assert_called_once()
        logger.info(f"Experiment logging initialized with ID: {experiment_id}")

    @patch("train.initialize_project_logging")
    @patch("train.log_system_info")
    def test_initialize_experiment_logging_minimal_args(
        self, mock_log_sys, mock_init_log
    ):
        """Test experiment logging with minimal arguments."""
        mock_args = Mock()
        mock_args.log_level = "WARNING"
        mock_args.experiment_name = None
        mock_args.model = "transformer-enhanced"
        mock_args.config = "test"
        mock_args.max_samples = None
        mock_args.wandb = False
        mock_args.generate = False

        experiment_id = train.initialize_experiment_logging(mock_args)

        self.assertIsInstance(experiment_id, str)
        self.assertIn("transformer-enhanced", experiment_id)
        self.assertIn("test", experiment_id)
        logger.info("Minimal experiment logging configuration tested")

    @patch("train.ConfigurationLoader")
    def test_validate_arguments_success(self, mock_config_loader_class):
        """Test successful argument validation."""
        # Mock configuration loader
        mock_config_loader = Mock()
        mock_config_loader.get_model_info.return_value = {
            "recommended_config": "production"
        }
        mock_config_loader_class.return_value = mock_config_loader

        # Create valid args
        mock_args = Mock()
        mock_args.model = "sprite-optimized"
        mock_args.config = "production"
        mock_args.max_samples = 1000
        mock_args.max_generate = 50
        mock_args.resume = None

        result = train.validate_arguments(mock_args)

        self.assertTrue(result)
        mock_config_loader.get_model_info.assert_called_once_with(
            "sprite-optimized"
        )
        logger.info("Argument validation success case verified")

    @patch("train.ConfigurationLoader")
    def test_validate_arguments_invalid_max_samples(
        self, mock_config_loader_class
    ):
        """Test argument validation with invalid max_samples."""
        mock_config_loader = Mock()
        mock_config_loader.get_model_info.return_value = {}
        mock_config_loader_class.return_value = mock_config_loader

        mock_args = Mock()
        mock_args.model = "lightweight-baseline"
        mock_args.config = "development"
        mock_args.max_samples = -10  # Invalid
        mock_args.max_generate = None
        mock_args.resume = None

        result = train.validate_arguments(mock_args)

        self.assertFalse(result)
        logger.info("Invalid max_samples validation handled correctly")

    @patch("train.ConfigurationLoader")
    def test_validate_arguments_invalid_max_generate(
        self, mock_config_loader_class
    ):
        """Test argument validation with invalid max_generate."""
        mock_config_loader = Mock()
        mock_config_loader.get_model_info.return_value = {}
        mock_config_loader_class.return_value = mock_config_loader

        mock_args = Mock()
        mock_args.model = "transformer-enhanced"
        mock_args.config = "production"
        mock_args.max_samples = None
        mock_args.max_generate = -5  # Negative value should fail
        mock_args.resume = None

        result = train.validate_arguments(mock_args)

        self.assertFalse(result)
        logger.info("Invalid max_generate validation handled correctly")

    @patch("train.ConfigurationLoader")
    def test_validate_arguments_nonexistent_resume_path(
        self, mock_config_loader_class
    ):
        """Test argument validation with nonexistent resume path."""
        mock_config_loader = Mock()
        mock_config_loader.get_model_info.return_value = {}
        mock_config_loader_class.return_value = mock_config_loader

        mock_args = Mock()
        mock_args.model = "sprite-optimized"
        mock_args.config = "development"
        mock_args.max_samples = None
        mock_args.max_generate = None
        mock_args.resume = "/nonexistent/path/checkpoint.pth"

        result = train.validate_arguments(mock_args)

        self.assertFalse(result)
        logger.info("Nonexistent resume path validation handled correctly")

    @patch("train.ConfigurationLoader")
    def test_validate_arguments_config_mismatch_warning(
        self, mock_config_loader_class
    ):
        """Test argument validation with config mismatch warning."""
        mock_config_loader = Mock()
        mock_config_loader.get_model_info.return_value = {
            "recommended_config": "production"
        }
        mock_config_loader_class.return_value = mock_config_loader

        mock_args = Mock()
        mock_args.model = "transformer-enhanced"
        mock_args.config = "test"  # Mismatch with recommended production
        mock_args.max_samples = None
        mock_args.max_generate = None
        mock_args.resume = None

        with patch.object(train.logger, "warning") as mock_warning:
            result = train.validate_arguments(mock_args)

            self.assertTrue(result)
            mock_warning.assert_called_once()
        logger.info("Config mismatch warning functionality verified")

    @patch("train.ConfigurationLoader")
    def test_validate_arguments_exception_handling(
        self, mock_config_loader_class
    ):
        """Test argument validation exception handling."""
        # Mock configuration loader to raise exception
        mock_config_loader_class.side_effect = Exception("Configuration error")

        mock_args = Mock()
        mock_args.model = "invalid-model"
        mock_args.config = "development"

        result = train.validate_arguments(mock_args)

        self.assertFalse(result)
        logger.info("Exception handling in argument validation verified")

    @patch("train.ConfigurationLoader")
    def test_print_model_recommendations(self, mock_config_loader_class):
        """Test model recommendations printing."""
        mock_config_loader = Mock()
        mock_model_info = {
            "description": "Test model description",
            "features": ["Feature 1", "Feature 2", "Feature 3"],
            "use_transfer_learning": True,
            "use_curriculum": False,
        }
        mock_config_loader.get_model_info.return_value = mock_model_info
        mock_config_loader_class.return_value = mock_config_loader

        mock_args = Mock()
        mock_args.model = "sprite-optimized"

        # Test should not raise any exceptions
        train.print_model_recommendations(mock_args)

        mock_config_loader.get_model_info.assert_called_once_with(
            "sprite-optimized"
        )
        logger.info("Model recommendations display functionality verified")

    @patch("train.ConfigurationLoader")
    def test_print_model_recommendations_no_info(
        self, mock_config_loader_class
    ):
        """Test model recommendations with no model info."""
        mock_config_loader = Mock()
        mock_config_loader.get_model_info.return_value = None
        mock_config_loader_class.return_value = mock_config_loader

        mock_args = Mock()
        mock_args.model = "unknown-model"

        # Should handle gracefully when no model info available
        train.print_model_recommendations(mock_args)

        mock_config_loader.get_model_info.assert_called_once_with(
            "unknown-model"
        )
        logger.info("No model info case handled correctly")

    @patch("train.ConfigurationLoader")
    def test_print_model_recommendations_with_curriculum(
        self, mock_config_loader_class
    ):
        """Test model recommendations with curriculum learning."""
        mock_config_loader = Mock()
        mock_model_info = {
            "description": "Advanced model with curriculum learning",
            "features": ["Advanced feature"],
            "use_transfer_learning": False,
            "use_curriculum": True,
        }
        mock_config_loader.get_model_info.return_value = mock_model_info
        mock_config_loader_class.return_value = mock_config_loader

        mock_args = Mock()
        mock_args.model = "transformer-enhanced"

        train.print_model_recommendations(mock_args)

        mock_config_loader.get_model_info.assert_called_once_with(
            "transformer-enhanced"
        )
        logger.info("Curriculum learning recommendations verified")


class TestMainFunction(unittest.TestCase):
    """Test the main function execution paths."""

    @patch("train.TrainingPipeline")
    @patch("train.ConfigurationLoader")
    @patch("train.print_model_recommendations")
    @patch("train.validate_arguments")
    @patch("train.initialize_experiment_logging")
    @patch("train.setup_argument_parser")
    @patch("sys.argv")
    def test_main_success_path(
        self,
        mock_argv,
        mock_parser_setup,
        mock_init_logging,
        mock_validate,
        mock_print_rec,
        mock_config_loader_class,
        mock_pipeline_class,
    ):
        """Test main function success execution path."""
        # Setup mocks
        mock_argv.__getitem__.return_value = [
            "train.py",
            "--model",
            "lightweight-baseline",
            "--config",
            "development",
        ]

        mock_parser = Mock()
        mock_args = Mock()
        mock_args.model = "lightweight-baseline"
        mock_args.config = "development"
        mock_parser.parse_args.return_value = mock_args
        mock_parser_setup.return_value = mock_parser

        mock_init_logging.return_value = "test_experiment_id"
        mock_validate.return_value = True

        mock_config_loader = Mock()
        mock_model_config = Mock()
        mock_training_config = Mock()
        mock_config_loader.load_and_validate_configs.return_value = (
            mock_model_config,
            mock_training_config,
        )
        mock_config_loader_class.return_value = mock_config_loader

        mock_pipeline = Mock()
        mock_pipeline.execute.return_value = True
        mock_pipeline_class.return_value = mock_pipeline

        # Test main function - should exit with code 0
        with patch("sys.exit") as mock_exit:
            train.main()
            mock_exit.assert_called_with(0)

        # Verify all components were called
        mock_parser.parse_args.assert_called_once()
        mock_init_logging.assert_called_once_with(mock_args)
        mock_validate.assert_called_once_with(mock_args)
        mock_print_rec.assert_called_once_with(mock_args)
        mock_config_loader.load_and_validate_configs.assert_called_once()
        mock_pipeline.execute.assert_called_once()
        logger.info("Main function success path verified")

    @patch("train.validate_arguments")
    @patch("train.initialize_experiment_logging")
    @patch("train.setup_argument_parser")
    @patch("sys.argv")
    def test_main_validation_failure(
        self, mock_argv, mock_parser_setup, mock_init_logging, mock_validate
    ):
        """Test main function with argument validation failure."""
        mock_argv.__getitem__.return_value = [
            "train.py",
            "--model",
            "invalid",
            "--config",
            "bad",
        ]

        mock_parser = Mock()
        mock_args = Mock()
        mock_parser.parse_args.return_value = mock_args
        mock_parser_setup.return_value = mock_parser

        mock_init_logging.return_value = "test_experiment_id"
        mock_validate.return_value = False  # Validation fails

        with patch("sys.exit") as mock_exit:
            train.main()
            mock_exit.assert_called_with(1)

        mock_validate.assert_called_once_with(mock_args)
        logger.info("Main function validation failure path verified")

    @patch("train.setup_argument_parser")
    @patch("sys.argv")
    def test_main_keyboard_interrupt(self, mock_argv, mock_parser_setup):
        """Test main function with keyboard interrupt."""
        mock_argv.__getitem__.return_value = [
            "train.py",
            "--model",
            "test",
            "--config",
            "test",
        ]

        mock_parser = Mock()
        mock_parser.parse_args.side_effect = KeyboardInterrupt()
        mock_parser_setup.return_value = mock_parser

        with patch("sys.exit") as mock_exit:
            train.main()
            mock_exit.assert_called_with(130)  # SIGINT exit code

        logger.info("Main function keyboard interrupt handling verified")

    @patch("train.TrainingPipeline")
    @patch("train.ConfigurationLoader")
    @patch("train.print_model_recommendations")
    @patch("train.validate_arguments")
    @patch("train.initialize_experiment_logging")
    @patch("train.setup_argument_parser")
    @patch("sys.argv")
    def test_main_pipeline_failure(
        self,
        mock_argv,
        mock_parser_setup,
        mock_init_logging,
        mock_validate,
        mock_print_rec,
        mock_config_loader_class,
        mock_pipeline_class,
    ):
        """Test main function with pipeline execution failure."""
        # Setup mocks for successful validation but failed pipeline
        mock_argv.__getitem__.return_value = [
            "train.py",
            "--model",
            "test",
            "--config",
            "test",
        ]

        mock_parser = Mock()
        mock_args = Mock()
        mock_parser.parse_args.return_value = mock_args
        mock_parser_setup.return_value = mock_parser

        mock_init_logging.return_value = "test_experiment_id"
        mock_validate.return_value = True

        mock_config_loader = Mock()
        mock_config_loader.load_and_validate_configs.return_value = (
            Mock(),
            Mock(),
        )
        mock_config_loader_class.return_value = mock_config_loader

        mock_pipeline = Mock()
        mock_pipeline.execute.return_value = False  # Pipeline fails
        mock_pipeline_class.return_value = mock_pipeline

        with patch("sys.exit") as mock_exit:
            train.main()
            mock_exit.assert_called_with(1)

        mock_pipeline.execute.assert_called_once()
        logger.info("Main function pipeline failure path verified")

    @patch("train.setup_argument_parser")
    @patch("sys.argv")
    def test_main_unexpected_exception(self, mock_argv, mock_parser_setup):
        """Test main function with unexpected exception."""
        mock_argv.__getitem__.return_value = [
            "train.py",
            "--model",
            "test",
            "--config",
            "test",
        ]

        mock_parser_setup.side_effect = Exception("Unexpected error")

        with patch("sys.exit") as mock_exit:
            train.main()
            mock_exit.assert_called_with(1)

        logger.info("Main function unexpected exception handling verified")


if __name__ == "__main__":
    unittest.main(verbosity=2)
