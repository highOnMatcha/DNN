#!/usr/bin/env python3
"""
Comprehensive tests for config/settings.py module.
Tests all configuration classes, utility functions, and edge cases.
"""

import unittest
import tempfile
import os
import shutil
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# Import the classes and functions we want to test
from src.config.settings import (
    TrainingConfig,
    ModelConfig,
    get_model_config,
    get_training_config,
    save_experiment_config,
    create_experiment_config,
    list_available_models,
    get_available_training_configs
)


class TestTrainingConfig(unittest.TestCase):
    """Test TrainingConfig dataclass."""

    def test_training_config_initialization(self):
        """Test TrainingConfig initialization with default values."""
        config = TrainingConfig()
        
        # Test default values based on actual dataclass
        self.assertEqual(config.epochs, 100)
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.learning_rate, 0.0002)
        self.assertEqual(config.beta1, 0.5)
        self.assertEqual(config.image_size, 64)
        self.assertTrue(config.augment_data)

    def test_training_config_custom_values(self):
        """Test TrainingConfig initialization with custom values."""
        custom_config = TrainingConfig(
            epochs=50,
            batch_size=32,
            learning_rate=0.001,
            lambda_l1=50.0,
            eval_frequency=10
        )
        
        self.assertEqual(custom_config.epochs, 50)
        self.assertEqual(custom_config.batch_size, 32)
        self.assertEqual(custom_config.learning_rate, 0.001)
        self.assertEqual(custom_config.lambda_l1, 50.0)
        self.assertEqual(custom_config.eval_frequency, 10)

    def test_training_config_validation(self):
        """Test TrainingConfig validation of values."""
        # Test valid config
        config = TrainingConfig(epochs=10, batch_size=8, learning_rate=0.0001)
        self.assertIsNotNone(config)
        
        # These would typically be validated by the dataclass
        self.assertGreater(config.epochs, 0)
        self.assertGreater(config.batch_size, 0)
        self.assertGreater(config.learning_rate, 0)

    def test_training_config_to_dict(self):
        """Test TrainingConfig conversion to dictionary."""
        config = TrainingConfig(epochs=25, batch_size=16)
        config_dict = asdict(config)
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['epochs'], 25)
        self.assertEqual(config_dict['batch_size'], 16)
        self.assertIn('learning_rate', config_dict)
        self.assertIn('lambda_l1', config_dict)

    def test_training_config_loss_parameters(self):
        """Test TrainingConfig loss function parameters."""
        config = TrainingConfig(
            lambda_l1=200.0,
            lambda_cycle=20.0,
            lambda_identity=1.0
        )
        
        self.assertEqual(config.lambda_l1, 200.0)
        self.assertEqual(config.lambda_cycle, 20.0)
        self.assertEqual(config.lambda_identity, 1.0)

    def test_training_config_hardware_settings(self):
        """Test TrainingConfig hardware configuration."""
        config = TrainingConfig(
            device="cuda",
            num_workers=8,
            pin_memory=False
        )
        
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.num_workers, 8)
        self.assertFalse(config.pin_memory)


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig dataclass."""

    def test_model_config_initialization(self):
        """Test ModelConfig initialization with required parameters."""
        config = ModelConfig(
            name="test_model",
            architecture="unet",
            output_dir="/test/dir"
        )
        
        self.assertEqual(config.name, "test_model")
        self.assertEqual(config.architecture, "unet")
        self.assertEqual(config.output_dir, "/test/dir")
        self.assertEqual(config.input_channels, 3)
        self.assertEqual(config.output_channels, 3)

    def test_model_config_with_custom_parameters(self):
        """Test ModelConfig with custom parameters."""
        config = ModelConfig(
            name="custom_model",
            architecture="pix2pix",
            output_dir="/custom/path",
            description="Custom test model",
            input_channels=1,
            output_channels=3,
            image_size=128
        )
        
        self.assertEqual(config.name, "custom_model")
        self.assertEqual(config.architecture, "pix2pix")
        self.assertEqual(config.description, "Custom test model")
        self.assertEqual(config.input_channels, 1)
        self.assertEqual(config.output_channels, 3)
        self.assertEqual(config.image_size, 128)

    def test_model_config_post_init(self):
        """Test ModelConfig __post_init__ method."""
        config = ModelConfig(
            name="test",
            architecture="unet",
            output_dir="/test"
        )
        
        # Check that parameters dict is initialized
        self.assertIsNotNone(config.parameters)
        self.assertIsInstance(config.parameters, dict)

    def test_model_config_with_parameters(self):
        """Test ModelConfig with custom parameters dict."""
        custom_params = {"filters": 64, "layers": 4}
        config = ModelConfig(
            name="param_test",
            architecture="unet",
            output_dir="/test",
            parameters=custom_params
        )
        
        self.assertEqual(config.parameters, custom_params)
        self.assertEqual(config.parameters["filters"], 64)

    def test_model_config_architectures(self):
        """Test ModelConfig with different architectures."""
        architectures = ["pix2pix", "unet", "cyclegan", "ddpm"]
        
        for arch in architectures:
            config = ModelConfig(
                name=f"test_{arch}",
                architecture=arch,
                output_dir=f"/test/{arch}"
            )
            self.assertEqual(config.architecture, arch)
            self.assertIsNotNone(config)


class TestConfigFunctions(unittest.TestCase):
    """Test configuration utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_get_model_config(self):
        """Test get_model_config function."""
        try:
            # Test with a potentially valid model name
            config = get_model_config("lightweight-baseline")
            
            # Function should return ModelConfig or None
            self.assertTrue(config is None or isinstance(config, ModelConfig))
            
            model_config_success = True
        except Exception as e:
            print(f"Get model config error: {e}")
            model_config_success = False
        
        self.assertTrue(model_config_success)

    def test_get_training_config(self):
        """Test get_training_config function."""
        try:
            config = get_training_config()
            self.assertIsNotNone(config)
            
            # Test that it returns a TrainingConfig instance
            self.assertTrue(hasattr(config, 'epochs'))
            self.assertTrue(hasattr(config, 'batch_size'))
            self.assertTrue(hasattr(config, 'learning_rate'))
            training_config_success = True
        except Exception as e:
            print(f"Get training config error: {e}")
            training_config_success = False
        
        self.assertTrue(training_config_success)

    def test_get_training_config_types(self):
        """Test get_training_config with different config types."""
        config_types = ["development", "production", "test"]
        
        for config_type in config_types:
            try:
                config = get_training_config(config_type)
                self.assertIsNotNone(config)
                self.assertIsInstance(config, TrainingConfig)
                type_success = True
            except Exception as e:
                print(f"Config type {config_type} error: {e}")
                type_success = False
            
            self.assertTrue(type_success)

    def test_list_available_models(self):
        """Test list_available_models function."""
        try:
            models = list_available_models()
            self.assertIsNotNone(models)
            
            # Should return a dictionary
            self.assertIsInstance(models, dict)
            
            list_models_success = True
        except Exception as e:
            print(f"List models error: {e}")
            list_models_success = False
        
        self.assertTrue(list_models_success)

    def test_get_available_training_configs(self):
        """Test get_available_training_configs function."""
        try:
            configs = get_available_training_configs()
            self.assertIsNotNone(configs)
            
            # Should return a list
            self.assertIsInstance(configs, list)
            
            training_configs_success = True
        except Exception as e:
            print(f"Get training configs error: {e}")
            training_configs_success = False
        
        self.assertTrue(training_configs_success)

    def test_create_experiment_config(self):
        """Test create_experiment_config function."""
        try:
            model_config, training_config = create_experiment_config(
                model_name="lightweight-baseline",
                config_type="development"
            )
            self.assertIsNotNone(model_config)
            self.assertIsNotNone(training_config)
            
            create_config_success = True
        except Exception as e:
            print(f"Create experiment config error: {e}")
            create_config_success = False
        
        self.assertTrue(create_config_success)

    def test_save_experiment_config(self):
        """Test save_experiment_config function."""
        try:
            training_config = TrainingConfig(epochs=25, batch_size=16)
            model_config = ModelConfig(
                name="test_model",
                architecture="unet",
                output_dir="/test/dir"
            )
            config_file = self.test_dir / "save_test.json"
            
            # Save the config
            save_experiment_config(model_config, training_config, str(config_file))
            
            # Verify file was created
            self.assertTrue(config_file.exists())
            
            save_config_success = True
        except Exception as e:
            print(f"Save config error: {e}")
            save_config_success = False
        
        self.assertTrue(save_config_success)


class TestConfigurationIntegration(unittest.TestCase):
    """Test configuration integration scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_config_with_model_integration(self):
        """Test config integration with model configuration."""
        try:
            # Get both configs
            training_config = get_training_config()
            model_config = get_model_config("basic_unet")
            
            self.assertIsNotNone(training_config)
            # model_config might be None if model doesn't exist
            
            # Test that they can work together
            self.assertTrue(hasattr(training_config, 'epochs'))
            
            integration_success = True
        except Exception as e:
            print(f"Integration error: {e}")
            integration_success = False
        
        self.assertTrue(integration_success)

    def test_config_serialization_workflow(self):
        """Test configuration serialization workflow."""
        try:
            # Create configs with various parameters
            configs_to_test = [
                TrainingConfig(epochs=10, batch_size=8),
                TrainingConfig(epochs=50, batch_size=16, learning_rate=0.001),
                TrainingConfig(epochs=100, batch_size=32, lambda_l1=200.0)
            ]
            
            for config in configs_to_test:
                # Validate basic properties
                self.assertGreater(config.epochs, 0)
                self.assertGreater(config.batch_size, 0)
                self.assertGreater(config.learning_rate, 0)
                
                # Test serialization
                config_dict = asdict(config)
                self.assertIsInstance(config_dict, dict)
                self.assertIn('epochs', config_dict)
            
            workflow_success = True
        except Exception as e:
            print(f"Workflow error: {e}")
            workflow_success = False
        
        self.assertTrue(workflow_success)

    def test_config_parameter_combinations(self):
        """Test various parameter combinations."""
        try:
            # Test different loss function combinations
            loss_configs = [
                TrainingConfig(lambda_l1=100.0, lambda_cycle=10.0),
                TrainingConfig(lambda_l1=50.0, lambda_identity=2.0),
                TrainingConfig(lambda_cycle=20.0, lambda_identity=1.0)
            ]
            
            for config in loss_configs:
                self.assertIsNotNone(config)
                self.assertGreater(config.lambda_l1, 0)
            
            # Test different hardware configurations
            hardware_configs = [
                TrainingConfig(device="cpu", num_workers=2),
                TrainingConfig(device="cuda", pin_memory=True),
                TrainingConfig(device="auto", num_workers=8)
            ]
            
            for config in hardware_configs:
                self.assertIsNotNone(config)
                self.assertIn(config.device, ["cpu", "cuda", "auto"])
            
            combinations_success = True
        except Exception as e:
            print(f"Parameter combinations error: {e}")
            combinations_success = False
        
        self.assertTrue(combinations_success)


class TestConfigurationEdgeCases(unittest.TestCase):
    """Test configuration edge cases and error handling."""

    def test_config_with_extreme_values(self):
        """Test config with extreme values."""
        try:
            # Test with very small values
            small_config = TrainingConfig(
                epochs=1,
                batch_size=1,
                learning_rate=1e-6
            )
            self.assertIsNotNone(small_config)
            
            # Test with larger values
            large_config = TrainingConfig(
                epochs=1000,
                batch_size=128,
                learning_rate=0.1,
                lambda_l1=1000.0
            )
            self.assertIsNotNone(large_config)
            
            extreme_values_success = True
        except Exception as e:
            print(f"Extreme values error: {e}")
            extreme_values_success = False
        
        self.assertTrue(extreme_values_success)

    def test_model_config_edge_cases(self):
        """Test ModelConfig edge cases."""
        try:
            # Test with minimal required parameters
            minimal_config = ModelConfig(
                name="minimal",
                architecture="unet",
                output_dir="/tmp"
            )
            self.assertIsNotNone(minimal_config)
            
            # Test with maximum parameters
            maximal_config = ModelConfig(
                name="maximal_test_model_with_long_name",
                architecture="cyclegan",
                output_dir="/very/long/path/to/output/directory",
                description="A very detailed description of this model configuration",
                input_channels=1,
                output_channels=16,
                image_size=512,
                parameters={"complex": {"nested": {"parameters": True}}}
            )
            self.assertIsNotNone(maximal_config)
            
            model_edge_cases_success = True
        except Exception as e:
            print(f"Model edge cases error: {e}")
            model_edge_cases_success = False
        
        self.assertTrue(model_edge_cases_success)

    def test_config_with_invalid_model_names(self):
        """Test config functions with invalid model names."""
        try:
            # Test with non-existent model
            result = get_model_config("non_existent_model_12345")
            # Should return None for non-existent models
            self.assertTrue(result is None)
            
            invalid_model_handled = True
        except Exception as e:
            print(f"Invalid model name error: {e}")
            invalid_model_handled = False
        
        self.assertTrue(invalid_model_handled)

    def test_config_boundary_values(self):
        """Test config with boundary values."""
        try:
            # Test boundary values for various parameters
            boundary_configs = [
                TrainingConfig(train_split=0.0),
                TrainingConfig(train_split=1.0),
                TrainingConfig(beta1=0.0, beta2=1.0),
                TrainingConfig(weight_decay=0.0),
                TrainingConfig(eval_frequency=1, save_frequency=1)
            ]
            
            for config in boundary_configs:
                self.assertIsNotNone(config)
            
            boundary_values_success = True
        except Exception as e:
            print(f"Boundary values error: {e}")
            boundary_values_success = False
        
        self.assertTrue(boundary_values_success)


class TestConfigurationEnvironmentIntegration(unittest.TestCase):
    """Test configuration integration with environment variables."""

    def test_config_with_different_environments(self):
        """Test config behavior in different environments."""
        try:
            # Test development config
            dev_config = get_training_config("development")
            self.assertIsNotNone(dev_config)
            
            # Test production config (if available)
            try:
                prod_config = get_training_config("production")
                self.assertIsNotNone(prod_config)
            except:
                # Production config might not be available
                pass
            
            env_integration_success = True
        except Exception as e:
            print(f"Environment integration error: {e}")
            env_integration_success = False
        
        self.assertTrue(env_integration_success)

    def test_config_logging_parameters(self):
        """Test configuration logging parameters."""
        try:
            config = TrainingConfig(
                log_dir="./custom_logs",
                tensorboard_log=False,
                wandb_log=True,
                wandb_project="custom_project"
            )
            
            self.assertEqual(config.log_dir, "./custom_logs")
            self.assertFalse(config.tensorboard_log)
            self.assertTrue(config.wandb_log)
            self.assertEqual(config.wandb_project, "custom_project")
            
            logging_params_success = True
        except Exception as e:
            print(f"Logging parameters error: {e}")
            logging_params_success = False
        
        self.assertTrue(logging_params_success)

    def test_config_data_parameters(self):
        """Test configuration data processing parameters."""
        try:
            config = TrainingConfig(
                image_size=256,
                max_samples=1000,
                train_split=0.9,
                augment_data=False
            )
            
            self.assertEqual(config.image_size, 256)
            self.assertEqual(config.max_samples, 1000)
            self.assertEqual(config.train_split, 0.9)
            self.assertFalse(config.augment_data)
            
            data_params_success = True
        except Exception as e:
            print(f"Data parameters error: {e}")
            data_params_success = False
        
        self.assertTrue(data_params_success)


if __name__ == '__main__':
    unittest.main()
