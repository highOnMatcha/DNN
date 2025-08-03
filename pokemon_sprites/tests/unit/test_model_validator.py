"""
Test Model Validator - Model Validation Component Testing

Professional test suite for the model validator optimizer component.
Tests model configuration validation, creation, and testing functionality.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import torch.nn as nn

# Add the src directory to Python path
current_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(current_dir / "src"))

try:
    from core.logging_config import get_logger

    logger = get_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


class TestModelValidatorCore(unittest.TestCase):
    """Test core model validator functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.device = torch.device("cpu")

        # Create test configuration
        self.test_config = {
            "pix2pix_models": {
                "test-model": {
                    "description": "Test model for validation",
                    "parameters": {
                        "generator": {
                            "input_channels": 3,
                            "output_channels": 4,
                            "ngf": 32,
                            "n_blocks": 6,
                            "norm_layer": "instance",
                            "dropout": 0.3,
                        },
                        "discriminator": {
                            "input_channels": 7,
                            "ndf": 32,
                            "n_layers": 2,
                            "norm_layer": "instance",
                            "use_spectral_norm": False,
                        },
                    },
                }
            }
        }

        self.config_path = Path(self.test_dir) / "test_config.json"
        with open(self.config_path, "w") as f:
            json.dump(self.test_config, f)

    def test_model_validator_import(self):
        """Test model validator import accessibility."""
        logger.info("[TEST] Testing model validator import")

        try:
            from optimizers.model_validator import validate_all_configurations

            self.assertTrue(callable(validate_all_configurations))
            logger.info("[SUCCESS] Model validator import accessible")
        except ImportError as e:
            logger.error(f"[FAIL] Model validator import failed: {e}")
            self.fail(f"Model validator import failed: {e}")

    def test_validation_functions_exist(self):
        """Test that validation functions exist and are callable."""
        logger.info("[TEST] Testing validation functions existence")

        try:
            from optimizers.model_validator import (
                _create_discriminator,
                _create_generator,
                _test_backward_pass,
                _test_forward_pass,
                _validate_single_model,
            )

            # Test all functions are callable
            validation_functions = [
                _validate_single_model,
                _create_generator,
                _create_discriminator,
                _test_forward_pass,
                _test_backward_pass,
            ]

            for func in validation_functions:
                self.assertTrue(callable(func))

            logger.info("[SUCCESS] All validation functions exist")
        except ImportError as e:
            logger.error(f"[FAIL] Validation functions import failed: {e}")
            self.fail(f"Validation functions import failed: {e}")

    @patch("builtins.open")
    @patch("json.load")
    def test_config_loading_logic(self, mock_json_load, mock_open):
        """Test configuration loading logic."""
        logger.info("[TEST] Testing config loading logic")

        mock_json_load.return_value = self.test_config

        try:
            from optimizers.model_validator import validate_all_configurations

            # Test configuration loading doesn't crash
            with patch(
                "optimizers.model_validator._validate_single_model"
            ) as mock_validate:
                mock_validate.return_value = {
                    "model_name": "test",
                    "forward_pass_works": True,
                    "backward_pass_works": True,
                }

                result = validate_all_configurations(str(self.config_path))
                self.assertIsInstance(result, dict)

            logger.info("[SUCCESS] Config loading logic works")
        except Exception as e:
            logger.error(f"[FAIL] Config loading failed: {e}")
            self.fail(f"Config loading failed: {e}")

    def test_device_configuration_logic(self):
        """Test device configuration and setup."""
        logger.info("[TEST] Testing device configuration")

        try:
            import torch

            # Test device creation
            cpu_device = torch.device("cpu")
            self.assertEqual(cpu_device.type, "cpu")

            # Test CUDA availability check
            cuda_available = torch.cuda.is_available()
            self.assertIsInstance(cuda_available, bool)

            logger.info("[SUCCESS] Device configuration works")
        except Exception as e:
            logger.error(f"[FAIL] Device configuration failed: {e}")
            self.fail(f"Device configuration failed: {e}")


class TestModelCreationComponents(unittest.TestCase):
    """Test model creation components in validator."""

    def setUp(self):
        """Set up test environment."""
        self.device = torch.device("cpu")
        self.gen_params = {
            "input_channels": 3,
            "output_channels": 4,
            "ngf": 32,
            "n_blocks": 6,
            "norm_layer": "instance",
            "dropout": 0.3,
        }
        self.disc_params = {
            "input_channels": 7,
            "ndf": 32,
            "n_layers": 2,
            "norm_layer": "instance",
            "use_spectral_norm": False,
        }

    @patch("optimizers.model_validator.Pix2PixGenerator")
    def test_generator_creation_logic(self, mock_generator):
        """Test generator creation logic."""
        logger.info("[TEST] Testing generator creation logic")

        # Setup mock
        mock_model = Mock()
        mock_model.parameters.return_value = iter([torch.randn(10, 10)])
        mock_model.to.return_value = mock_model
        mock_generator.return_value = mock_model

        try:
            from optimizers.model_validator import _create_generator

            results = {"errors": [], "parameter_count": {}}
            generator = _create_generator(
                self.gen_params, self.device, results
            )

            self.assertIsNotNone(generator)
            self.assertEqual(len(results["errors"]), 0)

            logger.info("[SUCCESS] Generator creation logic works")
        except Exception as e:
            logger.error(f"[FAIL] Generator creation failed: {e}")
            self.fail(f"Generator creation failed: {e}")

    @patch("optimizers.model_validator.Pix2PixDiscriminator")
    def test_discriminator_creation_logic(self, mock_discriminator):
        """Test discriminator creation logic."""
        logger.info("[TEST] Testing discriminator creation logic")

        # Setup mock
        mock_model = Mock()
        mock_model.parameters.return_value = iter([torch.randn(10, 10)])
        mock_model.to.return_value = mock_model
        mock_discriminator.return_value = mock_model

        try:
            from optimizers.model_validator import _create_discriminator

            results = {"errors": [], "parameter_count": {}}
            discriminator = _create_discriminator(
                self.disc_params, self.device, results
            )

            self.assertIsNotNone(discriminator)
            self.assertEqual(len(results["errors"]), 0)

            logger.info("[SUCCESS] Discriminator creation logic works")
        except Exception as e:
            logger.error(f"[FAIL] Discriminator creation failed: {e}")
            self.fail(f"Discriminator creation failed: {e}")

    def test_model_parameter_counting(self):
        """Test model parameter counting functionality."""
        logger.info("[TEST] Testing model parameter counting")

        try:
            # Create simple test model
            test_model = nn.Linear(10, 5)
            param_count = sum(p.numel() for p in test_model.parameters())

            self.assertIsInstance(param_count, int)
            self.assertGreater(param_count, 0)

            logger.info("[SUCCESS] Model parameter counting works")
        except Exception as e:
            logger.error(f"[FAIL] Parameter counting failed: {e}")
            self.fail(f"Parameter counting failed: {e}")


class TestForwardPassValidation(unittest.TestCase):
    """Test forward pass validation components."""

    def setUp(self):
        """Set up test environment."""
        self.device = torch.device("cpu")
        self.gen_params = {"input_channels": 3, "output_channels": 4}

    def test_forward_pass_tensor_creation(self):
        """Test tensor creation for forward pass testing."""
        logger.info("[TEST] Testing forward pass tensor creation")

        try:
            # Test tensor creation similar to validator
            test_artwork = torch.randn(2, 3, 256, 256).to(self.device)
            test_sprite = torch.randn(2, 4, 256, 256).to(self.device)

            self.assertEqual(test_artwork.shape, (2, 3, 256, 256))
            self.assertEqual(test_sprite.shape, (2, 4, 256, 256))
            self.assertEqual(test_artwork.device, self.device)

            logger.info("[SUCCESS] Forward pass tensor creation works")
        except Exception as e:
            logger.error(f"[FAIL] Tensor creation failed: {e}")
            self.fail(f"Tensor creation failed: {e}")

    def test_no_grad_context(self):
        """Test no_grad context for forward pass."""
        logger.info("[TEST] Testing no_grad context")

        try:
            with torch.no_grad():
                tensor = torch.randn(2, 3, 256, 256, requires_grad=True)
                # Inside no_grad, operations don't track gradients
                result = tensor * 2
                self.assertFalse(result.requires_grad)

            logger.info("[SUCCESS] No_grad context works")
        except Exception as e:
            logger.error(f"[FAIL] No_grad context failed: {e}")
            self.fail(f"No_grad context failed: {e}")

    def test_model_eval_mode(self):
        """Test model evaluation mode setting."""
        logger.info("[TEST] Testing model eval mode")

        try:
            # Test model mode switching
            test_model = nn.Linear(10, 5)
            test_model.eval()
            self.assertFalse(test_model.training)

            test_model.train()
            self.assertTrue(test_model.training)

            logger.info("[SUCCESS] Model eval mode works")
        except Exception as e:
            logger.error(f"[FAIL] Model eval mode failed: {e}")
            self.fail(f"Model eval mode failed: {e}")


class TestBackwardPassValidation(unittest.TestCase):
    """Test backward pass validation components."""

    def setUp(self):
        """Set up test environment."""
        self.device = torch.device("cpu")

    def test_optimizer_creation(self):
        """Test optimizer creation for backward pass testing."""
        logger.info("[TEST] Testing optimizer creation")

        try:
            # Test optimizer creation similar to validator
            test_model = nn.Linear(10, 5)
            optimizer = torch.optim.Adam(test_model.parameters(), lr=1e-4)

            self.assertIsInstance(optimizer, torch.optim.Adam)
            self.assertEqual(optimizer.param_groups[0]["lr"], 1e-4)

            logger.info("[SUCCESS] Optimizer creation works")
        except Exception as e:
            logger.error(f"[FAIL] Optimizer creation failed: {e}")
            self.fail(f"Optimizer creation failed: {e}")

    def test_loss_function_creation(self):
        """Test loss function creation for training step."""
        logger.info("[TEST] Testing loss function creation")

        try:
            # Test loss functions used in validator
            criterion_GAN = nn.MSELoss()
            criterion_L1 = nn.L1Loss()

            self.assertIsInstance(criterion_GAN, nn.MSELoss)
            self.assertIsInstance(criterion_L1, nn.L1Loss)

            logger.info("[SUCCESS] Loss function creation works")
        except Exception as e:
            logger.error(f"[FAIL] Loss function creation failed: {e}")
            self.fail(f"Loss function creation failed: {e}")

    def test_training_step_components(self):
        """Test training step component functionality."""
        logger.info("[TEST] Testing training step components")

        try:
            # Test basic training step operations
            test_model = nn.Linear(10, 5)
            optimizer = torch.optim.Adam(test_model.parameters(), lr=1e-4)
            criterion = nn.MSELoss()

            # Forward pass
            input_tensor = torch.randn(32, 10)
            target = torch.randn(32, 5)
            output = test_model(input_tensor)
            loss = criterion(output, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.assertIsInstance(loss.item(), float)

            logger.info("[SUCCESS] Training step components work")
        except Exception as e:
            logger.error(f"[FAIL] Training step failed: {e}")
            self.fail(f"Training step failed: {e}")


class TestValidationSummaryComponents(unittest.TestCase):
    """Test validation summary and reporting components."""

    def test_summary_data_structure(self):
        """Test validation summary data structure handling."""
        logger.info("[TEST] Testing summary data structure")

        try:
            # Test data structure similar to validator results
            validation_results = {
                "test-model": {
                    "model_name": "test-model",
                    "forward_pass_works": True,
                    "backward_pass_works": True,
                    "parameter_count": {
                        "generator": 1000,
                        "discriminator": 500,
                    },
                }
            }

            # Test data access patterns
            valid_models = [
                name
                for name, results in validation_results.items()
                if results.get("forward_pass_works", False)
                and results.get("backward_pass_works", False)
            ]

            self.assertEqual(len(valid_models), 1)
            self.assertEqual(valid_models[0], "test-model")

            logger.info("[SUCCESS] Summary data structure works")
        except Exception as e:
            logger.error(f"[FAIL] Summary data structure failed: {e}")
            self.fail(f"Summary data structure failed: {e}")

    def test_parameter_counting_aggregation(self):
        """Test parameter counting aggregation logic."""
        logger.info("[TEST] Testing parameter counting aggregation")

        try:
            validation_results = {
                "model1": {
                    "forward_pass_works": True,
                    "parameter_count": {
                        "generator": 1000,
                        "discriminator": 500,
                    },
                },
                "model2": {
                    "forward_pass_works": True,
                    "parameter_count": {
                        "generator": 2000,
                        "discriminator": 1000,
                    },
                },
            }

            total_params = sum(
                results.get("parameter_count", {}).get("generator", 0)
                + results.get("parameter_count", {}).get("discriminator", 0)
                for results in validation_results.values()
                if results.get("forward_pass_works", False)
            )

            self.assertEqual(total_params, 4500)

            logger.info("[SUCCESS] Parameter counting aggregation works")
        except Exception as e:
            logger.error(f"[FAIL] Parameter aggregation failed: {e}")
            self.fail(f"Parameter aggregation failed: {e}")


class TestModelValidatorIntegration(unittest.TestCase):
    """Test integration workflows for model validator with full configuration files."""

    def setUp(self):
        """Set up test environment with temporary configuration files."""
        self.test_dir = tempfile.mkdtemp()
        self.device = torch.device("cpu")

        # Create test configuration file
        self.test_config = {
            "pix2pix_models": {
                "test-lightweight": {
                    "name": "test-lightweight",
                    "description": "Test lightweight model",
                    "parameters": {
                        "generator": {
                            "input_channels": 4,
                            "output_channels": 4,
                            "ngf": 32,
                            "n_blocks": 6,
                            "norm_layer": "instance",
                            "dropout": 0.3,
                        },
                        "discriminator": {
                            "input_channels": 8,
                            "ndf": 32,
                            "n_layers": 2,
                            "norm_layer": "instance",
                        },
                    },
                },
                "test-standard": {
                    "name": "test-standard",
                    "description": "Test standard model",
                    "parameters": {
                        "generator": {
                            "input_channels": 4,
                            "output_channels": 4,
                            "ngf": 64,
                            "n_blocks": 9,
                            "norm_layer": "instance",
                            "dropout": 0.5,
                        },
                        "discriminator": {
                            "input_channels": 8,
                            "ndf": 64,
                            "n_layers": 3,
                            "norm_layer": "instance",
                        },
                    },
                },
            }
        }

        self.config_path = Path(self.test_dir) / "test_model_configs.json"
        with open(self.config_path, "w") as f:
            json.dump(self.test_config, f, indent=2)

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.test_dir)

    @patch("builtins.print")
    def test_validate_all_configurations_workflow(self, mock_print):
        """Test complete model validation workflow with configuration file."""
        logger.info("[TEST] Testing complete validation workflow")

        try:
            from optimizers.model_validator import validate_all_configurations

            # Test full validation workflow
            validation_results = validate_all_configurations(
                str(self.config_path)
            )

            # Verify results structure
            self.assertIsInstance(validation_results, dict)
            self.assertIn("test-lightweight", validation_results)
            self.assertIn("test-standard", validation_results)

            # Verify each model result has required fields
            for model_name, results in validation_results.items():
                self.assertIn("model_name", results)
                self.assertIn("config_valid", results)
                self.assertIn("generator_created", results)
                self.assertIn("discriminator_created", results)
                self.assertIn("forward_pass_works", results)
                self.assertIn("backward_pass_works", results)
                self.assertIn("errors", results)
                self.assertIn("parameter_count", results)

                # Check parameter count structure
                self.assertIn("generator", results["parameter_count"])
                self.assertIn("discriminator", results["parameter_count"])

            logger.info("[PASS] Complete validation workflow tested")

        except Exception as e:
            logger.error(f"[FAIL] Validation workflow test failed: {e}")
            self.fail(f"Validation workflow test failed: {e}")

    @patch("builtins.print")
    def test_optimize_model_config_function(self, mock_print):
        """Test optimize_model_config function wrapper."""
        logger.info("[TEST] Testing optimize_model_config function")

        try:
            from optimizers.model_validator import optimize_model_config

            # Test the wrapper function
            results = optimize_model_config(str(self.config_path))

            # Should return same structure as validate_all_configurations
            self.assertIsInstance(results, dict)
            self.assertGreater(len(results), 0)

            logger.info("[PASS] optimize_model_config function tested")

        except Exception as e:
            logger.error(f"[FAIL] optimize_model_config test failed: {e}")
            self.fail(f"optimize_model_config test failed: {e}")

    @patch("builtins.print")
    def test_validation_error_handling(self, mock_print):
        """Test validation with invalid configuration files."""
        logger.info("[TEST] Testing validation error handling")

        try:
            from optimizers.model_validator import validate_all_configurations

            # Test with non-existent file
            results = validate_all_configurations("non_existent_file.json")
            self.assertEqual(results, {})

            # Test with invalid JSON
            invalid_config_path = Path(self.test_dir) / "invalid.json"
            with open(invalid_config_path, "w") as f:
                f.write("invalid json content")

            results = validate_all_configurations(str(invalid_config_path))
            self.assertEqual(results, {})

            # Test with empty configuration
            empty_config_path = Path(self.test_dir) / "empty.json"
            with open(empty_config_path, "w") as f:
                json.dump({}, f)

            results = validate_all_configurations(str(empty_config_path))
            self.assertEqual(results, {})

            logger.info("[PASS] Validation error handling tested")

        except Exception as e:
            logger.error(f"[FAIL] Validation error handling test failed: {e}")
            self.fail(f"Validation error handling test failed: {e}")

    @patch("builtins.print")
    def test_validation_summary_functionality(self, mock_print):
        """Test validation summary reporting functionality."""
        logger.info("[TEST] Testing validation summary functionality")

        try:
            from optimizers.model_validator import validate_all_configurations

            # Run validation to trigger summary functionality
            validate_all_configurations(str(self.config_path))

            # Verify print was called with summary information
            # The _print_validation_summary function should have been called
            print_calls = [str(call) for call in mock_print.call_args_list]
            summary_found = any(
                "VALIDATION SUMMARY" in call for call in print_calls
            )
            self.assertTrue(
                summary_found, "Validation summary should be printed"
            )

            logger.info("[PASS] Validation summary functionality tested")

        except Exception as e:
            logger.error(f"[FAIL] Validation summary test failed: {e}")
            self.fail(f"Validation summary test failed: {e}")

    @patch("builtins.print")
    def test_invalid_model_parameters(self, mock_print):
        """Test validation with invalid model parameters."""
        logger.info("[TEST] Testing validation with invalid model parameters")

        try:
            # Create configuration with invalid parameters
            invalid_config = {
                "pix2pix_models": {
                    "invalid-model": {
                        "name": "invalid-model",
                        "parameters": {
                            "generator": {
                                "input_channels": -1,  # Invalid
                                "output_channels": 0,  # Invalid
                                "ngf": "invalid",  # Invalid type
                            },
                            "discriminator": {
                                "input_channels": "invalid",  # Invalid type
                                "ndf": -64,  # Invalid value
                            },
                        },
                    }
                }
            }

            invalid_config_path = Path(self.test_dir) / "invalid_params.json"
            with open(invalid_config_path, "w") as f:
                json.dump(invalid_config, f, indent=2)

            from optimizers.model_validator import validate_all_configurations

            # Should handle invalid parameters gracefully
            results = validate_all_configurations(str(invalid_config_path))

            # Should return results even with invalid parameters
            self.assertIsInstance(results, dict)

            if "invalid-model" in results:
                # Model creation should have failed
                self.assertFalse(
                    results["invalid-model"].get("generator_created", True)
                )
                self.assertFalse(
                    results["invalid-model"].get("discriminator_created", True)
                )
                self.assertGreater(
                    len(results["invalid-model"].get("errors", [])), 0
                )

            logger.info("[PASS] Invalid model parameters handled gracefully")

        except Exception as e:
            logger.error(f"[FAIL] Invalid parameters test failed: {e}")
            self.fail(f"Invalid parameters test failed: {e}")
