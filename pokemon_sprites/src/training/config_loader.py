"""
Configuration loading and validation for Pokemon sprite generation training.

This module handles loading, validation, and management of training and model
configurations from JSON files and command-line arguments.
"""

import sys
from pathlib import Path
from typing import Tuple

from config.settings import create_experiment_config, get_data_root_dir
from core.logging_config import get_logger

# Add src to path
current_dir = Path(__file__).parent
src_path = str(current_dir.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)


logger = get_logger(__name__)


class ConfigurationLoader:
    """Handles loading and validation of training configurations."""

    def __init__(self):
        """Initialize configuration loader."""
        self.supported_models = [
            "lightweight-baseline",
            "sprite-optimized",
            "sprite-production",
            "transformer-enhanced",
        ]
        self.supported_configs = ["test", "development", "production"]

    def load_and_validate_configs(
        self, model_name: str, config_type: str
    ) -> Tuple:
        """
        Load and validate model and training configurations.

        Args:
            model_name: Name of the model to train
            config_type: Type of training configuration

        Returns:
            Tuple of (model_config, training_config)

        Raises:
            ValueError: If configurations are invalid
        """
        try:
            # Validate model name
            if model_name not in self.supported_models:
                available = ", ".join(self.supported_models)
                raise ValueError(
                    f"Unsupported model '{model_name}'. "
                    f"Supported models: {available}"
                )

            # Validate config type
            if config_type not in self.supported_configs:
                available = ", ".join(self.supported_configs)
                raise ValueError(
                    f"Unsupported config type '{config_type}'. "
                    f"Supported configs: {available}"
                )

            # Load configurations using existing settings
            model_config, training_config = create_experiment_config(
                model_name, config_type
            )

            # Additional validation
            self._validate_model_config(model_config)
            self._validate_training_config(training_config)

            logger.info(
                f"Loaded configuration for {model_name} with {config_type} settings"
            )
            return model_config, training_config

        except Exception as e:
            logger.error(f"Configuration loading failed: {e}")
            raise

    def _validate_model_config(self, model_config) -> None:
        """
        Validate model configuration.

        Args:
            model_config: Model configuration object

        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ["name", "architecture", "parameters"]

        for field in required_fields:
            if not hasattr(model_config, field):
                raise ValueError(
                    f"Model config missing required field: {field}"
                )

        # Validate architecture-specific parameters
        if model_config.architecture == "pix2pix":
            self._validate_pix2pix_config(model_config)
        elif model_config.architecture == "pix2pix_transformer":
            self._validate_transformer_config(model_config)

    def _validate_pix2pix_config(self, model_config) -> None:
        """Validate Pix2Pix model configuration."""
        params = model_config.parameters

        # Generator validation
        if "generator" not in params:
            raise ValueError("Pix2Pix config missing generator parameters")

        gen_params = params["generator"]
        required_gen_fields = [
            "input_channels",
            "output_channels",
            "ngf",
            "n_blocks",
        ]

        for field in required_gen_fields:
            if field not in gen_params:
                raise ValueError(f"Generator config missing field: {field}")

        # Discriminator validation
        if "discriminator" not in params:
            raise ValueError("Pix2Pix config missing discriminator parameters")

        disc_params = params["discriminator"]
        required_disc_fields = ["input_channels", "ndf", "n_layers"]

        for field in required_disc_fields:
            if field not in disc_params:
                raise ValueError(
                    f"Discriminator config missing field: {field}"
                )

    def _validate_transformer_config(self, model_config) -> None:
        """Validate transformer-enhanced model configuration."""
        self._validate_pix2pix_config(model_config)  # Base validation

        gen_params = model_config.parameters["generator"]
        transformer_fields = ["transformer_layers", "attention_heads"]

        for field in transformer_fields:
            if field not in gen_params:
                raise ValueError(f"Transformer config missing field: {field}")

    def _validate_training_config(self, training_config) -> None:
        """
        Validate training configuration.

        Args:
            training_config: Training configuration object

        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = [
            "epochs",
            "batch_size",
            "learning_rate",
            "eval_frequency",
            "save_frequency",
        ]

        for field in required_fields:
            if not hasattr(training_config, field):
                raise ValueError(
                    f"Training config missing required field: {field}"
                )

        # Validate ranges
        if training_config.epochs <= 0:
            raise ValueError("Epochs must be positive")

        if training_config.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        if training_config.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

    def get_available_models(self) -> list:
        """
        Get list of available models.

        Returns:
            List of supported model names
        """
        return self.supported_models.copy()

    def get_available_configs(self) -> list:
        """
        Get list of available training configurations.

        Returns:
            List of supported configuration types
        """
        return self.supported_configs.copy()

    def get_data_root_dir(self) -> Path:
        """
        Get the data root directory.

        Returns:
            Path to data root directory
        """
        return Path(get_data_root_dir())

    def get_model_info(self, model_name: str) -> dict:
        """
        Get detailed information about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model information
        """
        model_info = {
            "lightweight-baseline": {
                "description": "Lightweight baseline for quick experimentation",
                "architecture": "pix2pix",
                "features": [
                    "Fast training",
                    "Minimal parameters",
                    "Good for prototyping",
                ],
                "recommended_config": "development",
                "use_transfer_learning": False,
                "use_curriculum": False,
            },
            "sprite-optimized": {
                "description": "State-of-the-art configuration optimized for pixel art sprites",
                "architecture": "pix2pix",
                "features": [
                    "Attention mechanisms",
                    "Spectral normalization",
                    "Advanced loss functions",
                ],
                "recommended_config": "production",
                "use_transfer_learning": True,
                "use_curriculum": True,
            },
            "transformer-enhanced": {
                "description": "Advanced transformer-enhanced architecture",
                "architecture": "pix2pix_transformer",
                "features": [
                    "Transformer layers",
                    "Multi-head attention",
                    "Positional encoding",
                ],
                "recommended_config": "production",
                "use_transfer_learning": True,
                "use_curriculum": True,
            },
        }

        return model_info.get(model_name, {})

    def print_configuration_summary(
        self, model_config, training_config
    ) -> None:
        """
        Print a summary of the loaded configuration.

        Args:
            model_config: Model configuration
            training_config: Training configuration
        """
        logger.info("=" * 60)
        logger.info("CONFIGURATION SUMMARY")
        logger.info("=" * 60)

        # Model information
        logger.info(f"Model: {model_config.name}")
        logger.info(f"Architecture: {model_config.architecture}")
        logger.info(f"Description: {model_config.description}")

        # Training information
        training_description = getattr(
            training_config, "description", "Standard training configuration"
        )
        logger.info(f"Training Config: {training_description}")
        logger.info(f"Epochs: {training_config.epochs}")
        logger.info(f"Batch Size: {training_config.batch_size}")
        logger.info(f"Learning Rate: {training_config.learning_rate}")

        # Advanced features
        model_info = self.get_model_info(model_config.name)
        if model_info:
            logger.info(
                f"Transfer Learning: {model_info.get('use_transfer_learning', False)}"
            )
            logger.info(
                f"Curriculum Learning: {model_info.get('use_curriculum', False)}"
            )

        logger.info("=" * 60)
