"""
Configuration settings for Pokemon sprite generation training pipeline.

This module provides comprehensive configuration management for image-to-image
translation training, including model architecture parameters, training hyperparameters,
and dataset configuration. It supports both JSON-based external configuration
and programmatic configuration through dataclasses.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path


# Path to external model configurations
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "model_configs.json")


def get_models_root_dir() -> str:
    """
    Get the absolute path to the pokemon_sprites/models directory.
    
    This ensures all models are stored and loaded from the same location
    regardless of where the script is executed from.
    
    Returns:
        Absolute path to the pokemon_sprites/models directory.
    """
    # Get the path to this file (pokemon_sprites/src/config/settings.py)
    current_file = Path(__file__).resolve()
    # Go up to pokemon_sprites directory: pokemon_sprites/src/config -> pokemon_sprites/src -> pokemon_sprites
    project_root = current_file.parent.parent.parent
    # Create models directory path
    models_dir = project_root / "models"
    # Ensure the directory exists
    models_dir.mkdir(exist_ok=True)
    return str(models_dir)


def get_data_root_dir() -> str:
    """
    Get the absolute path to the pokemon_sprites/data directory.
    
    Returns:
        Absolute path to the pokemon_sprites/data directory.
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    data_dir = project_root / "data"
    return str(data_dir)


def load_model_configs() -> Dict[str, Any]:
    """
    Load model configurations from JSON file.
    
    Attempts to load configuration data from the JSON file. If the file
    doesn't exist, returns a default empty configuration structure.
    
    Returns:
        Dictionary containing model configurations with keys for
        different model architectures and training configurations.
    """
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Configuration file {CONFIG_FILE} not found.")
        return {
            "pix2pix_models": {},
            "unet_models": {},
            "cyclegan_models": {},
            "diffusion_models": {},
            "training_configs": {}
        }
    except json.JSONDecodeError as e:
        print(f"Error parsing configuration file {CONFIG_FILE}: {e}")
        return {}


@dataclass
class TrainingConfig:
    """
    Training configuration parameters.
    
    Contains all hyperparameters and settings needed for training
    image-to-image translation models.
    """
    # Training parameters
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 0.0
    
    # Loss function parameters
    lambda_l1: float = 100.0  # L1 loss weight for Pix2Pix
    lambda_cycle: float = 10.0  # Cycle consistency loss weight for CycleGAN
    lambda_identity: float = 0.5  # Identity loss weight for CycleGAN
    
    # Training schedule
    eval_frequency: int = 5
    save_frequency: int = 10
    log_frequency: int = 50
    
    # Data parameters
    image_size: int = 64
    max_samples: Optional[int] = None
    train_split: float = 0.8
    augment_data: bool = True
    
    # Hardware
    device: str = "auto"  # "auto", "cpu", "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Checkpointing
    resume_from_checkpoint: Optional[str] = None
    save_best_only: bool = False
    
    # Logging
    log_dir: str = "./logs"
    tensorboard_log: bool = True
    wandb_log: bool = True
    wandb_project: str = "pokemon-sprite-generation"


@dataclass 
class ModelConfig:
    """
    Model architecture configuration.
    
    Contains parameters specific to model architecture including
    layer sizes, normalization types, and architectural choices.
    """
    name: str
    architecture: str  # "pix2pix", "unet", "cyclegan", "ddpm"
    output_dir: str
    description: str = ""
    
    # General parameters
    input_channels: int = 3
    output_channels: int = 3
    image_size: int = 64
    
    # Architecture-specific parameters
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """
    Get model configuration by name.
    
    Args:
        model_name: Name of the model configuration to retrieve.
        
    Returns:
        ModelConfig object if found, None otherwise.
    """
    configs = load_model_configs()
    
    # Search through all model categories
    for category in ["pix2pix_models", "unet_models", "cyclegan_models", "diffusion_models"]:
        if category in configs and model_name in configs[category]:
            model_data = configs[category][model_name]
            return ModelConfig(
                name=model_data["name"],
                architecture=model_data["architecture"],
                output_dir=os.path.join(get_models_root_dir(), model_data["output_dir"].lstrip("./")),
                description=model_data.get("description", ""),
                parameters=model_data.get("parameters", {})
            )
    
    return None


def get_training_config(config_type: str = "development") -> TrainingConfig:
    """
    Get training configuration by type.
    
    Args:
        config_type: Type of training configuration ("test", "development", "production").
        
    Returns:
        TrainingConfig object with appropriate settings.
    """
    configs = load_model_configs()
    training_configs = configs.get("training_configs", {})
    
    if config_type in training_configs:
        config_data = training_configs[config_type]
        return TrainingConfig(
            epochs=config_data.get("epochs", 100),
            batch_size=config_data.get("batch_size", 16),
            learning_rate=config_data.get("learning_rate", 0.0002),
            eval_frequency=config_data.get("eval_frequency", 5),
            save_frequency=config_data.get("save_frequency", 10),
            max_samples=config_data.get("max_samples"),
        )
    else:
        # Return default development configuration
        return TrainingConfig()


def list_available_models() -> Dict[str, List[str]]:
    """
    List all available model configurations grouped by architecture.
    
    Returns:
        Dictionary mapping architecture names to lists of available model names.
    """
    configs = load_model_configs()
    result = {}
    
    for category in ["pix2pix_models", "unet_models", "cyclegan_models", "diffusion_models"]:
        if category in configs:
            architecture = category.replace("_models", "")
            result[architecture] = list(configs[category].keys())
    
    return result


def get_available_training_configs() -> List[str]:
    """
    Get list of available training configuration names.
    
    Returns:
        List of training configuration names.
    """
    configs = load_model_configs()
    return list(configs.get("training_configs", {}).keys())


def create_experiment_config(model_name: str, config_type: str = "development") -> Tuple[ModelConfig, TrainingConfig]:
    """
    Create complete experiment configuration.
    
    Args:
        model_name: Name of the model to use.
        config_type: Type of training configuration.
        
    Returns:
        Tuple of (ModelConfig, TrainingConfig).
        
    Raises:
        ValueError: If model_name is not found.
    """
    model_config = get_model_config(model_name)
    if model_config is None:
        available_models = list_available_models()
        raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
    
    training_config = get_training_config(config_type)
    
    # Update training config with model-specific image size if specified
    if "image_size" in model_config.parameters:
        training_config.image_size = model_config.parameters["image_size"]
    
    return model_config, training_config


def save_experiment_config(model_config: ModelConfig, training_config: TrainingConfig, 
                          output_file: str) -> None:
    """
    Save experiment configuration to file for reproducibility.
    
    Args:
        model_config: Model configuration to save.
        training_config: Training configuration to save.
        output_file: Path to output JSON file.
    """
    config_dict = {
        "model": {
            "name": model_config.name,
            "architecture": model_config.architecture,
            "output_dir": model_config.output_dir,
            "description": model_config.description,
            "parameters": model_config.parameters
        },
        "training": {
            "epochs": training_config.epochs,
            "batch_size": training_config.batch_size,
            "learning_rate": training_config.learning_rate,
            "image_size": training_config.image_size,
            "max_samples": training_config.max_samples,
            "device": training_config.device,
        },
        "metadata": {
            "created_at": str(Path(__file__).stat().st_mtime),
            "config_version": "1.0"
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(config_dict, f, indent=2)


# Pre-defined configuration shortcuts for common use cases
def get_test_config(model_name: str = "pix2pix-small") -> Tuple[ModelConfig, TrainingConfig]:
    """Get test configuration for quick experimentation."""
    return create_experiment_config(model_name, "test")


def get_development_config(model_name: str = "pix2pix-medium") -> Tuple[ModelConfig, TrainingConfig]:
    """Get development configuration for experimentation."""
    return create_experiment_config(model_name, "development")


def get_production_config(model_name: str = "pix2pix-large") -> Tuple[ModelConfig, TrainingConfig]:
    """Get production configuration for final training."""
    return create_experiment_config(model_name, "production")
