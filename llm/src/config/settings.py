"""
Configuration settings for training and model parameters.

This module provides comprehensive configuration management for machine learning
training pipelines, including model architecture parameters, training hyperparameters,
and dataset configuration. It supports both JSON-based external configuration
and programmatic configuration through dataclasses.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any


# Path to external model configurations
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "model_configs.json")


def load_model_configs() -> Dict[str, Any]:
    """
    Load model configurations from JSON file.
    
    Attempts to load configuration data from the JSON file. If the file
    doesn't exist, returns a default empty configuration structure.
    
    Returns:
        Dictionary containing model configurations with keys for
        pretrained_models, custom_models, and training_configs.
    """
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {CONFIG_FILE} not found. Using default configurations.")
        return {
            "pretrained_models": {},
            "custom_models": {},
            "training_configs": {}
        }


# Load configurations from JSON
_MODEL_CONFIGS = load_model_configs()


@dataclass
class ModelConfig:
    """
    Model configuration parameters for both pre-trained and custom models.
    
    This class defines all parameters needed to configure a model, including
    both pre-trained models for fine-tuning and custom models built from scratch.
    
    Attributes:
        name: Model name or path (e.g., 'gpt2', 'microsoft/DialoGPT-medium').
        output_dir: Directory where trained model and checkpoints will be saved.
        max_sequence_length: Maximum sequence length for input processing.
        from_scratch: Whether to build model from scratch or use pre-trained.
        vocab_size: Vocabulary size for custom models.
        n_embd: Embedding dimension for transformer layers.
        n_layer: Number of transformer layers.
        n_head: Number of attention heads per layer.
        dropout: Dropout probability for regularization.
    """
    name: str = "gpt2"  # Default to base GPT-2
    output_dir: str = "./trained_model"
    max_sequence_length: int = 512
    from_scratch: bool = False  # Whether to build from scratch or use pre-trained
    
    # Architecture parameters for custom models
    vocab_size: int = 50257  # GPT-2 vocab size
    n_embd: int = 768        # Embedding dimension
    n_layer: int = 12        # Number of transformer layers
    n_head: int = 12         # Number of attention heads
    dropout: float = 0.1     # Dropout rate


# Predefined model configurations loaded from JSON
def get_all_model_options() -> Dict[str, Any]:
    """
    Get all available model options from JSON configuration.
    
    Combines both pre-trained and custom model configurations into a single
    dictionary for easy access and validation.
    
    Returns:
        Dictionary containing all available model configurations.
    """
    all_models = {}
    all_models.update(_MODEL_CONFIGS.get("pretrained_models", {}))
    all_models.update(_MODEL_CONFIGS.get("custom_models", {}))
    return all_models


MODEL_OPTIONS = get_all_model_options()


@dataclass
class TrainingConfig:
    """
    Training configuration parameters for model training.
    
    This class defines all hyperparameters and settings needed for the training
    process, including optimization parameters, logging frequency, and data
    processing settings.
    
    Common max_samples values:
    - test: 50 (quick testing/debugging)
    - development: 1000 (medium experiments)  
    - production: None (full dataset)
    
    Attributes:
        num_epochs: Number of training epochs to run.
        batch_size: Batch size for training and evaluation.
        learning_rate: Learning rate for the optimizer.
        warmup_steps: Number of warmup steps for learning rate scheduling.
        warmup_ratio: Alternative to warmup_steps - fraction of training for warmup.
        lr_scheduler_type: Type of learning rate scheduler to use.
        logging_steps: Frequency of logging training metrics.
        save_steps: Frequency of saving model checkpoints.
        eval_steps: Frequency of running evaluation.
        weight_decay: Weight decay coefficient for regularization.
        max_samples: Maximum number of samples to use for training. 
                    None = use full dataset (production)
                    Small number (50-1000) = quick testing/debugging
                    Medium number (1000-10000) = development experiments
        train_split: Fraction of data to use for training vs evaluation.
        patience: Number of evaluation steps to wait for improvement before stopping.
        early_stopping_threshold: Minimum improvement threshold for early stopping.
        metric_for_best_model: Metric to use for early stopping and best model selection.
    """
    num_epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    warmup_ratio: Optional[float] = None  # If set, overrides warmup_steps
    lr_scheduler_type: str = "cosine"  # linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
    weight_decay: float = 0.01
    max_samples: Optional[int] = None  # None = use full dataset, set to limit samples for testing
    train_split: float = 0.9
    patience: Optional[int] = None  # Number of eval steps to wait for improvement
    early_stopping_threshold: float = 0.0  # Minimum improvement needed
    metric_for_best_model: str = "eval_loss"  # Metric to track for early stopping
    # Streaming cache configuration
    cache_memory_percent: float = 0.1  # Percentage of RAM to use for database caching


@dataclass
class DatasetConfig:
    """
    Dataset configuration parameters for data loading and processing.
    
    This class defines settings for dataset management, including file paths,
    download behavior, and caching options.
    
    Attributes:
        data_dir: Directory containing dataset files.
        save_to_disk: Whether to save downloaded datasets to disk.
        force_download: Whether to force re-download of existing datasets.
    """
    data_dir: str = "data"  # Keep pointing to the existing data directory
    save_to_disk: bool = True
    force_download: bool = False


# Default configurations
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_DATASET_CONFIG = DatasetConfig()


def get_production_config() -> TrainingConfig:
    """
    Get configuration for production training.
    
    Loads production training parameters from JSON configuration or uses
    sensible defaults for full-scale training with larger datasets. Uses
    the full dataset (max_samples=None) by default.
    
    Returns:
        TrainingConfig instance configured for production training.
    """
    prod_config = _MODEL_CONFIGS.get("training_configs", {}).get("production", {})
    return TrainingConfig(
        num_epochs=prod_config.get("num_epochs", 3),
        batch_size=prod_config.get("batch_size", 8),
        learning_rate=prod_config.get("learning_rate", 2e-5),
        warmup_steps=prod_config.get("warmup_steps", 100),
        warmup_ratio=prod_config.get("warmup_ratio", None),
        lr_scheduler_type=prod_config.get("lr_scheduler_type", "cosine"),
        max_samples=prod_config.get("max_samples", None),
        save_steps=1000,
        eval_steps=1000,
        patience=prod_config.get("patience", 5),
        early_stopping_threshold=prod_config.get("early_stopping_threshold", 0.001),
        weight_decay=prod_config.get("weight_decay", 0.01),
        metric_for_best_model="eval_loss",
        cache_memory_percent=prod_config.get("cache_memory_percent", 0.15)
    )


def get_test_config() -> TrainingConfig:
    """
    Get configuration for testing and development.
    
    Loads test configuration parameters optimized for quick iteration
    and debugging with smaller datasets and fewer epochs. Uses 50 samples
    by default for fast testing cycles.
    
    Returns:
        TrainingConfig instance configured for testing.
    """
    test_config = _MODEL_CONFIGS.get("training_configs", {}).get("test", {})
    return TrainingConfig(
        num_epochs=test_config.get("num_epochs", 1),
        batch_size=test_config.get("batch_size", 2),
        learning_rate=test_config.get("learning_rate", 5e-5),
        warmup_steps=test_config.get("warmup_steps", 20),
        warmup_ratio=test_config.get("warmup_ratio", None),
        lr_scheduler_type=test_config.get("lr_scheduler_type", "constant_with_warmup"),
        max_samples=test_config.get("max_samples", 50),  # Small sample for quick testing
        save_steps=50,
        eval_steps=50,
        patience=test_config.get("patience", None),
        early_stopping_threshold=test_config.get("early_stopping_threshold", 0.0),
        weight_decay=test_config.get("weight_decay", 0.01),
        metric_for_best_model="eval_loss",
        cache_memory_percent=test_config.get("cache_memory_percent", 0.05)
    )


def get_development_config() -> TrainingConfig:
    """
    Get configuration for development training.
    
    Loads development configuration parameters that balance training time
    with model quality for intermediate-scale experiments. Uses 1000 samples
    by default for meaningful development work.
    
    Returns:
        TrainingConfig instance configured for development.
    """
    dev_config = _MODEL_CONFIGS.get("training_configs", {}).get("development", {})
    return TrainingConfig(
        num_epochs=dev_config.get("num_epochs", 2),
        batch_size=dev_config.get("batch_size", 4),
        learning_rate=dev_config.get("learning_rate", 3e-5),
        warmup_steps=dev_config.get("warmup_steps", 100),
        warmup_ratio=dev_config.get("warmup_ratio", None),
        lr_scheduler_type=dev_config.get("lr_scheduler_type", "cosine"),
        max_samples=dev_config.get("max_samples", 1000),
        save_steps=200,
        eval_steps=200,
        patience=dev_config.get("patience", 3),
        early_stopping_threshold=dev_config.get("early_stopping_threshold", 0.005),
        weight_decay=dev_config.get("weight_decay", 0.01),
        metric_for_best_model="eval_loss",
        cache_memory_percent=dev_config.get("cache_memory_percent", 0.1)
    )


def get_model_config(model_type: str = "gpt2-small") -> ModelConfig:
    """
    Get model configuration for a specific model type.
    
    Retrieves configuration parameters for the specified model type from
    the loaded JSON configuration. Handles both pre-trained and custom
    model configurations.
    
    Args:
        model_type: Identifier for the model configuration to retrieve.
    
    Returns:
        ModelConfig instance with the specified model parameters.
    
    Raises:
        ValueError: If the specified model type is not available.
    """
    if model_type not in MODEL_OPTIONS:
        available = ", ".join(MODEL_OPTIONS.keys())
        raise ValueError(f"Model type '{model_type}' not available. Choose from: {available}")
    
    model_info = MODEL_OPTIONS[model_type]
    config = ModelConfig(
        name=model_info["name"],
        output_dir=model_info["output_dir"],
        from_scratch=model_info.get("from_scratch", False)
    )
    
    # Set custom architecture parameters if building from scratch
    if config.from_scratch and "architecture" in model_info:
        arch = model_info["architecture"]
        config.n_embd = arch.get("n_embd", 768)
        config.n_layer = arch.get("n_layer", 12)
        config.n_head = arch.get("n_head", 12)
        config.vocab_size = arch.get("vocab_size", 50257)
        config.max_sequence_length = arch.get("max_sequence_length", 512)
        config.dropout = arch.get("dropout", 0.1)
    elif config.from_scratch:
        # Fallback to old format for backward compatibility
        config.n_embd = model_info.get("n_embd", 768)
        config.n_layer = model_info.get("n_layer", 12)
        config.n_head = model_info.get("n_head", 12)
    
    return config


def list_available_models() -> None:
    """
    Print available model configurations in a formatted display.
    
    Displays all available model configurations grouped by type (pre-trained
    vs custom) along with their key parameters and descriptions.
    """
    print("Available Model Configurations:")
    print("=" * 40)
    
    # Group by type from JSON structure
    pretrained = _MODEL_CONFIGS.get("pretrained_models", {})
    custom = _MODEL_CONFIGS.get("custom_models", {})
    
    if pretrained:
        print("\nPRE-TRAINED MODELS (Fine-tuning):")
        print("-" * 35)
        for key, info in pretrained.items():
            print(f"• {key}:")
            print(f"  Model: {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  Output: {info['output_dir']}")
            print()
    
    if custom:
        print("CUSTOM MODELS (From Scratch):")
        print("-" * 30)
        for key, info in custom.items():
            print(f"• {key}:")
            print(f"  Description: {info['description']}")
            arch = info.get("architecture", {})
            if arch:
                print(f"  Architecture: {arch.get('n_layer', 12)} layers, {arch.get('n_embd', 768)} dim")
            print(f"  Output: {info['output_dir']}")
            print()
    
    # Also show training configurations
    training_configs = _MODEL_CONFIGS.get("training_configs", {})
    if training_configs:
        print("TRAINING CONFIGURATIONS:")
        print("-" * 25)
        for key, config in training_configs.items():
            print(f"• {key}:")
            print(f"  Description: {config.get('description', 'No description')}")
            print(f"  Epochs: {config.get('num_epochs', 'N/A')}")
            print(f"  Batch size: {config.get('batch_size', 'N/A')}")
            print(f"  Max samples: {config.get('max_samples', 'All')}")
            print()


def add_custom_model_to_config(
    model_key: str,
    name: str,
    description: str,
    output_dir: str,
    n_embd: int = 768,
    n_layer: int = 12,
    n_head: int = 12,
    vocab_size: int = 50257,
    max_sequence_length: int = 512,
    dropout: float = 0.1
) -> None:
    """
    Add a new custom model configuration to the JSON file.
    
    Creates a new custom model configuration with the specified parameters
    and saves it to the configuration file. This allows dynamic addition
    of new model architectures without manual JSON editing.
    
    Args:
        model_key: Unique identifier for the model configuration.
        name: Human-readable name for the model.
        description: Description of the model's purpose or characteristics.
        output_dir: Directory where the model will be saved.
        n_embd: Embedding dimension for the transformer.
        n_layer: Number of transformer layers.
        n_head: Number of attention heads per layer.
        vocab_size: Size of the vocabulary.
        max_sequence_length: Maximum sequence length.
        dropout: Dropout probability for regularization.
    """
    new_model = {
        "name": name,
        "description": description,
        "output_dir": output_dir,
        "from_scratch": True,
        "architecture": {
            "n_embd": n_embd,
            "n_layer": n_layer,
            "n_head": n_head,
            "vocab_size": vocab_size,
            "max_sequence_length": max_sequence_length,
            "dropout": dropout
        }
    }
    
    # Load current config
    configs = load_model_configs()
    
    # Add new model
    if "custom_models" not in configs:
        configs["custom_models"] = {}
    configs["custom_models"][model_key] = new_model
    
    # Save back to file
    with open(CONFIG_FILE, 'w') as f:
        json.dump(configs, f, indent=2)
    
    print(f"Added custom model '{model_key}' to configuration file.")
    
    # Reload configurations
    global _MODEL_CONFIGS, MODEL_OPTIONS
    _MODEL_CONFIGS = load_model_configs()
    MODEL_OPTIONS = get_all_model_options()
