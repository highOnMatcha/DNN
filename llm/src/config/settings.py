"""
Configuration settings for training and model parameters.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional


# Path to external model configurations
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "model_configs.json")


def load_model_configs():
    """Load model configurations from JSON file."""
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
    """Model configuration parameters."""
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
def get_all_model_options():
    """Get all available model options from JSON config."""
    all_models = {}
    all_models.update(_MODEL_CONFIGS.get("pretrained_models", {}))
    all_models.update(_MODEL_CONFIGS.get("custom_models", {}))
    return all_models


MODEL_OPTIONS = get_all_model_options()


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    num_epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
    weight_decay: float = 0.01
    max_samples: Optional[int] = 1000  # For testing, use None for full dataset
    train_split: float = 0.9


@dataclass
class DatasetConfig:
    """Dataset configuration parameters."""
    data_dir: str = "data"  # Keep pointing to the existing data directory
    save_to_disk: bool = True
    force_download: bool = False


# Default configurations
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_DATASET_CONFIG = DatasetConfig()


def get_production_config():
    """Get configuration for production training."""
    prod_config = _MODEL_CONFIGS.get("training_configs", {}).get("production", {})
    return TrainingConfig(
        num_epochs=prod_config.get("num_epochs", 3),
        batch_size=prod_config.get("batch_size", 8),
        learning_rate=prod_config.get("learning_rate", 2e-5),
        max_samples=prod_config.get("max_samples", None),
        save_steps=1000,
        eval_steps=1000
    )


def get_test_config():
    """Get configuration for testing/development."""
    test_config = _MODEL_CONFIGS.get("training_configs", {}).get("test", {})
    return TrainingConfig(
        num_epochs=test_config.get("num_epochs", 1),
        batch_size=test_config.get("batch_size", 2),
        learning_rate=test_config.get("learning_rate", 5e-5),
        max_samples=test_config.get("max_samples", 100),
        save_steps=50,
        eval_steps=50
    )


def get_development_config():
    """Get configuration for development training."""
    dev_config = _MODEL_CONFIGS.get("training_configs", {}).get("development", {})
    return TrainingConfig(
        num_epochs=dev_config.get("num_epochs", 2),
        batch_size=dev_config.get("batch_size", 4),
        learning_rate=dev_config.get("learning_rate", 3e-5),
        max_samples=dev_config.get("max_samples", 1000),
        save_steps=200,
        eval_steps=200
    )


def get_model_config(model_type: str = "gpt2-small") -> ModelConfig:
    """Get model configuration for a specific model type."""
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


def list_available_models():
    """Print available model configurations."""
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
):
    """Add a new custom model configuration to the JSON file."""
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
