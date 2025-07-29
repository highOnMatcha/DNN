"""
Configuration settings for time series models and training.
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    name: str
    model_type: str
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    bidirectional: bool = False
    output_size: int = 1
    use_attention: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    batch_size: int
    learning_rate: float
    num_epochs: int
    weight_decay: float
    scheduler_type: str
    scheduler_params: Dict[str, Any]
    early_stopping_patience: int
    gradient_clip_norm: float
    validation_split: float
    save_every_n_epochs: int


@dataclass
class DataConfig:
    """Configuration for data processing."""
    sequence_length: int
    prediction_horizon: int
    features: List[str]
    target_column: str
    normalization_method: str
    train_split: float
    val_split: float
    test_split: float
    include_technical_indicators: bool
    technical_indicators: List[str]


def load_config() -> Dict[str, Any]:
    """Load configuration from JSON file."""
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, 'r') as f:
        return json.load(f)


_CONFIG = load_config()


def get_model_config(model_name: str) -> ModelConfig:
    """Get model configuration by name."""
    if model_name not in _CONFIG["model_configs"]:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(_CONFIG['model_configs'].keys())}")
    
    config_dict = _CONFIG["model_configs"][model_name]
    return ModelConfig(name=model_name, **config_dict)


def get_training_config(config_name: str) -> TrainingConfig:
    """Get training configuration by name."""
    if config_name not in _CONFIG["training_configs"]:
        raise ValueError(f"Training config '{config_name}' not found. Available: {list(_CONFIG['training_configs'].keys())}")
    
    return TrainingConfig(**_CONFIG["training_configs"][config_name])


def get_data_config(config_name: str) -> DataConfig:
    """Get data configuration by name."""
    if config_name not in _CONFIG["data_configs"]:
        raise ValueError(f"Data config '{config_name}' not found. Available: {list(_CONFIG['data_configs'].keys())}")
    
    return DataConfig(**_CONFIG["data_configs"][config_name])


def list_available_models() -> List[str]:
    """List all available model configurations."""
    return list(_CONFIG["model_configs"].keys())


def list_available_training_configs() -> List[str]:
    """List all available training configurations."""
    return list(_CONFIG["training_configs"].keys())


def list_available_data_configs() -> List[str]:
    """List all available data configurations."""
    return list(_CONFIG["data_configs"].keys())


def list_available_symbols() -> List[str]:
    """List all available stock symbols."""
    return list(_CONFIG["stock_symbols"].keys())


def get_symbol_info(symbol: str) -> str:
    """Get company name for a stock symbol."""
    return _CONFIG["stock_symbols"].get(symbol, f"Unknown symbol: {symbol}")


def get_device() -> torch.device:
    """Get the best available device for training."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_model_save_path(model_name: str, symbol: str, config_name: str) -> str:
    """Generate a standardized model save path."""
    return f"models/{model_name}_{symbol}_{config_name}"


def get_experiment_name(model_name: str, symbol: str, config_name: str) -> str:
    """Generate a standardized experiment name for WandB."""
    return f"{model_name}-{symbol}-{config_name}"
