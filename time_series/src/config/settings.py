"""
Configuration settings for time series models and training.

This module provides configuration classes and utilities for managing
model architectures, training parameters, and data processing settings
for stock price prediction using LSTM neural networks.

Author: Time Series Team
Version: 1.0.0
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import torch

# Constants for configuration validation
MIN_EPOCHS = 1
MAX_EPOCHS = 1000
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 512
MIN_LEARNING_RATE = 1e-6
MAX_LEARNING_RATE = 1.0
VALID_ARCHITECTURES = ["lstm", "gru", "transformer"]
VALID_SCHEDULERS = ["step", "cosine", "exponential", "plateau", "cosine_warm_restart"]
VALID_NORMALIZERS = ["minmax", "standard", "robust"]

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """
    Configuration for neural network model architecture.
    
    This class defines the structure and parameters for LSTM-based models
    used in stock price prediction.
    
    Attributes:
        name (str): Unique identifier for the model configuration
        model_type (str): Type of neural network architecture (lstm, gru, transformer)
        input_size (int): Number of input features
        hidden_size (int): Size of hidden layers
        num_layers (int): Number of stacked layers
        dropout (float): Dropout rate for regularization (0.0-1.0)
        bidirectional (bool): Whether to use bidirectional architecture
        output_size (int): Number of output predictions
        use_attention (bool): Whether to include attention mechanism
    """
    name: str
    model_type: str
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    bidirectional: bool = False
    output_size: int = 1
    use_attention: bool = False
    
    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """
        Validate all configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        if self.model_type not in VALID_ARCHITECTURES:
            raise ValueError(f"Invalid model_type '{self.model_type}'. Must be one of {VALID_ARCHITECTURES}")
        
        if not isinstance(self.input_size, int) or self.input_size <= 0:
            raise ValueError(f"input_size must be a positive integer, got {self.input_size}")
        
        if not isinstance(self.hidden_size, int) or self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be a positive integer, got {self.hidden_size}")
        
        if not isinstance(self.num_layers, int) or self.num_layers <= 0:
            raise ValueError(f"num_layers must be a positive integer, got {self.num_layers}")
        
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(f"dropout must be between 0.0 and 1.0, got {self.dropout}")


@dataclass
class TrainingConfig:
    """
    Configuration for training parameters and optimization settings.
    
    This class defines all parameters needed for training LSTM models,
    including optimization, scheduling, and regularization settings.
    
    Attributes:
        batch_size (int): Number of samples per training batch
        learning_rate (float): Initial learning rate for optimizer
        num_epochs (int): Total number of training epochs
        weight_decay (float): L2 regularization coefficient
        scheduler_type (str): Type of learning rate scheduler
        scheduler_params (Dict[str, Any]): Parameters for the scheduler
        early_stopping_patience (int): Number of epochs to wait for improvement
        gradient_clip_norm (float): Maximum gradient norm for clipping
        validation_split (float): Fraction of data used for validation
        save_every_n_epochs (int): Frequency of checkpoint saving
    """
    batch_size: int
    learning_rate: float
    num_epochs: int
    weight_decay: float
    scheduler_type: str
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 1.0
    validation_split: float = 0.2
    save_every_n_epochs: int = 10
    
    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """
        Validate all training configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        if not MIN_BATCH_SIZE <= self.batch_size <= MAX_BATCH_SIZE:
            raise ValueError(f"batch_size must be between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE}")
        
        if not MIN_LEARNING_RATE <= self.learning_rate <= MAX_LEARNING_RATE:
            raise ValueError(f"learning_rate must be between {MIN_LEARNING_RATE} and {MAX_LEARNING_RATE}")
        
        if not MIN_EPOCHS <= self.num_epochs <= MAX_EPOCHS:
            raise ValueError(f"num_epochs must be between {MIN_EPOCHS} and {MAX_EPOCHS}")
        
        if self.scheduler_type not in VALID_SCHEDULERS:
            raise ValueError(f"scheduler_type must be one of {VALID_SCHEDULERS}")
        
        if not 0.0 <= self.validation_split <= 1.0:
            raise ValueError(f"validation_split must be between 0.0 and 1.0")


@dataclass
class DataConfig:
    """
    Configuration for data processing and feature engineering.
    
    This class defines parameters for data preprocessing, feature engineering,
    and dataset splitting for stock price prediction.
    
    Attributes:
        sequence_length (int): Length of input sequences for LSTM
        prediction_horizon (int): Number of days to predict ahead
        features (List[str]): List of feature columns to use
        target_column (str): Name of the target column to predict
        normalization_method (str): Method for data normalization
        train_split (float): Fraction of data for training
        val_split (float): Fraction of data for validation
        test_split (float): Fraction of data for testing
        include_technical_indicators (bool): Whether to add technical indicators
        technical_indicators (List[str]): List of technical indicators to compute
    """
    sequence_length: int
    prediction_horizon: int
    features: List[str]
    target_column: str
    normalization_method: str
    train_split: float
    val_split: float
    test_split: float
    include_technical_indicators: bool
    technical_indicators: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """
        Validate all data configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        
        if self.prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive")
        
        if self.normalization_method not in VALID_NORMALIZERS:
            raise ValueError(f"normalization_method must be one of {VALID_NORMALIZERS}")
        
        # Validate data splits sum to 1.0
        total_split = self.train_split + self.val_split + self.test_split
        if not 0.99 <= total_split <= 1.01:  # Allow small floating point errors
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
        
        if not all(0.0 <= split <= 1.0 for split in [self.train_split, self.val_split, self.test_split]):
            raise ValueError("All data splits must be between 0.0 and 1.0")


def load_config() -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Returns:
        Dict[str, Any]: Configuration dictionary loaded from config.json
        
    Raises:
        FileNotFoundError: If config.json file is not found
        json.JSONDecodeError: If config.json contains invalid JSON
    """
    config_path = Path(__file__).parent / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


# Global configuration instance - loaded once at module import
try:
    _CONFIG = load_config()
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    raise


def get_model_config(model_name: str) -> ModelConfig:
    """
    Get model configuration by name.
    
    Args:
        model_name (str): Name of the model configuration to retrieve
        
    Returns:
        ModelConfig: Model configuration object
        
    Raises:
        ValueError: If model_name is not found in configurations
    """
    if not isinstance(model_name, str):
        raise ValueError(f"model_name must be a string, got {type(model_name)}")
    
    if model_name not in _CONFIG["model_configs"]:
        available = list(_CONFIG["model_configs"].keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available}")
    
    config_dict = _CONFIG["model_configs"][model_name]
    logger.debug(f"Retrieved model config for '{model_name}'")
    
    return ModelConfig(name=model_name, **config_dict)


def get_training_config(config_name: str) -> TrainingConfig:
    """
    Get training configuration by name.
    
    Args:
        config_name (str): Name of the training configuration to retrieve
        
    Returns:
        TrainingConfig: Training configuration object
        
    Raises:
        ValueError: If config_name is not found in configurations
    """
    if not isinstance(config_name, str):
        raise ValueError(f"config_name must be a string, got {type(config_name)}")
    
    if config_name not in _CONFIG["training_configs"]:
        available = list(_CONFIG["training_configs"].keys())
        raise ValueError(f"Training config '{config_name}' not found. Available configs: {available}")
    
    logger.debug(f"Retrieved training config for '{config_name}'")
    return TrainingConfig(**_CONFIG["training_configs"][config_name])


def get_data_config(config_name: str) -> DataConfig:
    """
    Get data configuration by name.
    
    Args:
        config_name (str): Name of the data configuration to retrieve
        
    Returns:
        DataConfig: Data configuration object
        
    Raises:
        ValueError: If config_name is not found in configurations
    """
    if not isinstance(config_name, str):
        raise ValueError(f"config_name must be a string, got {type(config_name)}")
    
    if config_name not in _CONFIG["data_configs"]:
        available = list(_CONFIG["data_configs"].keys())
        raise ValueError(f"Data config '{config_name}' not found. Available configs: {available}")
    
    logger.debug(f"Retrieved data config for '{config_name}'")
    return DataConfig(**_CONFIG["data_configs"][config_name])


def list_available_models() -> List[str]:
    """
    List all available model configurations.
    
    Returns:
        List[str]: List of available model configuration names
    """
    return list(_CONFIG["model_configs"].keys())


def list_available_training_configs() -> List[str]:
    """
    List all available training configurations.
    
    Returns:
        List[str]: List of available training configuration names
    """
    return list(_CONFIG["training_configs"].keys())


def list_available_data_configs() -> List[str]:
    """
    List all available data configurations.
    
    Returns:
        List[str]: List of available data configuration names
    """
    return list(_CONFIG["data_configs"].keys())


def list_available_symbols() -> List[str]:
    """
    List all available stock symbols.
    
    Returns:
        List[str]: List of available stock symbols
    """
    return list(_CONFIG["stock_symbols"].keys())


def get_symbol_info(symbol: str) -> str:
    """
    Get company name for a stock symbol.
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL')
        
    Returns:
        str: Company name or error message if symbol not found
    """
    if not isinstance(symbol, str):
        raise ValueError(f"symbol must be a string, got {type(symbol)}")
    
    return _CONFIG["stock_symbols"].get(symbol, f"Unknown symbol: {symbol}")


def get_device() -> torch.device:
    """
    Get the best available device for training.
    
    Automatically detects and returns the most suitable device for PyTorch
    operations, prioritizing CUDA GPU, then MPS (Apple Silicon), then CPU.
    
    Returns:
        torch.device: The best available device for computation
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def get_model_save_path(model_name: str, symbol: str, config_name: str) -> str:
    """
    Generate a standardized model save path.
    
    Args:
        model_name (str): Name of the model architecture
        symbol (str): Stock symbol being predicted
        config_name (str): Training configuration name
        
    Returns:
        str: Standardized path for saving model files
        
    Raises:
        ValueError: If any argument is not a string
    """
    if not all(isinstance(arg, str) for arg in [model_name, symbol, config_name]):
        raise ValueError("All arguments must be strings")
    
    # Sanitize inputs to create valid file paths
    clean_model = model_name.replace(" ", "_").lower()
    clean_symbol = symbol.upper()
    clean_config = config_name.replace(" ", "_").lower()
    
    return f"models/{clean_model}_{clean_symbol}_{clean_config}"


def get_experiment_name(model_name: str, symbol: str, config_name: str) -> str:
    """
    Generate a standardized experiment name for WandB tracking.
    
    Args:
        model_name (str): Name of the model architecture
        symbol (str): Stock symbol being predicted  
        config_name (str): Training configuration name
        
    Returns:
        str: Standardized experiment name for experiment tracking
        
    Raises:
        ValueError: If any argument is not a string
    """
    if not all(isinstance(arg, str) for arg in [model_name, symbol, config_name]):
        raise ValueError("All arguments must be strings")
    
    # Create clean experiment name
    clean_model = model_name.replace("_", "-").lower()
    clean_symbol = symbol.upper()
    clean_config = config_name.replace("_", "-").lower()
    
    return f"{clean_model}-{clean_symbol}-{clean_config}"


def validate_config_integrity() -> bool:
    """
    Validate the integrity of the loaded configuration.
    
    Checks that all required sections exist and contain valid data.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    required_sections = ["model_configs", "training_configs", "data_configs", "stock_symbols"]
    
    try:
        for section in required_sections:
            if section not in _CONFIG:
                logger.error(f"Missing required configuration section: {section}")
                return False
            
            if not isinstance(_CONFIG[section], dict) or not _CONFIG[section]:
                logger.error(f"Configuration section '{section}' is empty or invalid")
                return False
        
        logger.info("Configuration integrity validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False
