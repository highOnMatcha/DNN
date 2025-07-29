"""Logging utilities for the time series project."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 log_dir: str = "logs") -> None:
    """Setup logging configuration."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_path / log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Create default log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_log_file = log_path / f"time_series_{timestamp}.log"
    default_handler = logging.FileHandler(default_log_file)
    default_handler.setLevel(level)
    default_handler.setFormatter(formatter)
    root_logger.addHandler(default_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)


class TrainingProgressLogger:
    """Logger for tracking training progress with metrics."""
    
    def __init__(self, name: str = "training"):
        self.logger = get_logger(name)
        self.start_time = None
        self.epoch_start_time = None
        
    def start_training(self, total_epochs: int, model_name: str, symbol: str):
        """Log training start."""
        self.start_time = datetime.now()
        self.logger.info(f"Starting training for {model_name} on {symbol}")
        self.logger.info(f"Total epochs: {total_epochs}")
        
    def start_epoch(self, epoch: int):
        """Log epoch start."""
        self.epoch_start_time = datetime.now()
        
    def log_epoch_metrics(self, epoch: int, train_loss: float, val_loss: float, 
                         train_mae: float = None, val_mae: float = None,
                         learning_rate: float = None):
        """Log epoch metrics."""
        epoch_time = datetime.now() - self.epoch_start_time if self.epoch_start_time else None
        
        message = f"Epoch {epoch} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
        
        if train_mae is not None and val_mae is not None:
            message += f", Train MAE: {train_mae:.6f}, Val MAE: {val_mae:.6f}"
            
        if learning_rate is not None:
            message += f", LR: {learning_rate:.2e}"
            
        if epoch_time:
            message += f", Time: {epoch_time.total_seconds():.1f}s"
            
        self.logger.info(message)
        
    def log_best_model(self, epoch: int, metric_value: float, metric_name: str = "val_loss"):
        """Log when a new best model is found."""
        self.logger.info(f"New best model at epoch {epoch}: {metric_name} = {metric_value:.6f}")
        
    def log_early_stopping(self, epoch: int, patience: int):
        """Log early stopping."""
        self.logger.info(f"Early stopping at epoch {epoch} (patience: {patience})")
        
    def finish_training(self, total_epochs: int, best_epoch: int = None, best_metric: float = None):
        """Log training completion."""
        if self.start_time:
            total_time = datetime.now() - self.start_time
            self.logger.info(f"Training completed in {total_time}")
            
        self.logger.info(f"Total epochs completed: {total_epochs}")
        
        if best_epoch is not None and best_metric is not None:
            self.logger.info(f"Best model at epoch {best_epoch} with metric: {best_metric:.6f}")


def log_system_info():
    """Log system information."""
    logger = get_logger("system")
    
    import torch
    import platform
    import psutil
    
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Memory info
    memory = psutil.virtual_memory()
    logger.info(f"Total RAM: {memory.total / (1024**3):.1f} GB")
    logger.info(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    
    # CPU info
    logger.info(f"CPU cores: {psutil.cpu_count()}")


def log_data_info(data, symbol: str):
    """Log data information."""
    logger = get_logger("data")
    
    if hasattr(data, '__len__'):
        logger.info(f"Data for {symbol}: {len(data)} samples")
    
    if hasattr(data, 'columns'):
        logger.info(f"Features: {list(data.columns)}")
        
    if hasattr(data, 'dtypes'):
        logger.info(f"Data types: {dict(data.dtypes)}")
        
    if hasattr(data, 'isnull'):
        null_counts = data.isnull().sum()
        if null_counts.any():
            logger.warning(f"Null values found: {dict(null_counts[null_counts > 0])}")


def log_model_info(model, model_name: str):
    """Log model information."""
    logger = get_logger("model")
    
    logger.info(f"Model: {model_name}")
    
    if hasattr(model, 'parameters'):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    if hasattr(model, '__str__'):
        logger.debug(f"Model architecture:\n{model}")


# Initialize logging when module is imported
setup_logging()
