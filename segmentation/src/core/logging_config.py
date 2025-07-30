"""
Centralized logging configuration for the segmentation training pipeline.

This module provides a comprehensive logging setup with multiple handlers,
formatters, and log levels for different components of the system.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


class JsonFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    
    Converts log records to JSON format for better parsing and analysis
    in log aggregation systems.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(getattr(record, 'extra_fields', {}))
            
        return json.dumps(log_entry)


class SegmentationTrainingFilter(logging.Filter):
    """Filter to add training context to log records."""
    
    def __init__(self, model_name: Optional[str] = None, config_type: Optional[str] = None):
        super().__init__()
        self.model_name = model_name
        self.config_type = config_type
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add training context to log record."""
        if not hasattr(record, 'extra_fields'):
            setattr(record, 'extra_fields', {})
        
        extra_fields = getattr(record, 'extra_fields', {})
        if self.model_name:
            extra_fields['model_name'] = self.model_name
        if self.config_type:
            extra_fields['config_type'] = self.config_type
            
        return True


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    config_type: Optional[str] = None,
    enable_json_logging: bool = False,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
    max_file_size_mb: int = 50,
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup comprehensive logging configuration for the segmentation pipeline.
    
    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files. If None, uses segmentation/logs/
        model_name: Name of the model being trained (for context)
        config_type: Type of training configuration (for context)
        enable_json_logging: Whether to use JSON formatting for file logs
        enable_file_logging: Whether to enable file logging
        enable_console_logging: Whether to enable console logging
        max_file_size_mb: Maximum size of log files before rotation
        backup_count: Number of backup log files to keep
    
    Returns:
        Configured logger instance
    """
    
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Get default log directory if not provided
    if log_dir is None:
        current_file = Path(__file__).resolve()
        segmentation_root = current_file.parent.parent.parent
        log_dir = segmentation_root / "logs"
        log_dir.mkdir(exist_ok=True)
    
    # Clear any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create main logger
    logger = logging.getLogger("segmentation")
    logger.setLevel(numeric_level)
    logger.handlers.clear()
    
    # Create formatter
    if enable_json_logging:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Add console handler
    if enable_console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        
        # Add training context filter
        if model_name or config_type:
            console_handler.addFilter(SegmentationTrainingFilter(model_name, config_type))
        
        logger.addHandler(console_handler)
    
    # Add file handler with rotation
    if enable_file_logging:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"segmentation_{timestamp}.log"
        log_filepath = Path(log_dir) / log_filename
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_filepath,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        
        # Add training context filter
        if model_name or config_type:
            file_handler.addFilter(SegmentationTrainingFilter(model_name, config_type))
        
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(f"segmentation.{name}")


def initialize_project_logging(
    model_name: str = "unknown",
    config_type: str = "default",
    log_level: str = "INFO",
    enable_json_logging: bool = False
) -> logging.Logger:
    """
    Initialize project-wide logging with model context.
    
    Args:
        model_name: Name of the model being trained
        config_type: Type of configuration being used
        log_level: Minimum log level
        enable_json_logging: Whether to use JSON formatting
    
    Returns:
        Main project logger
    """
    logger = setup_logging(
        log_level=log_level,
        model_name=model_name,
        config_type=config_type,
        enable_json_logging=enable_json_logging
    )
    
    logger.info(f"Initialized logging for segmentation project")
    logger.info(f"Model: {model_name}")
    logger.info(f"Config: {config_type}")
    logger.info(f"Log level: {log_level}")
    
    return logger


def log_system_info(logger: logging.Logger) -> None:
    """
    Log comprehensive system information.
    
    Args:
        logger: Logger instance to use for logging
    """
    import torch
    import platform
    
    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    logger.info("=== End System Information ===")


def log_model_info(logger: logging.Logger, model: Any, config: Any) -> None:
    """
    Log model architecture and configuration information.
    
    Args:
        logger: Logger instance to use for logging
        model: Model instance
        config: Configuration object
    """
    logger.info("=== Model Information ===")
    logger.info(f"Model name: {config.name}")
    logger.info(f"Architecture: {config.architecture}")
    logger.info(f"Encoder: {config.encoder_name}")
    logger.info(f"Classes: {config.classes}")
    logger.info(f"Input size: {config.image_size}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info("=== End Model Information ===")


class TrainingProgressLogger:
    """
    Logger for tracking training progress and metrics.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.epoch_start_time = None
        self.training_start_time = None
    
    def log_training_start(self, total_epochs: int) -> None:
        """Log training start."""
        import time
        self.training_start_time = time.time()
        self.logger.info(f"Starting training for {total_epochs} epochs")
    
    def log_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Log epoch start."""
        import time
        self.epoch_start_time = time.time()
        self.logger.info(f"Epoch {epoch}/{total_epochs} started")
    
    def log_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log epoch end with metrics."""
        import time
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        
        # Log metrics
        for metric_name, value in metrics.items():
            self.logger.info(f"{metric_name}: {value:.4f}")
    
    def log_training_end(self) -> None:
        """Log training completion."""
        import time
        if self.training_start_time:
            total_time = time.time() - self.training_start_time
            self.logger.info(f"Training completed in {total_time:.2f}s ({total_time/60:.2f}min)")
    
    def log_checkpoint_saved(self, epoch: int, path: str) -> None:
        """Log checkpoint saving."""
        self.logger.info(f"Checkpoint saved at epoch {epoch}: {path}")
    
    def log_best_model_saved(self, epoch: int, metric_value: float, path: str) -> None:
        """Log best model saving."""
        self.logger.info(f"New best model saved at epoch {epoch} (metric: {metric_value:.4f}): {path}")
