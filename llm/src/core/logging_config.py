"""
Centralized logging configuration for the LLM training pipeline.

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


class ModelTrainingFilter(logging.Filter):
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
    Setup comprehensive logging configuration for the LLM pipeline.
    
    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files. If None, uses llm/logs/
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
    
    # Create log directory
    if log_dir is None:
        # Default to llm/logs/ directory
        current_file = Path(__file__).resolve()
        llm_root = current_file.parent.parent.parent  # src/core/ -> src/ -> llm/
        log_dir_path = llm_root / "logs"
    else:
        log_dir_path = Path(log_dir)
    
    log_dir_path.mkdir(exist_ok=True, parents=True)
    
    # Clear any existing handlers to avoid duplicates
    logging.getLogger().handlers.clear()
    
    # Create root logger
    logger = logging.getLogger("llm")
    logger.setLevel(numeric_level)
    
    # Prevent duplicate logs from propagating to root logger
    logger.propagate = False
    
    # Create formatters
    console_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(module)s:%(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    json_formatter = JsonFormatter()
    
    # Setup console handler
    if enable_console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(console_formatter)
        
        # Add training context filter
        if model_name or config_type:
            training_filter = ModelTrainingFilter(model_name, config_type)
            console_handler.addFilter(training_filter)
        
        logger.addHandler(console_handler)
    
    # Setup file handlers
    if enable_file_logging:
        # General application log
        app_log_file = log_dir_path / "llm_training.log"
        app_handler = logging.handlers.RotatingFileHandler(
            app_log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        app_handler.setLevel(numeric_level)
        
        if enable_json_logging:
            app_handler.setFormatter(json_formatter)
        else:
            app_handler.setFormatter(detailed_formatter)
        
        # Add training context filter
        if model_name or config_type:
            training_filter = ModelTrainingFilter(model_name, config_type)
            app_handler.addFilter(training_filter)
        
        logger.addHandler(app_handler)
        
        # Error-only log file
        error_log_file = log_dir_path / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)
        
        # Training-specific log (if training context provided)
        if model_name and config_type:
            training_log_file = log_dir_path / f"training_{model_name}_{config_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            training_handler = logging.FileHandler(training_log_file)
            training_handler.setLevel(logging.DEBUG)
            training_handler.setFormatter(detailed_formatter)
            training_filter = ModelTrainingFilter(model_name, config_type)
            training_handler.addFilter(training_filter)
            logger.addHandler(training_handler)
    
    # Log the logging setup
    logger.info(f"Logging initialized with level: {log_level}")
    logger.info(f"Log directory: {log_dir_path}")
    if model_name:
        logger.info(f"Model context: {model_name}")
    if config_type:
        logger.info(f"Config context: {config_type}")
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (usually __name__ from calling module)
    
    Returns:
        Logger instance
    """
    if name is None:
        return logging.getLogger("llm")
    
    # Ensure it's under the llm namespace
    if not name.startswith("llm"):
        name = f"llm.{name}"
    
    return logging.getLogger(name)


def log_system_info(logger: logging.Logger) -> None:
    """Log system information for debugging purposes."""
    import torch
    import platform
    import psutil
    
    logger.info("=" * 60)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 60)
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        try:
            import torch.version
            logger.info(f"CUDA version: {torch.version.cuda}")
        except (AttributeError, ImportError):
            logger.info("CUDA version: Unknown")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # System memory
    memory = psutil.virtual_memory()
    logger.info(f"System RAM: {memory.total / 1e9:.1f} GB (Available: {memory.available / 1e9:.1f} GB)")
    
    # CPU info
    logger.info(f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    logger.info("=" * 60)


def log_training_config(logger: logging.Logger, config: Dict[str, Any]) -> None:
    """Log training configuration in a structured way."""
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)
    
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"  {sub_key}: {sub_value}")
        else:
            logger.info(f"{key}: {value}")
    
    logger.info("=" * 60)


def log_model_info(logger: logging.Logger, model, tokenizer=None) -> None:
    """Log model architecture and parameter information."""
    logger.info("=" * 60)
    logger.info("MODEL INFORMATION")
    logger.info("=" * 60)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    logger.info(f"Model size (MB): {total_params * 4 / 1e6:.1f}")  # Assuming float32
    
    if tokenizer:
        logger.info(f"Tokenizer vocab size: {len(tokenizer):,}")
        logger.info(f"Tokenizer type: {type(tokenizer).__name__}")
    
    logger.info("=" * 60)


class TrainingProgressLogger:
    """Helper class for logging training progress with consistent formatting."""
    
    def __init__(self, logger: logging.Logger, total_steps: int):
        self.logger = logger
        self.total_steps = total_steps
        self.start_time = None
    
    def start_training(self):
        """Log training start."""
        import time
        self.start_time = time.time()
        self.logger.info("🚀 Training started")
    
    def log_step(self, step: int, loss: float, learning_rate: float, **kwargs):
        """Log training step with progress."""
        import time
        
        progress = (step / self.total_steps) * 100
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        extra_info = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                for k, v in kwargs.items()])
        
        self.logger.info(
            f"Step {step:,}/{self.total_steps:,} ({progress:.1f}%) | "
            f"Loss: {loss:.4f} | LR: {learning_rate:.2e} | "
            f"Elapsed: {elapsed/60:.1f}min"
            + (f" | {extra_info}" if extra_info else "")
        )
    
    def log_epoch(self, epoch: int, train_loss: float, eval_loss: Optional[float] = None, **kwargs):
        """Log epoch completion."""
        message = f"📊 Epoch {epoch} completed | Train Loss: {train_loss:.4f}"
        
        if eval_loss is not None:
            message += f" | Eval Loss: {eval_loss:.4f}"
        
        for key, value in kwargs.items():
            if isinstance(value, float):
                message += f" | {key}: {value:.4f}"
            else:
                message += f" | {key}: {value}"
        
        self.logger.info(message)
    
    def finish_training(self, final_loss: float):
        """Log training completion."""
        import time
        
        total_time = time.time() - self.start_time if self.start_time else 0
        self.logger.info(f"✅ Training completed in {total_time/60:.1f} minutes | Final Loss: {final_loss:.4f}")


# Module-level function to setup logging for the entire project
def initialize_project_logging(
    log_level: str = "INFO",
    model_name: Optional[str] = None,
    config_type: Optional[str] = None
) -> logging.Logger:
    """
    Initialize logging for the entire LLM project.
    
    This function should be called once at the start of training or generation
    to set up consistent logging across all modules.
    """
    return setup_logging(
        log_level=log_level,
        model_name=model_name,
        config_type=config_type,
        enable_json_logging=False,  # Can be made configurable
        enable_file_logging=True,
        enable_console_logging=True
    )
