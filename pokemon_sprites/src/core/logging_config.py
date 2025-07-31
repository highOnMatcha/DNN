"""
Centralized logging configuration for the Pokemon sprite generation training pipeline.

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
    
    def __init__(self, model_name: str = "unknown", experiment_id: str = "unknown"):
        super().__init__()
        self.model_name = model_name
        self.experiment_id = experiment_id
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add training context to log record."""
        record.model_name = self.model_name
        record.experiment_id = self.experiment_id
        return True


class TrainingProgressLogger:
    """
    Specialized logger for tracking training progress with metrics.
    
    Provides structured logging for training metrics, loss values,
    and performance indicators with support for different log levels
    and output formats.
    """
    
    def __init__(self, logger_name: str = "training.progress"):
        self.logger = logging.getLogger(logger_name)
        self.start_time = None
        self.epoch_start_time = None
        
    def start_training(self, total_epochs: int, total_batches: int):
        """Log training start."""
        self.start_time = datetime.now()
        self.logger.info("Training started", extra={
            'extra_fields': {
                'event': 'training_start',
                'total_epochs': total_epochs,
                'total_batches': total_batches,
                'start_time': self.start_time.isoformat()
            }
        })
    
    def start_epoch(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        self.epoch_start_time = datetime.now()
        self.logger.info(f"Starting epoch {epoch + 1}/{total_epochs}", extra={
            'extra_fields': {
                'event': 'epoch_start',
                'epoch': epoch + 1,
                'total_epochs': total_epochs,
                'epoch_start_time': self.epoch_start_time.isoformat()
            }
        })
    
    def log_batch(self, epoch: int, batch: int, total_batches: int, 
                  losses: Dict[str, float], metrics: Optional[Dict[str, float]] = None):
        """Log batch training progress."""
        elapsed = datetime.now() - self.epoch_start_time if self.epoch_start_time else None
        
        log_data = {
            'event': 'batch_complete',
            'epoch': epoch + 1,
            'batch': batch + 1,
            'total_batches': total_batches,
            'progress_pct': ((batch + 1) / total_batches) * 100,
            'losses': losses
        }
        
        if metrics:
            log_data['metrics'] = metrics
            
        if elapsed:
            log_data['elapsed_seconds'] = elapsed.total_seconds()
            log_data['batches_per_second'] = (batch + 1) / elapsed.total_seconds()
        
        # Create summary message
        loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
        message = f"Epoch {epoch + 1}/{total_batches} [{batch + 1}/{total_batches}] - {loss_str}"
        
        self.logger.info(message, extra={'extra_fields': log_data})
    
    def end_epoch(self, epoch: int, total_epochs: int, avg_losses: Dict[str, float], 
                  val_metrics: Optional[Dict[str, float]] = None):
        """Log epoch completion."""
        epoch_time = datetime.now() - self.epoch_start_time if self.epoch_start_time else None
        total_time = datetime.now() - self.start_time if self.start_time else None
        
        log_data = {
            'event': 'epoch_complete',
            'epoch': epoch + 1,
            'total_epochs': total_epochs,
            'avg_losses': avg_losses
        }
        
        if val_metrics:
            log_data['validation_metrics'] = val_metrics
            
        if epoch_time:
            log_data['epoch_duration_seconds'] = epoch_time.total_seconds()
            
        if total_time:
            log_data['total_training_time_seconds'] = total_time.total_seconds()
            estimated_total = (total_time.total_seconds() / (epoch + 1)) * total_epochs
            log_data['estimated_total_time_seconds'] = estimated_total
        
        # Create summary message
        loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()])
        message = f"Epoch {epoch + 1}/{total_epochs} completed - Avg {loss_str}"
        
        if val_metrics:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            message += f" | Val: {metrics_str}"
        
        self.logger.info(message, extra={'extra_fields': log_data})
    
    def end_training(self, final_metrics: Optional[Dict[str, float]] = None):
        """Log training completion."""
        total_time = datetime.now() - self.start_time if self.start_time else None
        
        log_data = {
            'event': 'training_complete',
            'end_time': datetime.now().isoformat()
        }
        
        if total_time:
            log_data['total_training_time_seconds'] = total_time.total_seconds()
            log_data['total_training_time_human'] = str(total_time)
            
        if final_metrics:
            log_data['final_metrics'] = final_metrics
        
        message = "Training completed"
        if total_time:
            message += f" in {total_time}"
            
        self.logger.info(message, extra={'extra_fields': log_data})


def get_log_directory() -> Path:
    """
    Get the absolute path to the logs directory.
    
    Creates the directory if it doesn't exist.
    
    Returns:
        Path object pointing to the logs directory.
    """
    # Get project root directory
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir


def initialize_project_logging(
    project_name: str = "pokemon_sprites",
    log_level: str = "INFO",
    model_name: str = "unknown",
    experiment_id: Optional[str] = None,
    enable_file_logging: bool = True,
    enable_json_logging: bool = True
) -> None:
    """
    Initialize comprehensive logging for the Pokemon sprite generation project.
    
    Sets up multiple log handlers including console output, file logging,
    and structured JSON logging for different components of the system.
    
    Args:
        project_name: Name of the project for log file naming.
        log_level: Minimum log level ("DEBUG", "INFO", "WARNING", "ERROR").
        model_name: Name of the model being trained.
        experiment_id: Unique identifier for this training run.
        enable_file_logging: Whether to enable file-based logging.
        enable_json_logging: Whether to enable JSON-formatted logging.
    """
    if experiment_id is None:
        experiment_id = f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set root logger level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    json_formatter = JsonFormatter()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    if enable_file_logging:
        log_dir = get_log_directory()
        
        # Main log file
        main_log_file = log_dir / f"{experiment_id}.log"
        file_handler = logging.FileHandler(main_log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Error log file
        error_log_file = log_dir / f"{experiment_id}_errors.log"
        error_handler = logging.FileHandler(error_log_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
        
        if enable_json_logging:
            # JSON log file for structured data
            json_log_file = log_dir / f"{experiment_id}_structured.jsonl"
            json_handler = logging.FileHandler(json_log_file)
            json_handler.setLevel(logging.INFO)
            json_handler.setFormatter(json_formatter)
            root_logger.addHandler(json_handler)
    
    # Add training context filter to all handlers
    training_filter = ModelTrainingFilter(model_name, experiment_id)
    for handler in root_logger.handlers:
        handler.addFilter(training_filter)
    
    # Log initialization
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized for {project_name}")
    logger.info(f"Experiment ID: {experiment_id}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Log level: {log_level}")
    
    if enable_file_logging:
        logger.info(f"Log directory: {log_dir}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Name for the logger (usually __name__).
        
    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)


def log_system_info():
    """Log system information for debugging and reproducibility."""
    import platform
    import torch
    import sys
    
    logger = get_logger(__name__)
    
    logger.info("System Information:")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Architecture: {platform.architecture()}")
    logger.info(f"Processor: {platform.processor()}")
    
    # PyTorch information
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {gpu_props.name} ({gpu_props.total_memory // 1024**3} GB)")
    
    # Memory information
    import psutil
    memory = psutil.virtual_memory()
    logger.info(f"Total memory: {memory.total // 1024**3} GB")
    logger.info(f"Available memory: {memory.available // 1024**3} GB")


def log_model_summary(model, input_shape: tuple, logger_name: str = "model.summary"):
    """
    Log model architecture summary.
    
    Args:
        model: PyTorch model instance.
        input_shape: Shape of input tensor (excluding batch dimension).
        logger_name: Name for the logger.
    """
    logger = get_logger(logger_name)
    
    try:
        from torchsummary import summary
        
        # Capture summary output
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            summary(model, input_shape)
        summary_str = f.getvalue()
        
        logger.info("Model Architecture Summary:")
        for line in summary_str.split('\n'):
            if line.strip():
                logger.info(line)
                
    except ImportError:
        logger.warning("torchsummary not available. Install with: pip install torchsummary")
        
        # Fallback: basic parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")


# Module-level logger for this file
logger = get_logger(__name__)
