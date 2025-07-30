"""
Segmentation model trainer with comprehensive logging and monitoring.

This module provides a training framework for semantic segmentation models with
WandB integration for experiment tracking, comprehensive metrics calculation,
and model checkpointing.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from tqdm import tqdm

from .logging_config import get_logger, log_model_info, TrainingProgressLogger
from models.unet import create_segmentation_model, count_parameters
from utils.metrics import (
    SegmentationMetrics, 
    dice_loss, 
    focal_loss, 
    combined_loss,
    visualize_predictions
)

# Initialize module logger
logger = get_logger(__name__)


class WandBCallback:
    """
    WandB logging callback for segmentation training.
    
    This callback integrates with Weights & Biases to track training progress,
    model parameters, and system resources during training.
    """
    
    def __init__(self, wandb_run: Optional[Any] = None):
        """
        Initialize the WandB callback.
        
        Args:
            wandb_run: Active WandB run instance for logging metrics
        """
        self.wandb_run = wandb_run
        self.start_time: Optional[float] = None
        self.logged_model_info: bool = False
    
    def on_train_begin(self, model: nn.Module, config: Any) -> None:
        """
        Called at the beginning of training.
        
        Args:
            model: Model instance
            config: Configuration object
        """
        if self.wandb_run and not self.logged_model_info:
            self.start_time = time.time()
            self.wandb_run.watch(model, log="all", log_freq=10, log_graph=False)
            self.logged_model_info = True
    
    def on_epoch_end(self, epoch: int, train_metrics: Dict, val_metrics: Dict) -> None:
        """
        Called at the end of each epoch.
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
        """
        if self.wandb_run:
            log_dict = {
                "epoch": epoch,
                "train/loss": train_metrics.get("loss", 0),
                "train/pixel_accuracy": train_metrics.get("pixel_accuracy", 0),
                "train/mean_iou": train_metrics.get("mean_iou", 0),
                "train/mean_dice": train_metrics.get("mean_dice", 0),
                "val/loss": val_metrics.get("loss", 0),
                "val/pixel_accuracy": val_metrics.get("pixel_accuracy", 0),
                "val/mean_iou": val_metrics.get("mean_iou", 0),
                "val/mean_dice": val_metrics.get("mean_dice", 0),
            }
            
            # Add GPU memory usage if available
            if torch.cuda.is_available():
                log_dict["system/gpu_memory_gb"] = torch.cuda.memory_allocated() / 1e9
            
            # Add elapsed time
            if self.start_time:
                log_dict["system/elapsed_minutes"] = (time.time() - self.start_time) / 60
            
            self.wandb_run.log(log_dict)
    
    def on_batch_end(self, step: int, loss: float, lr: float) -> None:
        """
        Called at the end of each batch (for detailed logging).
        
        Args:
            step: Current step number
            loss: Batch loss
            lr: Current learning rate
        """
        if self.wandb_run and step % 100 == 0:  # Log every 100 steps
            self.wandb_run.log({
                "step": step,
                "batch_loss": loss,
                "learning_rate": lr
            })


class SegmentationTrainer:
    """
    Main trainer class for semantic segmentation.
    
    This class handles the complete training pipeline including model initialization,
    training loop, validation, metrics calculation, and model checkpointing.
    """
    
    def __init__(self, config, wandb_run: Optional[Any] = None):
        """
        Initialize the segmentation trainer.
        
        Args:
            config: Configuration object containing training parameters
            wandb_run: Optional WandB run instance for experiment tracking
        """
        self.config = config
        self.wandb_run = wandb_run
        self.device = self._get_device()
        
        # Initialize model
        self.model = create_segmentation_model(config)
        self.model.to(self.device)
        
        # Setup mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Initialize metrics
        self.train_metrics = SegmentationMetrics(config.classes)
        self.val_metrics = SegmentationMetrics(config.classes)
        
        # Setup callbacks
        self.wandb_callback = WandBCallback(wandb_run)
        self.progress_logger = TrainingProgressLogger(logger)
        
        # Initialize tracking variables
        self.best_val_metric = 0.0
        self.current_epoch = 0
        self.global_step = 0
        
        # Log model information
        total_params, trainable_params = count_parameters(self.model)
        logger.info(f"Model created: {config.name}")
        logger.info(f"Architecture: {config.architecture}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Device: {self.device}")
        
        if self.wandb_run:
            # Log configuration to WandB
            config_dict = {
                "model_name": config.name,
                "architecture": config.architecture,
                "encoder_name": config.encoder_name,
                "classes": config.classes,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "num_epochs": config.num_epochs,
                "image_size": config.image_size,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "device": str(self.device)
            }
            self.wandb_run.config.update(config_dict)
    
    def _get_device(self) -> torch.device:
        """Get the appropriate device for training."""
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(self.config.device)
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on configuration."""
        if self.config.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def _setup_scheduler(self, optimizer: optim.Optimizer) -> Optional[Any]:
        """Setup learning rate scheduler based on configuration."""
        if self.config.scheduler.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.num_epochs,
                eta_min=1e-6
            )
        elif self.config.scheduler.lower() == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.scheduler.lower() == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=0.95
            )
        else:
            return None
    
    def _setup_loss_function(self) -> nn.Module:
        """Setup loss function based on configuration."""
        # You can extend this to support different loss functions
        return combined_loss
    
    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss for a batch."""
        return combined_loss(predictions, targets)
    
    def train_epoch(
        self, 
        train_loader: DataLoader, 
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer instance
            scheduler: Optional learning rate scheduler
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            targets = batch['mask'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if self.scaler:
                with torch.cuda.amp.autocast():
                    predictions = self.model(images)
                    loss = self._compute_loss(predictions, targets)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                predictions = self.model(images)
                loss = self._compute_loss(predictions, targets)
                loss.backward()
                optimizer.step()
            
            # Update metrics
            self.train_metrics.update(predictions.detach(), targets)
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log to WandB if enabled
            if self.wandb_callback and self.global_step % self.config.log_frequency == 0:
                self.wandb_callback.on_batch_end(
                    self.global_step, 
                    loss.item(), 
                    optimizer.param_groups[0]['lr']
                )
            
            self.global_step += 1
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Compute epoch metrics
        metrics = self.train_metrics.compute_all_metrics()
        metrics['loss'] = total_loss / num_batches
        
        return metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                targets = batch['mask'].to(self.device)
                
                predictions = self.model(images)
                loss = self._compute_loss(predictions, targets)
                
                # Update metrics
                self.val_metrics.update(predictions, targets)
                total_loss += loss.item()
        
        # Compute epoch metrics
        metrics = self.val_metrics.compute_all_metrics()
        metrics['loss'] = total_loss / num_batches
        
        return metrics
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False) -> str:
        """
        Save model checkpoint.
        
        Args:
            metrics: Current metrics dictionary
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'global_step': self.global_step
        }
        
        # Create checkpoint filename
        if is_best:
            filename = "best_model.pt"
        else:
            filename = f"checkpoint-{self.current_epoch}.pt"
        
        filepath = os.path.join(self.config.output_dir, filename)
        torch.save(checkpoint, filepath)
        
        logger.info(f"Checkpoint saved: {filepath}")
        if is_best:
            logger.info(f"New best model saved with validation metric: {metrics.get('mean_iou', 0):.4f}")
        
        return filepath
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader
    ) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Dictionary with training history
        """
        # Setup optimizer and scheduler
        optimizer = self._setup_optimizer()
        scheduler = self._setup_scheduler(optimizer)
        
        # Initialize training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_mean_iou': [],
            'val_mean_iou': [],
            'train_pixel_accuracy': [],
            'val_pixel_accuracy': []
        }
        
        # Log training start
        self.progress_logger.log_training_start(self.config.num_epochs)
        self.wandb_callback.on_train_begin(self.model, self.config)
        
        for epoch in range(1, self.config.num_epochs + 1):
            self.current_epoch = epoch
            
            # Log epoch start
            self.progress_logger.log_epoch_start(epoch, self.config.num_epochs)
            
            # Training phase
            train_metrics = self.train_epoch(train_loader, optimizer, scheduler)
            
            # Validation phase
            val_metrics = self.validate_epoch(val_loader)
            
            # Log epoch end
            self.progress_logger.log_epoch_end(epoch, {**train_metrics, **val_metrics})
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_mean_iou'].append(train_metrics['mean_iou'])
            history['val_mean_iou'].append(val_metrics['mean_iou'])
            history['train_pixel_accuracy'].append(train_metrics['pixel_accuracy'])
            history['val_pixel_accuracy'].append(val_metrics['pixel_accuracy'])
            
            # Log to WandB
            self.wandb_callback.on_epoch_end(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            is_best = val_metrics['mean_iou'] > self.best_val_metric
            if is_best:
                self.best_val_metric = val_metrics['mean_iou']
            
            if epoch % self.config.save_frequency == 0 or is_best:
                self.save_checkpoint(val_metrics, is_best)
        
        # Log training completion
        self.progress_logger.log_training_end()
        
        return history
    
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on a batch of images.
        
        Args:
            images: Input images tensor (B, C, H, W)
            
        Returns:
            Predicted segmentation masks (B, H, W)
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(images.to(self.device))
            return torch.argmax(predictions, dim=1).cpu()
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Resumed from epoch {self.current_epoch}")
