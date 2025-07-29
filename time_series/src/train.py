#!/usr/bin/env python3
"""LSTM training pipeline for stock price prediction."""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import wandb
from dotenv import load_dotenv
import click

from config.settings import (
    get_model_config, 
    get_training_config, 
    get_data_config,
    list_available_models,
    list_available_training_configs,
    list_available_data_configs,
    list_available_symbols,
    get_device,
    get_model_save_path,
    get_experiment_name
)
from models.lstm_models import create_model, count_parameters
from data.preprocessing import load_and_process_data
from utils.logging import TrainingProgressLogger, log_system_info, log_model_info, get_logger

load_dotenv()
logger = get_logger(__name__)


class StockTrainer:
    """Trainer class for stock price prediction models."""
    
    def __init__(self, 
                 model_config, 
                 training_config, 
                 data_config,
                 device: torch.device,
                 save_path: str,
                 use_wandb: bool = False):
        """
        Initialize trainer.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
            data_config: Data configuration
            device: Computing device
            save_path: Path to save model checkpoints
            use_wandb: Whether to use WandB for experiment tracking
        """
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        self.device = device
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        # Logging
        self.progress_logger = TrainingProgressLogger()
        
    def setup_model(self, input_size: int):
        """Setup model, optimizer, and scheduler."""
        # Update model config with actual input size
        self.model_config.input_size = input_size
        
        # Create model
        self.model = create_model(self.model_config)
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )
        
        # Setup scheduler
        if self.training_config.scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, **self.training_config.scheduler_params
            )
        elif self.training_config.scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, **self.training_config.scheduler_params
            )
        elif self.training_config.scheduler_type == "cosine_warm_restart":
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, **self.training_config.scheduler_params
            )
        else:
            self.scheduler = None
        
        # Log model info
        total_params, trainable_params = count_parameters(self.model)
        logger.info(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
        
        if self.use_wandb:
            wandb.watch(self.model, log="all", log_freq=100)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output.squeeze(), target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.training_config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.training_config.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            mae = torch.mean(torch.abs(output.squeeze() - target)).item()
            total_mae += mae
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return {'loss': avg_loss, 'mae': avg_mae}
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output.squeeze(), target)
                
                total_loss += loss.item()
                mae = torch.mean(torch.abs(output.squeeze() - target)).item()
                total_mae += mae
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return {'loss': avg_loss, 'mae': avg_mae}
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'model_config': self.model_config,
            'training_config': self.training_config,
            'data_config': self.data_config,
            'metrics': metrics,
            'best_val_loss': self.best_val_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_path / f"checkpoint-{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_path / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader,
              symbol: str,
              resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            symbol: Stock symbol being trained on
            resume_from: Path to checkpoint to resume from
            
        Returns:
            Training history and metrics
        """
        start_epoch = 0
        history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': [], 'learning_rate': []}
        
        # Resume from checkpoint if specified
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
        
        # Start training
        self.progress_logger.start_training(
            self.training_config.num_epochs, 
            self.model_config.name, 
            symbol
        )
        
        for epoch in range(start_epoch, self.training_config.num_epochs):
            self.progress_logger.start_epoch(epoch + 1)
            
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            
            # Validation phase
            val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            self.progress_logger.log_epoch_metrics(
                epoch + 1,
                train_metrics['loss'],
                val_metrics['loss'],
                train_metrics['mae'],
                val_metrics['mae'],
                current_lr
            )
            
            # Store history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_mae'].append(train_metrics['mae'])
            history['val_mae'].append(val_metrics['mae'])
            history['learning_rate'].append(current_lr)
            
            # WandB logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'train_mae': train_metrics['mae'],
                    'val_mae': val_metrics['mae'],
                    'learning_rate': current_lr
                })
            
            # Check for best model
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.progress_logger.log_best_model(epoch + 1, val_metrics['loss'])
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % self.training_config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch + 1, val_metrics, is_best)
            
            # Early stopping
            if self.early_stopping_counter >= self.training_config.early_stopping_patience:
                self.progress_logger.log_early_stopping(epoch + 1, self.training_config.early_stopping_patience)
                break
        
        # Save final checkpoint
        self.save_checkpoint(epoch + 1, val_metrics, False)
        
        self.progress_logger.finish_training(epoch + 1, None, self.best_val_loss)
        
        return history


def setup_wandb(project_name: str, experiment_name: str, config: Dict[str, Any]) -> bool:
    """Setup WandB experiment tracking."""
    load_dotenv()
    wandb_api_key = os.getenv('WANDB_API_KEY')
    
    if not wandb_api_key:
        logger.warning("No WANDB_API_KEY found in environment")
        logger.info("Set WANDB_API_KEY in environment or .env file")
        logger.info("Training will continue without WandB logging")
        return False
    
    try:
        wandb.login(key=wandb_api_key)
        logger.info("WandB authentication successful")
        
        wandb.init(
            project=project_name,
            name=experiment_name,
            config=config,
            tags=["stock-prediction", "lstm", "time-series"]
        )
        logger.info(f"WandB initialized for experiment: {experiment_name}")
        return True
        
    except Exception as e:
        logger.error(f"WandB setup failed: {e}")
        logger.info("Training will continue without WandB logging")
        return False


@click.command()
@click.option('--model', '-m', default='lstm-small', help='Model architecture to use')
@click.option('--symbol', '-s', default='AAPL', help='Stock symbol to train on')
@click.option('--symbols', multiple=True, help='Multiple symbols to train on')
@click.option('--config', '-c', default='development', help='Training configuration')
@click.option('--data-config', default='enhanced', help='Data configuration')
@click.option('--resume', help='Path to checkpoint to resume from')
@click.option('--data-dir', default='data/raw', help='Directory containing data files')
@click.option('--output-dir', default='models', help='Directory to save models')
@click.option('--wandb/--no-wandb', 'use_wandb', default=True, help='Use WandB for experiment tracking')
@click.option('--list-models', is_flag=True, help='List available models')
@click.option('--list-configs', is_flag=True, help='List available configurations')
@click.option('--list-symbols', is_flag=True, help='List available symbols')
def main(model, symbol, symbols, config, data_config, resume, data_dir, output_dir, use_wandb, 
         list_models, list_configs, list_symbols):
    """Train LSTM models for stock price prediction."""
    
    # List options
    if list_models:
        print("Available models:")
        for model_name in list_available_models():
            print(f"  {model_name}")
        return
    
    if list_configs:
        print("Available training configs:")
        for config_name in list_available_training_configs():
            print(f"  {config_name}")
        print("\nAvailable data configs:")
        for config_name in list_available_data_configs():
            print(f"  {config_name}")
        return
    
    if list_symbols:
        print("Available symbols:")
        for sym in list_available_symbols():
            print(f"  {sym}")
        return
    
    # Setup logging
    log_system_info()
    
    # Get configurations
    try:
        model_config = get_model_config(model)
        training_config = get_training_config(config)
        data_config_obj = get_data_config(data_config)
    except ValueError as e:
        logger.error(str(e))
        return
    
    # Determine symbols to train on
    if symbols:
        symbols_to_train = list(symbols)
    else:
        symbols_to_train = [symbol]
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Train on each symbol
    for sym in symbols_to_train:
        logger.info(f"Starting training for symbol: {sym}")
        
        try:
            # Load and process data
            logger.info("Loading and processing data...")
            processed_data = load_and_process_data(sym, data_config_obj, data_dir)
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(processed_data['X_train']),
                torch.FloatTensor(processed_data['y_train'])
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(processed_data['X_val']),
                torch.FloatTensor(processed_data['y_val'])
            )
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=training_config.batch_size, 
                shuffle=True,
                num_workers=2
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=training_config.batch_size, 
                shuffle=False,
                num_workers=2
            )
            
            # Setup trainer
            save_path = get_model_save_path(model, sym, config)
            trainer = StockTrainer(
                model_config=model_config,
                training_config=training_config,
                data_config=data_config_obj,
                device=device,
                save_path=save_path,
                use_wandb=use_wandb
            )
            
            # Setup WandB first
            if use_wandb:
                experiment_name = get_experiment_name(model, sym, config)
                wandb_config = {
                    'model': model,
                    'symbol': sym,
                    'config': config,
                    'data_config': data_config,
                    **training_config.__dict__,
                    **model_config.__dict__
                }
                
                use_wandb = setup_wandb("stock-prediction", experiment_name, wandb_config)
                trainer.use_wandb = use_wandb
            
            # Setup model after WandB is initialized
            input_size = processed_data['X_train'].shape[-1]
            trainer.setup_model(input_size)
            
            # Update WandB config with model info
            if use_wandb and trainer.use_wandb:
                wandb.config.update({
                    'actual_input_size': input_size,
                    'model_params': count_parameters(trainer.model)[0]
                }, allow_val_change=True)
            
            # Train model
            logger.info("Starting training...")
            history = trainer.train(train_loader, val_loader, sym, resume)
            
            logger.info(f"Training completed for {sym}")
            logger.info(f"Best validation loss: {trainer.best_val_loss:.6f}")
            
            # Cleanup WandB
            if use_wandb and trainer.use_wandb:
                wandb.finish()
            
        except Exception as e:
            logger.error(f"Error training on symbol {sym}: {str(e)}")
            continue
    
    logger.info("Training pipeline completed")


if __name__ == "__main__":
    main()
