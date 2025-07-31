#!/usr/bin/env python3
"""
Main training script for semantic segmentation models.

This script provides a comprehensive training pipeline for semantic segmentation
with experiment tracking, model evaluation, and visualization capabilities.
It supports different configurations and models with extensive logging and monitoring.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Add the src directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import torch
import wandb

# Import project modules
from core.logging_config import initialize_project_logging, get_logger, log_system_info
from core.trainer import SegmentationTrainer
from config.settings import get_config, list_available_configs, PASCAL_VOC_CLASSES
from data.loaders import create_datasets, create_dataloaders
from utils.metrics import visualize_predictions, plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Initialize module logger
logger = get_logger(__name__)


def log_test_predictions(trainer, val_loader, config, wandb_run=None, num_samples=8):
    """Log sample predictions from validation/test set with visualizations."""
    logger.info(f"Logging {num_samples} test predictions...")
    
    trainer.model.eval()
    device = next(trainer.model.parameters()).device
    
    samples_logged = 0
    prediction_images = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if samples_logged >= num_samples:
                break
                
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = trainer.model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            for i in range(images.size(0)):
                if samples_logged >= num_samples:
                    break
                    
                image = images[i].cpu().numpy().transpose(1, 2, 0)
                image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                image = np.clip(image, 0, 1)
                
                true_mask = masks[i].cpu().numpy()
                pred_mask = predictions[i].cpu().numpy()
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(image)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                axes[1].imshow(true_mask, cmap='tab20', vmin=0, vmax=20)
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                axes[2].imshow(pred_mask, cmap='tab20', vmin=0, vmax=20)
                axes[2].set_title('Prediction')
                axes[2].axis('off')
                
                plt.tight_layout()
                
                if wandb_run:
                    wandb_run.log({f"test_prediction_{samples_logged}": wandb.Image(fig)})
                
                prediction_images.append(fig)
                plt.close(fig)
                
                samples_logged += 1
    
    logger.info(f"Successfully logged {samples_logged} test predictions")
    return prediction_images


def setup_wandb(
    project_name: str = "image-segmentation", 
    model_name: str = "unknown", 
    config_type: str = "default"
) -> Optional[Any]:
    """Setup WandB authentication and project initialization."""
    try:
        wandb_run = wandb.init(
            project=project_name,
            name=f"{model_name}_{config_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags=[model_name, config_type, "semantic_segmentation"],
            notes=f"Training {model_name} with {config_type} configuration"
        )
        
        logger.info(f"WandB initialized: {wandb_run.name}")
        logger.info(f"Project: {project_name}")
        logger.info(f"View at: {wandb_run.url}")
        
        return wandb_run
    
    except Exception as e:
        logger.warning(f"Failed to initialize WandB: {e}")
        logger.warning("Continuing without WandB logging")
        return None


def display_model_info(config: Any, config_name: str) -> None:
    """Display comprehensive model and training information."""
    logger.info("=" * 80)
    logger.info("SEGMENTATION TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Configuration: {config_name}")
    logger.info(f"Model Name: {config.name}")
    logger.info(f"Architecture: {config.architecture}")
    logger.info(f"Encoder: {config.encoder_name}")
    logger.info(f"Pre-trained Weights: {config.encoder_weights}")
    logger.info(f"Number of Classes: {config.classes}")
    logger.info(f"Input Size: {config.image_size}")
    logger.info(f"Batch Size: {config.batch_size}")
    logger.info(f"Epochs: {config.num_epochs}")
    logger.info(f"Learning Rate: {config.learning_rate}")
    logger.info(f"Optimizer: {config.optimizer}")
    logger.info(f"WandB Logging: {'Enabled' if config.use_wandb else 'Disabled'}")
    logger.info(f"Data Augmentation: {'Enabled' if config.use_augmentation else 'Disabled'}")
    logger.info(f"Mixed Precision: {'Enabled' if config.mixed_precision else 'Disabled'}")
    logger.info("=" * 80)


def evaluate_model(
    trainer: SegmentationTrainer,
    val_loader,
    config,
    save_visualizations: bool = True
) -> Dict[str, float]:
    """Evaluate the trained model and generate visualizations."""
    logger.info("Starting model evaluation...")
    
    val_metrics = trainer.validate_epoch(val_loader)
    
    logger.info("=== Evaluation Results ===")
    logger.info(f"Validation Loss: {val_metrics['loss']:.4f}")
    logger.info(f"Pixel Accuracy: {val_metrics['pixel_accuracy']:.4f}")
    logger.info(f"Mean IoU: {val_metrics['mean_iou']:.4f}")
    logger.info(f"Mean Dice: {val_metrics['mean_dice']:.4f}")
    
    logger.info("\n=== Per-Class Metrics ===")
    for i, class_name in enumerate(PASCAL_VOC_CLASSES):
        if i < len(val_metrics['iou_per_class']):
            iou = val_metrics['iou_per_class'][i]
            dice = val_metrics['dice_per_class'][i]
            logger.info(f"{class_name}: IoU={iou:.4f}, Dice={dice:.4f}")
    
    if save_visualizations:
        trainer.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= 2:
                    break
                
                images = batch['image'].to(trainer.device)
                targets = batch['mask']
                predictions = trainer.model(images)
                
                from config.settings import get_results_root_dir
                results_dir = get_results_root_dir()
                save_path = os.path.join(results_dir, f"predictions_batch_{batch_idx}.png")
                
                visualize_predictions(
                    images.cpu(),
                    predictions.cpu(),
                    targets,
                    PASCAL_VOC_CLASSES,
                    save_path=save_path,
                    num_samples=min(4, images.size(0))
                )
                
                logger.info(f"Visualization saved: {save_path}")
    
    return val_metrics


def run_training(
    config_name: str = "default",
    resume_from: Optional[str] = None,
    use_wandb: bool = True,
    evaluate_only: bool = False
) -> None:
    """Run the complete training pipeline."""
    config = get_config(config_name)
    
    display_model_info(config, config_name)
    
    wandb_run = None
    if use_wandb and config.use_wandb:
        wandb_run = setup_wandb(
            project_name=config.wandb_project,
            model_name=config.name,
            config_type=config_name
        )
    
    try:
        # Create datasets
        logger.info("Creating datasets...")
        train_dataset, val_dataset = create_datasets(config)
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        # Create data loaders
        train_loader, val_loader = create_dataloaders(config, train_dataset, val_dataset)
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = SegmentationTrainer(config, wandb_run)
        
        # Resume from checkpoint if specified
        if resume_from:
            trainer.load_checkpoint(resume_from)
            logger.info(f"Resumed training from {resume_from}")
        
        if evaluate_only:
            if not resume_from:
                raise ValueError("evaluate_only requires resume_from checkpoint")
            
            # Only run evaluation
            eval_metrics = evaluate_model(trainer, val_loader, config)
            
            # Log test predictions for evaluation-only mode
            logger.info("Logging test set predictions...")
            log_test_predictions(trainer, val_loader, config, wandb_run, num_samples=8)
            
            if wandb_run:
                wandb_run.log({"final_eval": eval_metrics})
        else:
            # Run training
            logger.info("Starting training...")
            history = trainer.train(train_loader, val_loader)
            
            # Run final evaluation
            logger.info("Running final evaluation...")
            eval_metrics = evaluate_model(trainer, val_loader, config)
            
            # Log test predictions
            logger.info("Logging test set predictions...")
            log_test_predictions(trainer, val_loader, config, wandb_run, num_samples=8)
            
            # Log final metrics to WandB
            if wandb_run:
                wandb_run.log({"final_eval": eval_metrics})
                
                # Log training history as plots
                import matplotlib.pyplot as plt
                
                # Plot training curves
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                epochs = range(1, len(history['train_loss']) + 1)
                
                axes[0, 0].plot(epochs, history['train_loss'], label='Train')
                axes[0, 0].plot(epochs, history['val_loss'], label='Validation')
                axes[0, 0].set_title('Loss')
                axes[0, 0].legend()
                
                axes[0, 1].plot(epochs, history['train_mean_iou'], label='Train')
                axes[0, 1].plot(epochs, history['val_mean_iou'], label='Validation')
                axes[0, 1].set_title('Mean IoU')
                axes[0, 1].legend()
                
                axes[1, 0].plot(epochs, history['train_pixel_accuracy'], label='Train')
                axes[1, 0].plot(epochs, history['val_pixel_accuracy'], label='Validation')
                axes[1, 0].set_title('Pixel Accuracy')
                axes[1, 0].legend()
                
                axes[1, 1].axis('off')  # Leave empty for now
                
                plt.tight_layout()
                wandb_run.log({"training_curves": wandb.Image(fig)})
                plt.close()
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        if wandb_run:
            wandb_run.finish()


def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(
        description="Semantic Segmentation Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Configurations:
  default      - UNet with ResNet50 encoder (production settings)
  test         - Fast testing config (small images, few epochs)
  development  - Development config (moderate settings)
  production   - Full production config (large images, many epochs)

Examples:
  python src/train.py --config test
  python src/train.py --config default
  python src/train.py --config production --resume checkpoints/model.pth
  python src/train.py --evaluate-only --resume checkpoints/best_model.pth
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=list_available_configs(),
        help="Configuration to use for training (default: %(default)s)"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging (WandB is enabled by default)"
    )
    
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only run evaluation (requires --resume)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Initialize logging
    logger = initialize_project_logging(
        model_name="segmentation",
        config_type=args.config,
        log_level=args.log_level
    )
    
    # Log system information
    log_system_info(logger)
    
    # Log script arguments
    logger.info("=== Training Arguments ===")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info("=== End Arguments ===")
    
    # Run training
    run_training(
        config_name=args.config,
        resume_from=args.resume,
        use_wandb=not args.no_wandb,
        evaluate_only=args.evaluate_only
    )


if __name__ == "__main__":
    main()
