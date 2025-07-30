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

# Add the src directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import torch
import wandb
from dotenv import load_dotenv

# Import project modules
from core.logging_config import initialize_project_logging, get_logger, log_system_info
from core.trainer import SegmentationTrainer
from config.settings import get_config, list_available_configs, PASCAL_VOC_CLASSES
from data.loaders import create_datasets, create_dataloaders, visualize_sample
from utils.metrics import visualize_predictions, plot_confusion_matrix

# Initialize module logger
logger = get_logger(__name__)


def setup_wandb(
    project_name: str = "segmentation-bootcamp", 
    model_name: str = "unknown", 
    config_type: str = "default"
) -> Optional[Any]:
    """
    Setup WandB authentication and project initialization.
    
    Args:
        project_name: WandB project name
        model_name: Model identifier
        config_type: Configuration type being used
        
    Returns:
        WandB run instance or None if disabled
    """
    # Load environment variables
    load_dotenv()
    
    try:
        # Initialize WandB
        wandb_run = wandb.init(
            project=project_name,
            name=f"{model_name}_{config_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags=[model_name, config_type, "semantic_segmentation"],
            notes=f"Training {model_name} with {config_type} configuration"
        )
        
        logger.info(f"WandB initialized: {wandb_run.name}")
        logger.info(f"Project: {project_name}")
        
        return wandb_run
    
    except Exception as e:
        logger.warning(f"Failed to initialize WandB: {e}")
        logger.warning("Continuing without WandB logging")
        return None


def evaluate_model(
    trainer: SegmentationTrainer,
    val_loader,
    config,
    save_visualizations: bool = True
) -> Dict[str, float]:
    """
    Evaluate the trained model and generate visualizations.
    
    Args:
        trainer: Trained model trainer instance
        val_loader: Validation data loader
        config: Configuration object
        save_visualizations: Whether to save visualization plots
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Starting model evaluation...")
    
    # Run validation
    val_metrics = trainer.validate_epoch(val_loader)
    
    # Log evaluation results
    logger.info("=== Evaluation Results ===")
    logger.info(f"Validation Loss: {val_metrics['loss']:.4f}")
    logger.info(f"Pixel Accuracy: {val_metrics['pixel_accuracy']:.4f}")
    logger.info(f"Mean IoU: {val_metrics['mean_iou']:.4f}")
    logger.info(f"Mean Dice: {val_metrics['mean_dice']:.4f}")
    
    # Per-class metrics
    logger.info("\n=== Per-Class Metrics ===")
    for i, class_name in enumerate(PASCAL_VOC_CLASSES):
        if i < len(val_metrics['iou_per_class']):
            iou = val_metrics['iou_per_class'][i]
            dice = val_metrics['dice_per_class'][i]
            logger.info(f"{class_name}: IoU={iou:.4f}, Dice={dice:.4f}")
    
    if save_visualizations:
        # Generate sample predictions
        trainer.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= 2:  # Only visualize first 2 batches
                    break
                
                images = batch['image'].to(trainer.device)
                targets = batch['mask']
                predictions = trainer.model(images)
                
                # Save visualization
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
    """
    Run the complete training pipeline.
    
    Args:
        config_name: Configuration name to use
        resume_from: Optional checkpoint path to resume from
        use_wandb: Whether to use WandB logging
        evaluate_only: Whether to only run evaluation (requires resume_from)
    """
    # Load configuration
    config = get_config(config_name)
    logger.info(f"Using configuration: {config_name}")
    logger.info(f"Model: {config.name}")
    logger.info(f"Architecture: {config.architecture}")
    
    # Initialize WandB if enabled
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
        
        # Visualize sample data
        logger.info("Visualizing sample data...")
        from config.settings import get_results_root_dir
        results_dir = get_results_root_dir()
        sample_viz_path = os.path.join(results_dir, "dataset_sample.png")
        visualize_sample(train_dataset, 0, save_path=sample_viz_path)
        
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
            
            if wandb_run:
                wandb_run.log({"final_eval": eval_metrics})
        else:
            # Run training
            logger.info("Starting training...")
            history = trainer.train(train_loader, val_loader)
            
            # Run final evaluation
            logger.info("Running final evaluation...")
            eval_metrics = evaluate_model(trainer, val_loader, config)
            
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
    parser = argparse.ArgumentParser(description="Train semantic segmentation models")
    
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=list_available_configs(),
        help="Configuration to use for training"
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
        help="Disable WandB logging"
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
