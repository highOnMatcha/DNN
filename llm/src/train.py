#!/usr/bin/env python3
"""
Single model training pipeline with comprehensive logging and WandB integration.

This module provides a comprehensive training pipeline for dialog models with
experiment tracking, model evaluation, and generation testing capabilities.
It supports both development and production training configurations with
extensive logging and monitoring.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Add the src directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import torch
import wandb
from dotenv import load_dotenv

# Import logging configuration
from core.logging_config import initialize_project_logging, get_logger, log_system_info, TrainingProgressLogger

from core.trainer import DialogTrainer
from config.settings import (
    get_model_config, 
    get_test_config, 
    get_development_config, 
    get_production_config,
    list_available_models
)
from data.loaders import get_dataset_manager, get_database_manager
from data.streaming import StreamingConfig

# Initialize module logger
logger = get_logger(__name__)


def setup_wandb(project_name: str = "dialog-model-training", model_name: str = "unknown", 
               config_type: str = "test") -> Optional[Any]:
    """
    Setup WandB authentication and project initialization.
    
    This function handles WandB login, project initialization, and run configuration
    for experiment tracking. It gracefully handles authentication failures and
    missing API keys.
    
    Args:
        project_name: Name of the WandB project for organizing experiments.
        model_name: Name of the model being trained for run identification.
        config_type: Type of configuration being used (test/dev/prod).
    
    Returns:
        WandB run object if initialization succeeds, None otherwise.
    """
    load_dotenv()
    wandb_api_key = os.getenv('WANDB_API_KEY')
    
    if not wandb_api_key:
        logger.warning("No WANDB_API_KEY found in environment")
        logger.info("Set WANDB_API_KEY in environment or .env file")
        logger.info("Training will continue without WandB logging")
        return None
    
    try:
        wandb.login(key=wandb_api_key)
        logger.info("WandB authentication successful")
        
        run_name = f"{model_name}-{config_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        wandb_run = wandb.init(
            project=project_name,
            name=run_name,
            tags=[model_name, config_type, "single-model", "pipeline"],
            notes=f"Training {model_name} using {config_type} configuration",
            config={
                "pipeline_version": "1.0",
                "training_type": "single_model",
                "config_type": config_type,
                "timestamp": datetime.now().isoformat(),
            }
        )
        
        logger.info(f"WandB run initialized: {run_name}")
        return wandb_run
        
    except Exception as e:
        logger.error(f"WandB setup failed: {e}")
        logger.info("Training will continue without WandB logging")
        return None

def load_and_validate_configs(model_name: str, config_type: str, max_samples: Optional[int] = None):
    """Load and validate model and training configurations."""
    logger.info("Loading configurations...")
    model_config = get_model_config(model_name)
    
    if config_type == "test":
        training_config = get_test_config()
    elif config_type == "development":
        training_config = get_development_config()
    elif config_type == "production":
        training_config = get_production_config()
    else:
        raise ValueError(f"Unknown config type: {config_type}")
    
    if max_samples is not None:
        training_config.max_samples = max_samples
    
    logger.info(f"Model config: {model_config.name}")
    logger.info(f"Training config: {config_type}")
    logger.info(f"  - Epochs: {training_config.num_epochs}")
    logger.info(f"  - Batch size: {training_config.batch_size}")
    logger.info(f"  - Learning rate: {training_config.learning_rate}")
    logger.info(f"  - LR scheduler: {training_config.lr_scheduler_type}")
    if training_config.warmup_ratio:
        logger.info(f"  - Warmup ratio: {training_config.warmup_ratio:.1%}")
    else:
        logger.info(f"  - Warmup steps: {training_config.warmup_steps}")
    logger.info(f"  - Weight decay: {training_config.weight_decay}")
    if training_config.patience:
        logger.info(f"  - Early stopping patience: {training_config.patience}")
        logger.info(f"  - Early stopping threshold: {training_config.early_stopping_threshold}")
    else:
        logger.info("  - Early stopping: Disabled")
    logger.info(f"  - Max samples: {training_config.max_samples or 'All'}")
    
    return model_config, training_config


def load_and_prepare_dataset(training_config):
    """Load dataset and prepare train/eval splits, prioritizing database sources."""
    logger.info("Loading dataset...")
    dataset_manager = get_dataset_manager()
    
    # Use the new loading method that prioritizes database with shuffling
    df = dataset_manager.load_dataset(
        force_download=False,
        save_to_disk=False,  # Discourage CSV files
        prefer_database=True,  # Prioritize database for streaming support
        shuffle_database=True  # Enable random shuffling for better training diversity
    )
    
    logger.info(f"Dataset loaded: {len(df)} total samples")
    dataset_manager.print_dataset_summary(df)
    
    return df


def check_database_availability() -> bool:
    """Check if database is available and contains data."""
    try:
        db_manager = get_database_manager()
        df = db_manager.load_from_database(shuffle=False)  # No shuffling for availability check
        if df is not None and len(df) > 0:
            logger.info(f"Database available with {len(df)} rows")
            return True
        else:
            logger.warning("Database is available but contains no data")
            return False
    except Exception as e:
        logger.warning(f"Database not available: {e}")
        return False


def check_streaming_availability() -> bool:
    """
    Check if streaming is available by verifying database connectivity.
    
    Returns:
        True if streaming is available, False otherwise
    """
    return check_database_availability()


def prepare_streaming_datasets(trainer, training_config):
    """Prepare streaming datasets for large data training."""
    logger.info("Preparing streaming datasets...")
    
    # Check streaming availability
    if not check_streaming_availability():
        logger.warning("Database not available for streaming")
    
    # Create streaming config based on training config
    streaming_config = StreamingConfig(
        batch_size=min(1000, training_config.max_samples or 1000),
        max_length=512,
        train_split=0.9,
        max_samples=training_config.max_samples  # Pass the max_samples limit
    )
    
    train_dataset, eval_dataset = trainer.prepare_streaming_datasets(
        streaming_config=streaming_config
    )
    
    logger.info("Streaming datasets prepared")
    
    return train_dataset, eval_dataset, streaming_config


def execute_streaming_training(trainer, streaming_config, training_config, resume_from_checkpoint):
    """Execute training with streaming datasets."""
    print("5. Starting streaming training...")
    print("-" * 40)
    start_time = time.time()
    
    trainer_obj = trainer.train_streaming(
        streaming_config=streaming_config,
        training_config=training_config,
        resume_from_checkpoint=resume_from_checkpoint
    )
    
    training_duration = time.time() - start_time
    print("-" * 40)
    print(f"Streaming training completed in {training_duration/60:.1f} minutes")
    print()
    
    return trainer_obj, training_duration


def initialize_trainer(model_config, wandb_run):
    """Initialize trainer with model configuration."""
    print("3. Initializing trainer...")
    trainer = DialogTrainer(
        model_config=model_config,
        wandb_run=wandb_run
    )
    print("Trainer initialized")
    print()
    
    return trainer


def prepare_training_datasets(trainer, df, training_config):
    """Prepare training and evaluation datasets."""
    print("4. Preparing training data...")
    train_dataset, eval_dataset = trainer.prepare_dataset(
        df,
        train_split=0.9,
        max_samples=training_config.max_samples
    )
    print("Dataset prepared")
    print()
    
    return train_dataset, eval_dataset


def execute_training(trainer, train_dataset, eval_dataset, training_config, resume_from_checkpoint):
    """Execute the training process."""
    print("5. Starting training...")
    print("-" * 40)
    start_time = time.time()
    
    trainer_obj = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_config=training_config,
        resume_from_checkpoint=resume_from_checkpoint
    )
    
    training_duration = time.time() - start_time
    print("-" * 40)
    print(f"Training completed in {training_duration/60:.1f} minutes")
    print()
    
    return trainer_obj, training_duration


def test_generation(trainer, wandb_run):
    """Test model generation capabilities and log metrics."""
    print("6. Testing generation capabilities...")
    test_instructions = [
        "Explain what machine learning is in simple terms.",
        "How do I make a good cup of coffee?",
        "What are the benefits of exercise?",
        "Tell me about the solar system.",
        "What is the capital of France?",
        "How does photosynthesis work?",
        "What is the meaning of life?",
        "Describe the process of DNA replication.",
        "What are the main components of a computer?"
    ]
    
    if wandb_run:
        generation_table = wandb.Table(columns=[
            "Instruction", "Response", "Response_Length", 
            "Word_Count", "Generation_Time_Seconds"
        ])
    
    generation_metrics = {
        "total_time": 0.0,
        "total_responses": 0.0,
        "total_length": 0.0,
        "total_words": 0.0
    }
    
    for i, instruction in enumerate(test_instructions, 1):
        print(f"\n[{i}/{len(test_instructions)}] Testing: {instruction[:50]}...")
        
        try:
            start_time = time.time()
            response = trainer.generate_response(
                instruction, 
                max_length=300, 
                log_to_wandb=bool(wandb_run)
            )
            generation_time = time.time() - start_time
            
            print(f"Response: {response[:200]}{'...' if len(response) > 200 else ''}")
            
            generation_metrics["total_time"] += generation_time
            generation_metrics["total_responses"] += 1
            generation_metrics["total_length"] += len(response)
            generation_metrics["total_words"] += len(response.split())
            
            if wandb_run:
                generation_table.add_data(
                    instruction[:100] + "..." if len(instruction) > 100 else instruction,
                    response[:200] + "..." if len(response) > 200 else response,
                    len(response),
                    len(response.split()),
                    round(generation_time, 3)
                )
            
        except Exception as e:
            print(f"Generation failed: {e}")
            if wandb_run:
                wandb_run.log({f"generation/error_{i}": str(e)})
    
    print(f"\nGeneration testing completed")
    print(f"  - Average response time: {generation_metrics['total_time']/max(generation_metrics['total_responses'], 1):.3f}s")
    print(f"  - Average response length: {generation_metrics['total_length']/max(generation_metrics['total_responses'], 1):.1f} chars")
    print(f"  - Average word count: {generation_metrics['total_words']/max(generation_metrics['total_responses'], 1):.1f} words")
    print()
    
    if wandb_run:
        wandb_run.log({"generation_results": generation_table})
        
        if generation_metrics["total_responses"] > 0:
            wandb_run.log({
                "generation_summary/avg_time": generation_metrics["total_time"] / generation_metrics["total_responses"],
                "generation_summary/avg_length": generation_metrics["total_length"] / generation_metrics["total_responses"],
                "generation_summary/avg_words": generation_metrics["total_words"] / generation_metrics["total_responses"],
                "generation_summary/total_responses": generation_metrics["total_responses"],
                "generation_summary/success_rate": generation_metrics["total_responses"] / len(test_instructions)
            })
    
    return generation_metrics


def train_single_model(
    model_name: str = "custom-tiny",
    config_type: str = "test",
    max_samples: Optional[int] = None,
    project_name: str = "dialog-model-training",
    resume_from_checkpoint: Optional[str] = None
) -> Tuple[Any, Dict[str, float]]:
    """Train a single model with comprehensive logging and evaluation."""
    
    # Initialize project-wide logging
    project_logger = initialize_project_logging(
        log_level="INFO",
        model_name=model_name,
        config_type=config_type
    )
    
    logger.info("=" * 60)
    logger.info("SINGLE MODEL TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Configuration: {config_type}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Log comprehensive system information
    log_system_info(logger)
    
    wandb_run = setup_wandb(project_name, model_name, config_type)
    
    try:
        # Setup: Load configurations
        model_config, training_config = load_and_validate_configs(model_name, config_type, max_samples)
        
        # Setup: Load dataset
        df = load_and_prepare_dataset(training_config)
        
        # Setup: Initialize trainer
        trainer = initialize_trainer(model_config, wandb_run)
        
        # Prepare: Create datasets
        train_dataset, eval_dataset = prepare_training_datasets(trainer, df, training_config)
        
        # Train: Execute training
        trainer_obj, training_duration = execute_training(
            trainer, train_dataset, eval_dataset, training_config, resume_from_checkpoint
        )
        
        # Test: Generation capabilities
        generation_metrics = test_generation(trainer, wandb_run)
        
        # Log completion
        if wandb_run:
            wandb_run.log({
                "experiment/status": "completed",
                "experiment/total_duration_minutes": training_duration / 60,
                "experiment/completion_timestamp": datetime.now().isoformat(),
            })
        
        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Model: {model_name}")
        logger.info(f"Output directory: {model_config.output_dir}")
        logger.info(f"Training duration: {training_duration/60:.1f} minutes")
        if wandb_run:
            logger.info(f"WandB run: {wandb_run.name}")
            logger.info(f"WandB URL: {wandb_run.url}")
        
        return trainer, generation_metrics
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        if wandb_run:
            wandb_run.log({
                "experiment/status": "failed",
                "experiment/error": str(e)
            })
        raise
        
    finally:
        if wandb_run:
            wandb_run.finish()
            logger.info("WandB run completed")


def train_single_model_streaming(
    model_name: str = "custom-tiny",
    config_type: str = "test",
    max_samples: Optional[int] = None,
    project_name: str = "dialog-model-training",
    resume_from_checkpoint: Optional[str] = None
) -> Tuple[Any, Dict[str, float]]:
    """Train a single model using streaming for large datasets that don't fit in memory."""
    print("=" * 60)
    print("STREAMING MODEL TRAINING PIPELINE")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Configuration: {config_type}")
    print(f"Streaming Mode: ENABLED")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    wandb_run = setup_wandb(project_name, f"{model_name}-streaming", config_type)
    
    try:
        # Setup: Load configurations
        model_config, training_config = load_and_validate_configs(model_name, config_type, max_samples)
        
        # Setup: Initialize trainer
        trainer = initialize_trainer(model_config, wandb_run)
        
        # Prepare: Create streaming datasets
        train_dataset, eval_dataset, streaming_config = prepare_streaming_datasets(trainer, training_config)
        
        # Train: Execute streaming training
        trainer_obj, training_duration = execute_streaming_training(
            trainer, streaming_config, training_config, resume_from_checkpoint
        )
        
        # Test: Generation capabilities
        generation_metrics = test_generation(trainer, wandb_run)
        
        # Log completion
        if wandb_run:
            wandb_run.log({
                "experiment/status": "completed",
                "experiment/total_duration_minutes": training_duration / 60,
                "experiment/completion_timestamp": datetime.now().isoformat(),
                "experiment/streaming_enabled": True,
                "experiment/streaming_source": "database",
            })
        
        print("=" * 60)
        print("STREAMING TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Model: {model_name}")
        print("Streaming source: database")
        print(f"Output directory: {model_config.output_dir}")
        print(f"Training duration: {training_duration/60:.1f} minutes")
        if wandb_run:
            print(f"WandB run: {wandb_run.name}")
            print(f"WandB URL: {wandb_run.url}")
        print()
        
        return trainer, generation_metrics
        
    except Exception as e:
        print(f"Streaming training failed: {e}")
        if wandb_run:
            wandb_run.log({
                "experiment/status": "failed",
                "experiment/error": str(e),
                "experiment/streaming_enabled": True
            })
        raise
        
    finally:
        if wandb_run:
            wandb_run.finish()
            print("WandB run completed")


def main() -> None:
    """
    Main entry point with command-line argument parsing.
    
    Parses command-line arguments and executes the training pipeline
    with the specified configuration. Provides comprehensive help
    and examples for different training scenarios.
    """
    parser = argparse.ArgumentParser(
        description="Single Model Training Pipeline with WandB Integration and Streaming Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --model custom-tiny --config test --samples 50
  python train.py --model gpt2-small --config development
  python train.py --model custom-small --config production --project my-experiment
  python train.py --model custom-tiny --config test --streaming
  python train.py --list-models
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="custom-tiny",
        help="Model to train (default: custom-tiny)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        choices=["test", "development", "production"],
        default="test",
        help="Training configuration (default: test)"
    )
    
    parser.add_argument(
        "--samples", "-s",
        type=int,
        help="Maximum number of samples to use (overrides config)"
    )
    
    parser.add_argument(
        "--project", "-p",
        type=str,
        default="dialog-model-training",
        help="WandB project name (default: dialog-model-training)"
    )
    
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable streaming mode for large datasets that don't fit in memory"
    )
    
    parser.add_argument(
        "--resume", "-r",
        type=str,
        help="Path to checkpoint directory to resume from"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model configurations and exit"
    )
    
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA even if available"
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available model configurations:")
        list_available_models()
        return
    
    if args.no_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Run the training pipeline
    try:
        if args.streaming:
            print("Starting training in STREAMING mode...")
            trainer, metrics = train_single_model_streaming(
                model_name=args.model,
                config_type=args.config,
                max_samples=args.samples,
                project_name=args.project,
                resume_from_checkpoint=args.resume
            )
        else:
            print("Starting training in STANDARD mode...")
            trainer, metrics = train_single_model(
                model_name=args.model,
                config_type=args.config,
                max_samples=args.samples,
                project_name=args.project,
                resume_from_checkpoint=args.resume
            )
        
        print("Training pipeline completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
