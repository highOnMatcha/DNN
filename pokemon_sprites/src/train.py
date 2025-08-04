#!/usr/bin/env python3
"""
Pokemon Sprite Generation Training Pipeline

Refactored training script with modular architecture supporting the three best models:
- lightweight-baseline: Fast training with minimal parameters
- sprite-optimized: State-of-the-art configuration optimized for pixel art sprites
- transformer-enhanced: Advanced transformer-enhanced architecture

Usage:
    python train.py --model sprite-optimized --config production
    python train.py --model lightweight-baseline --config development --max-samples 1000
    python train.py --model transformer-enhanced --config production --wandb --generate
"""

import argparse
import sys
from pathlib import Path

from core.logging_config import (
    get_logger,
    initialize_project_logging,
    log_system_info,
)
from training.config_loader import ConfigurationLoader
from training.pipeline import TrainingPipeline

# Add the src directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


# Initialize module logger
logger = get_logger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Setup command line argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Pokemon Sprite Generation Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick development training
  python train.py --model lightweight-baseline --config development

  # Production training with transfer learning and curriculum
  python train.py --model sprite-optimized --config production --wandb

  # Advanced transformer model with all features
  python train.py --model transformer-enhanced --config production --wandb --generate

  # Test configuration with limited samples
  python train.py --model sprite-optimized --config test --max-samples 100
        """,
    )

    # Model selection
    config_loader = ConfigurationLoader()
    available_models = config_loader.get_available_models()
    available_configs = config_loader.get_available_configs()

    parser.add_argument(
        "--model",
        choices=available_models,
        required=True,
        help="Model architecture to train",
    )

    parser.add_argument(
        "--config",
        choices=available_configs,
        required=True,
        help="Training configuration to use",
    )

    # Training options
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of training samples (for testing/debugging)",
    )

    parser.add_argument(
        "--augmentation",
        choices=["none", "conservative", "minimal", "moderate", "strong"],
        help="Data augmentation level (defaults to config level)",
    )

    # Advanced features
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases tracking"
    )

    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate missing sprites after training",
    )

    parser.add_argument(
        "--max-generate",
        type=int,
        help="Maximum number of sprites to generate",
    )

    # Resuming training
    parser.add_argument(
        "--resume", type=str, help="Path to checkpoint to resume training from"
    )

    # Logging options
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Custom experiment name for logging",
    )

    return parser


def initialize_experiment_logging(args) -> str:
    """
    Initialize logging for the experiment.

    Args:
        args: Command line arguments

    Returns:
        Experiment ID string
    """
    # Initialize project logging
    initialize_project_logging(
        log_level=args.log_level, experiment_id=args.experiment_name
    )

    # Log system information
    log_system_info()

    # Generate experiment ID
    import time

    experiment_id = f"{args.model}_{args.config}_{int(time.time())}"

    logger.info("=" * 80)
    logger.info("POKEMON SPRITE GENERATION TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Experiment ID: {experiment_id}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Configuration: {args.config}")

    if args.max_samples:
        logger.info(f"Max samples: {args.max_samples}")
    if args.wandb:
        logger.info("WandB tracking: ENABLED")
    if args.generate:
        logger.info("Post-training generation: ENABLED")

    logger.info("=" * 80)

    return experiment_id


def validate_arguments(args) -> bool:
    """
    Validate command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        True if arguments are valid, False otherwise
    """
    try:
        # Validate model and config combination
        config_loader = ConfigurationLoader()
        model_info = config_loader.get_model_info(args.model)

        if model_info and "recommended_config" in model_info:
            recommended = model_info["recommended_config"]
            if args.config == "test" and recommended == "production":
                logger.warning(
                    f"Using test config with {args.model} model. "
                    f"Recommended config: {recommended}"
                )

        # Validate max_samples
        if args.max_samples and args.max_samples <= 0:
            logger.error("max-samples must be positive")
            return False

        # Validate max_generate
        if args.max_generate and args.max_generate <= 0:
            logger.error("max-generate must be positive")
            return False

        # Validate resume path
        if args.resume and not Path(args.resume).exists():
            logger.error(f"Resume checkpoint not found: {args.resume}")
            return False

        return True

    except Exception as e:
        logger.error(f"Argument validation failed: {e}")
        return False


def print_model_recommendations(args) -> None:
    """Print model-specific recommendations."""
    config_loader = ConfigurationLoader()
    model_info = config_loader.get_model_info(args.model)

    if not model_info:
        return

    logger.info(f"Model: {args.model}")
    logger.info(
        f"Description: {model_info.get('description', 'No description available')}"
    )

    features = model_info.get("features", [])
    if features:
        logger.info("Features:")
        for feature in features:
            logger.info(f"  - {feature}")

    # Recommendations
    recommendations = []
    if model_info.get("use_transfer_learning", False):
        recommendations.append("Transfer learning recommended")
    if model_info.get("use_curriculum", False):
        recommendations.append("Curriculum learning will be applied")

    if recommendations:
        logger.info("Training features:")
        for rec in recommendations:
            logger.info(f"  - {rec}")


def main():
    """Main training function."""
    try:
        # Parse arguments
        parser = setup_argument_parser()
        args = parser.parse_args()

        # Initialize logging
        initialize_experiment_logging(args)

        # Validate arguments
        if not validate_arguments(args):
            logger.error("Argument validation failed")
            sys.exit(1)

        # Print model information
        print_model_recommendations(args)

        # Load and print configuration summary
        config_loader = ConfigurationLoader()
        model_config, training_config = (
            config_loader.load_and_validate_configs(args.model, args.config)
        )
        config_loader.print_configuration_summary(
            model_config, training_config
        )

        # Create and execute training pipeline
        pipeline = TrainingPipeline(args)
        success = pipeline.execute()

        if success:
            logger.info("Training pipeline completed successfully")
            sys.exit(0)
        else:
            logger.error("Training pipeline failed")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT

    except Exception as e:
        logger.error(
            f"Training pipeline failed with unexpected error: {e}",
            exc_info=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
