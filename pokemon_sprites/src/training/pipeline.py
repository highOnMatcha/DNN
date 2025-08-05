"""
Main training pipeline orchestration for Pokemon sprite generation.

This module coordinates all training components including dataset management,
model training, transfer learning, and curriculum learning.
"""

import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import wandb
from torch.utils.data import DataLoader

from core.curriculum_training import CurriculumTrainingManager
from core.dataset_manager import DatasetManager
from core.logging_config import get_logger
from core.trainer import PokemonSpriteTrainer
from core.transfer_learning import TransferLearningManager
from training.config_loader import ConfigurationLoader

# Add src to path
current_dir = Path(__file__).parent
src_path = str(current_dir.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)


logger = get_logger(__name__)


class TrainingPipeline:
    """Orchestrates the complete training pipeline for Pokemon sprite generation."""

    def __init__(self, args):
        """
        Initialize training pipeline.

        Args:
            args: Command line arguments or configuration object
        """
        self.args = args
        self.config_loader = ConfigurationLoader()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize configuration attributes
        self.model_config = None
        self.training_config = None

        # Initialize managers
        self.dataset_manager = None
        self.transfer_manager = None
        self.curriculum_manager = None
        self.trainer = None
        self.wandb_run = None

    def setup_environment(self) -> bool:
        """
        Setup the training environment.

        Returns:
            True if setup successful, False otherwise
        """
        try:
            logger.info("Setting up training environment...")

            # Load and validate configurations
            # Re-create config_loader to ensure it uses mocked version in tests
            self.config_loader = ConfigurationLoader()
            self.model_config, self.training_config = (
                self.config_loader.load_and_validate_configs(
                    self.args.model, self.args.config
                )
            )

            # Override max samples if specified
            if hasattr(self.args, "max_samples") and self.args.max_samples:
                self.training_config.max_samples = self.args.max_samples

            # Initialize dataset manager
            data_root = self.config_loader.get_data_root_dir()
            self.dataset_manager = DatasetManager(data_root)

            # Setup dataset if missing
            if not self.dataset_manager.validate_and_prepare_data():
                logger.error("Failed to setup dataset")
                return False

            # Initialize transfer learning manager
            self.transfer_manager = TransferLearningManager(self.device)

            # Initialize curriculum learning manager if needed
            if self._should_use_curriculum():
                processed_dir = (
                    self.dataset_manager.pokemon_data_dir / "processed"
                )
                self.curriculum_manager = CurriculumTrainingManager(
                    self.training_config.__dict__, processed_dir
                )

            logger.info("Environment setup complete")
            return True

        except Exception as e:
            logger.error(f"Environment setup failed: {e}")
            return False

    def setup_wandb(self) -> Optional[Any]:
        """
        Setup Weights & Biases tracking.

        Returns:
            WandB run object or None if setup failed
        """
        try:
            if not hasattr(self.args, "wandb") or not self.args.wandb:
                logger.info("WandB tracking disabled")
                return None

            if self.model_config is None or self.training_config is None:
                logger.warning(
                    "Model or training configuration not loaded, skipping WandB setup"
                )
                return None

            # Load environment variables from .env file
            try:
                import os

                from dotenv import load_dotenv

                # Look for .env file in project root
                project_root = Path(__file__).parent.parent.parent
                env_file = project_root / ".env"

                if env_file.exists():
                    load_dotenv(env_file)
                    logger.debug(
                        f"Loaded environment variables from {env_file}"
                    )
                else:
                    logger.debug(
                        "No .env file found, using system environment variables"
                    )

            except ImportError:
                logger.warning(
                    "python-dotenv not installed, using system environment variables only"
                )
                import os

            # Check for API key (now potentially loaded from .env file)
            wandb_key = os.getenv("WANDB_API_KEY")
            if not wandb_key:
                logger.warning(
                    "WANDB_API_KEY not found in environment or .env file, skipping WandB setup"
                )
                return None

            logger.info("WandB API key found, initializing tracking...")

            # Initialize WandB
            config = {
                "model": self.model_config.name,
                "architecture": self.model_config.architecture,
                "training_config": self.args.config,
                "batch_size": self.training_config.batch_size,
                "learning_rate": self.training_config.learning_rate,
                "epochs": self.training_config.epochs,
            }

            run = wandb.init(
                project="pokemon-sprite-generation",
                name=f"{self.model_config.name}_{self.args.config}_{int(time.time())}",
                config=config,
                tags=[self.model_config.name, self.args.config],
            )

            logger.info(f"WandB tracking initialized: {run.name}")
            return run

        except Exception as e:
            logger.warning(f"WandB setup failed: {e}")
            return None

    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Create training and validation data loaders.

        Returns:
            Tuple of (train_loader, val_loader)
        """
        if self.training_config is None:
            raise AttributeError(
                "Training configuration not loaded. Call setup_environment() first."
            )

        from training.data_loaders import create_data_loaders

        augmentation_level = (
            self.args.augmentation
            if hasattr(self.args, "augmentation") and self.args.augmentation
            else "conservative"
        )

        max_samples = (
            self.args.max_samples
            if hasattr(self.args, "max_samples")
            else None
        )

        return create_data_loaders(
            self.training_config, augmentation_level, max_samples
        )

    def create_trainer(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
    ) -> PokemonSpriteTrainer:
        """
        Create and configure the trainer.

        Args:
            train_loader: Training data loader (for compatibility, not used in current implementation)
            val_loader: Validation data loader (for compatibility, not used in current implementation)

        Returns:
            Configured trainer instance
        """
        if self.model_config is None or self.training_config is None:
            raise AttributeError(
                "Model or training configuration not loaded. Call setup_environment() first."
            )

        trainer = PokemonSpriteTrainer(
            model_config=self.model_config,
            training_config=self.training_config,
            wandb_run=self.wandb_run,
        )

        # Apply transfer learning if recommended
        if (
            self.transfer_manager
            and self.transfer_manager.should_use_transfer_learning(
                self.model_config.__dict__
            )
        ):
            logger.info("Transfer learning recommended for this model")
            # Transfer learning will be applied during model creation in
            # trainer

        return trainer

    def run_training(
        self,
        trainer: Optional[PokemonSpriteTrainer] = None,
        wandb_run: Optional[Any] = None,
    ) -> bool:
        """
        Execute the main training loop.

        Args:
            trainer: The trainer instance (for compatibility with tests)
            wandb_run: WandB run instance (for compatibility with tests)

        Returns:
            True if training completed successfully, False otherwise
        """
        try:
            logger.info("Starting training pipeline...")

            # Use provided trainer or create data loaders and trainer
            if trainer is None:
                # Create data loaders
                train_loader, val_loader = self.create_data_loaders()

                # Create trainer
                self.trainer = self.create_trainer()
            else:
                # Use provided trainer for testing
                self.trainer = trainer
                # Create data loaders for the actual training
                train_loader, val_loader = self.create_data_loaders()

            # Set wandb_run if provided
            if wandb_run is not None:
                self.wandb_run = wandb_run

            # Execute training
            start_time = time.time()

            if self.curriculum_manager and self._should_use_curriculum():
                success = self._run_curriculum_training(
                    train_loader, val_loader
                )
            else:
                success = self._run_standard_training(train_loader, val_loader)

            training_time = time.time() - start_time

            if success and self.training_config:
                logger.info(
                    f"Training completed successfully in {training_time:.2f} seconds"
                )
                logger.info(
                    f"Average time per epoch: "
                    f"{training_time / self.training_config.epochs:.2f} seconds"
                )
            else:
                logger.error("Training failed")

            return success

        except Exception as e:
            logger.error(f"Training execution failed: {e}")
            return False

    def _run_standard_training(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> bool:
        """Run standard training without curriculum learning."""
        try:
            if self.trainer:
                result = self.trainer.train(train_loader, val_loader)
                # Handle case where trainer.train returns a boolean
                if result is False:
                    return False
                return True
            else:
                logger.error("No trainer available for training")
                return False
        except Exception as e:
            logger.error(f"Standard training failed: {e}")
            return False

    def _run_curriculum_training(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> bool:
        """Run curriculum-based training."""
        try:
            if not self.curriculum_manager or not self.training_config:
                logger.error(
                    "Curriculum manager or training config not available"
                )
                return False

            # Get phase configuration
            phase_epochs = self.curriculum_manager.get_phase_epochs(
                self.training_config.epochs
            )

            for phase_idx, epochs in enumerate(phase_epochs):
                phase_info = (
                    self.curriculum_manager.get_curriculum_phase_info()
                )
                logger.info(
                    f"Starting curriculum phase {phase_info['phase']}/{phase_info['total_phases']} "
                    f"({phase_info['current_scale']}px) for {epochs} epochs"
                )

                # Update training config for this phase
                phase_config = self.training_config
                phase_config.epochs = epochs

                # Train for this phase
                if self.trainer:
                    self.trainer.train(train_loader, val_loader)
                else:
                    logger.error(
                        "No trainer available for curriculum training"
                    )
                    return False

                # Advance curriculum for next phase
                if not self.curriculum_manager.advance_curriculum():
                    break

            logger.info("Curriculum training completed")
            return True

        except Exception as e:
            logger.error(f"Curriculum training failed: {e}")
            return False

    def _should_use_curriculum(self) -> bool:
        """Check if curriculum learning should be used."""
        if not self.curriculum_manager or not self.model_config:
            return False
        return self.curriculum_manager.should_use_curriculum(
            self.model_config.name
        )

    def handle_post_training_generation(self) -> None:
        """Handle sprite generation after training completion."""
        try:
            if not hasattr(self.args, "generate") or not self.args.generate:
                return

            if not self.trainer:
                logger.warning(
                    "No trainer available for post-training generation"
                )
                return

            logger.info("Starting post-training sprite generation...")

            # Import generation utilities
            from training.sprite_generator import (
                handle_post_training_generation,
            )

            handle_post_training_generation(
                self.args, self.trainer, self.wandb_run
            )

        except Exception as e:
            logger.error(f"Post-training generation failed: {e}")

    def cleanup(self) -> None:
        """Clean up resources and finalize tracking."""
        try:
            # Clean up curriculum temporary structures
            if self.curriculum_manager:
                self.curriculum_manager.cleanup_temp_structures()

            # Finalize WandB
            if self.wandb_run:
                import wandb

                wandb.finish()

            logger.info("Pipeline cleanup completed")

        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

    def execute(self) -> bool:
        """
        Execute the complete training pipeline.

        Returns:
            True if pipeline executed successfully, False otherwise
        """
        try:
            # Setup environment
            if not self.setup_environment():
                return False

            # Setup tracking
            self.wandb_run = self.setup_wandb()

            # Run training
            success = self.run_training()

            # Handle post-training tasks
            if success:
                self.handle_post_training_generation()

            return success

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return False
        finally:
            self.cleanup()
