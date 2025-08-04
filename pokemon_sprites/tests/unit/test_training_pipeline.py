"""
Comprehensive unit tests for training/pipeline.py module.

This module provides testing coverage for the TrainingPipeline class functionality,
including environment setup, data loading, model initialization, and training execution.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.logging_config import get_logger
from training.pipeline import TrainingPipeline

logger = get_logger(__name__)


class TestTrainingPipelineSetup(unittest.TestCase):
    """Test TrainingPipeline setup and initialization functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.mock_args = Mock()
        self.mock_args.model = "lightweight-baseline"
        self.mock_args.config = "development"
        self.mock_args.max_samples = None
        self.mock_args.wandb = False
        self.mock_args.generate = False

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    @patch("training.config_loader.ConfigurationLoader")
    def test_pipeline_initialization(self, mock_config_loader_class):
        """Test TrainingPipeline initialization."""
        mock_config_loader = Mock()
        mock_config_loader_class.return_value = mock_config_loader

        pipeline = TrainingPipeline(self.mock_args)

        self.assertEqual(pipeline.args, self.mock_args)
        self.assertIsNotNone(pipeline.config_loader)
        logger.info("Pipeline initialization verified")

    @patch("training.pipeline.ConfigurationLoader")
    @patch("training.pipeline.DatasetManager")
    def test_setup_environment_success(
        self, mock_dataset_manager_class, mock_config_loader_class
    ):
        """Test successful environment setup."""
        # Mock configuration loader
        mock_config_loader = Mock()
        mock_model_config = Mock()
        mock_training_config = Mock()
        mock_config_loader.load_and_validate_configs.return_value = (
            mock_model_config,
            mock_training_config,
        )
        mock_config_loader.get_data_root_dir.return_value = Path("/tmp")
        mock_config_loader_class.return_value = mock_config_loader

        # Mock dataset manager
        mock_dataset_manager = Mock()
        mock_dataset_manager.validate_and_prepare_data.return_value = True
        mock_dataset_manager_class.return_value = mock_dataset_manager

        with patch("core.transfer_learning.TransferLearningManager"):
            pipeline = TrainingPipeline(self.mock_args)
            result = pipeline.setup_environment()

            self.assertTrue(result)
            mock_config_loader.load_and_validate_configs.assert_called_once()
            mock_dataset_manager.validate_and_prepare_data.assert_called_once()
            logger.info("Environment setup success path verified")

    @patch("training.pipeline.ConfigurationLoader")
    def test_setup_environment_config_failure(self, mock_config_loader_class):
        """Test environment setup with configuration failure."""
        mock_config_loader = Mock()
        mock_config_loader.load_and_validate_configs.side_effect = Exception(
            "Config error"
        )
        mock_config_loader_class.return_value = mock_config_loader

        pipeline = TrainingPipeline(self.mock_args)
        result = pipeline.setup_environment()

        self.assertFalse(result)
        mock_config_loader.load_and_validate_configs.assert_called_once()
        logger.info("Environment setup config failure handled correctly")

    @patch("training.pipeline.ConfigurationLoader")
    @patch("training.pipeline.DatasetManager")
    def test_setup_environment_dataset_failure(
        self, mock_dataset_manager_class, mock_config_loader_class
    ):
        """Test environment setup with dataset validation failure."""
        # Mock successful config loading
        mock_config_loader = Mock()
        mock_config_loader.load_and_validate_configs.return_value = (
            Mock(),
            Mock(),
        )
        mock_config_loader.get_data_root_dir.return_value = Path("/tmp")
        mock_config_loader_class.return_value = mock_config_loader

        # Mock dataset manager failure
        mock_dataset_manager = Mock()
        mock_dataset_manager.validate_and_prepare_data.return_value = False
        mock_dataset_manager_class.return_value = mock_dataset_manager

        pipeline = TrainingPipeline(self.mock_args)
        result = pipeline.setup_environment()

        self.assertFalse(result)
        mock_config_loader.load_and_validate_configs.assert_called_once()
        mock_dataset_manager.validate_and_prepare_data.assert_called_once()
        logger.info("Environment setup dataset failure handled correctly")

    @patch("training.data_loaders.create_data_loaders")
    @patch("training.config_loader.ConfigurationLoader")
    def test_create_data_loaders_success(
        self, mock_config_loader_class, mock_create_loaders
    ):
        """Test successful data loader creation."""
        mock_config_loader = Mock()
        mock_config_loader.load_and_validate_configs.return_value = (
            Mock(),
            Mock(),
        )
        mock_config_loader.get_data_root_dir.return_value = Path("/tmp")
        mock_config_loader_class.return_value = mock_config_loader

        mock_train_loader = Mock()
        mock_val_loader = Mock()
        mock_create_loaders.return_value = (mock_train_loader, mock_val_loader)

        with patch(
            "core.dataset_manager.DatasetManager"
        ) as mock_dataset_manager_class:
            mock_dataset_manager = Mock()
            mock_dataset_manager.validate_and_prepare_data.return_value = True
            mock_dataset_manager_class.return_value = mock_dataset_manager

            with patch("core.transfer_learning.TransferLearningManager"):
                pipeline = TrainingPipeline(self.mock_args)
                pipeline.setup_environment()  # Setup environment first
                train_loader, val_loader = pipeline.create_data_loaders()

                self.assertEqual(train_loader, mock_train_loader)
                self.assertEqual(val_loader, mock_val_loader)
                mock_create_loaders.assert_called_once()
                logger.info("Data loader creation success verified")

    @patch("training.data_loaders.create_data_loaders")
    @patch("training.config_loader.ConfigurationLoader")
    def test_create_data_loaders_with_max_samples(
        self, mock_config_loader_class, mock_create_loaders
    ):
        """Test data loader creation with max samples limitation."""
        mock_config_loader = Mock()
        mock_config_loader.load_and_validate_configs.return_value = (
            Mock(),
            Mock(),
        )
        mock_config_loader.get_data_root_dir.return_value = Path("/tmp")
        mock_config_loader_class.return_value = mock_config_loader

        # Set max_samples in args
        self.mock_args.max_samples = 100
        self.mock_args.augmentation = "moderate"

        mock_train_loader = Mock()
        mock_val_loader = Mock()
        mock_create_loaders.return_value = (mock_train_loader, mock_val_loader)

        with patch(
            "core.dataset_manager.DatasetManager"
        ) as mock_dataset_manager_class:
            mock_dataset_manager = Mock()
            mock_dataset_manager.validate_and_prepare_data.return_value = True
            mock_dataset_manager_class.return_value = mock_dataset_manager

            with patch("core.transfer_learning.TransferLearningManager"):
                pipeline = TrainingPipeline(self.mock_args)
                pipeline.setup_environment()  # Setup environment first
                train_loader, val_loader = pipeline.create_data_loaders()

                # Verify create_data_loaders was called with correct parameters
                mock_create_loaders.assert_called_once()
                call_args = mock_create_loaders.call_args
                # Note: create_data_loaders doesn't use kwargs anymore, check positional args
                self.assertEqual(
                    call_args[0][1], "moderate"
                )  # augmentation_level
                self.assertEqual(call_args[0][2], 100)  # max_samples
                logger.info("Data loader creation with max samples verified")

    @patch("training.pipeline.PokemonSpriteTrainer")
    @patch("training.config_loader.ConfigurationLoader")
    def test_create_trainer_success(
        self, mock_config_loader_class, mock_trainer_class
    ):
        """Test successful trainer creation."""
        mock_config_loader = Mock()
        mock_model_config = Mock()
        mock_training_config = Mock()
        mock_training_config.device = (
            "cpu"  # Add device attribute to avoid errors
        )
        mock_config_loader.load_and_validate_configs.return_value = (
            mock_model_config,
            mock_training_config,
        )
        mock_config_loader.get_data_root_dir.return_value = Path("/tmp")
        mock_config_loader_class.return_value = mock_config_loader

        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer

        with patch(
            "core.dataset_manager.DatasetManager"
        ) as mock_dataset_manager_class:
            mock_dataset_manager = Mock()
            mock_dataset_manager.validate_and_prepare_data.return_value = True
            mock_dataset_manager_class.return_value = mock_dataset_manager

            with patch(
                "core.transfer_learning.TransferLearningManager"
            ) as mock_transfer_class:
                mock_transfer = Mock()
                mock_transfer.should_use_transfer_learning.return_value = False
                mock_transfer_class.return_value = mock_transfer

                pipeline = TrainingPipeline(self.mock_args)
                pipeline.setup_environment()  # Setup environment first

                # Mock data loaders
                mock_train_loader = Mock()
                mock_val_loader = Mock()

                trainer = pipeline.create_trainer(
                    mock_train_loader, mock_val_loader
                )

                self.assertEqual(trainer, mock_trainer)
                mock_trainer_class.assert_called_once()
                logger.info("Trainer creation success verified")

    @patch("training.config_loader.ConfigurationLoader")
    def test_create_trainer_missing_configs(self, mock_config_loader_class):
        """Test trainer creation with missing configurations."""
        mock_config_loader = Mock()
        mock_config_loader_class.return_value = mock_config_loader

        pipeline = TrainingPipeline(self.mock_args)
        # Don't set model_config and training_config

        mock_train_loader = Mock()
        mock_val_loader = Mock()

        with self.assertRaises(AttributeError):
            pipeline.create_trainer(mock_train_loader, mock_val_loader)

        logger.info("Trainer creation with missing configs handled correctly")


class TestTrainingPipelineExecution(unittest.TestCase):
    """Test TrainingPipeline execution workflow."""

    def setUp(self):
        """Set up test environment."""
        self.mock_args = Mock()
        self.mock_args.model = "sprite-optimized"
        self.mock_args.config = "production"
        self.mock_args.wandb = True
        self.mock_args.generate = True

    @patch("training.pipeline.TrainingPipeline.run_training")
    @patch("training.pipeline.TrainingPipeline.setup_wandb")
    @patch("training.pipeline.TrainingPipeline.setup_environment")
    @patch("training.config_loader.ConfigurationLoader")
    def test_execute_success_with_wandb(
        self,
        mock_config_loader_class,
        mock_setup_env,
        mock_setup_wandb,
        mock_run_training,
    ):
        """Test successful pipeline execution with WandB."""
        mock_config_loader = Mock()
        mock_config_loader_class.return_value = mock_config_loader

        # Mock successful execution path
        mock_setup_env.return_value = True
        mock_wandb_run = Mock()
        mock_setup_wandb.return_value = mock_wandb_run
        mock_run_training.return_value = True

        pipeline = TrainingPipeline(self.mock_args)
        result = pipeline.execute()

        self.assertTrue(result)
        mock_setup_env.assert_called_once()
        mock_setup_wandb.assert_called_once()
        mock_run_training.assert_called_once()
        logger.info("Pipeline execution with WandB success verified")

    @patch("training.pipeline.TrainingPipeline.setup_environment")
    @patch("training.config_loader.ConfigurationLoader")
    def test_execute_environment_setup_failure(
        self, mock_config_loader_class, mock_setup_env
    ):
        """Test pipeline execution with environment setup failure."""
        mock_config_loader = Mock()
        mock_config_loader_class.return_value = mock_config_loader

        mock_setup_env.return_value = False  # Environment setup fails

        pipeline = TrainingPipeline(self.mock_args)
        result = pipeline.execute()

        self.assertFalse(result)
        mock_setup_env.assert_called_once()
        logger.info("Pipeline execution environment failure handled correctly")

    @patch("training.pipeline.TrainingPipeline.run_training")
    @patch("training.pipeline.TrainingPipeline.create_trainer")
    @patch("training.pipeline.TrainingPipeline.create_data_loaders")
    @patch("training.pipeline.TrainingPipeline.setup_wandb")
    @patch("training.pipeline.TrainingPipeline.setup_environment")
    @patch("training.config_loader.ConfigurationLoader")
    def test_execute_training_failure(
        self,
        mock_config_loader_class,
        mock_setup_env,
        mock_setup_wandb,
        mock_create_loaders,
        mock_create_trainer,
        mock_run_training,
    ):
        """Test pipeline execution with training failure."""
        mock_config_loader = Mock()
        mock_config_loader_class.return_value = mock_config_loader

        # Mock successful setup but failed training
        mock_setup_env.return_value = True
        mock_setup_wandb.return_value = None  # No WandB
        mock_create_loaders.return_value = (Mock(), Mock())
        mock_create_trainer.return_value = Mock()
        mock_run_training.return_value = False  # Training fails

        pipeline = TrainingPipeline(self.mock_args)
        result = pipeline.execute()

        self.assertFalse(result)
        mock_run_training.assert_called_once()
        logger.info("Pipeline execution training failure handled correctly")

    @patch("training.pipeline.TrainingPipeline.setup_environment")
    @patch("training.config_loader.ConfigurationLoader")
    def test_execute_with_exception(
        self, mock_config_loader_class, mock_setup_env
    ):
        """Test pipeline execution with unexpected exception."""
        mock_config_loader = Mock()
        mock_config_loader_class.return_value = mock_config_loader

        mock_setup_env.side_effect = Exception("Unexpected error")

        pipeline = TrainingPipeline(self.mock_args)
        result = pipeline.execute()

        self.assertFalse(result)
        logger.info("Pipeline execution exception handling verified")

    @patch("core.trainer.PokemonSpriteTrainer")
    @patch("training.config_loader.ConfigurationLoader")
    def test_run_training_success(
        self, mock_config_loader_class, mock_trainer_class
    ):
        """Test successful training execution."""
        mock_config_loader = Mock()
        mock_config_loader.load_and_validate_configs.return_value = (
            Mock(),
            Mock(),
        )
        mock_config_loader.get_data_root_dir.return_value = Path("/tmp")
        mock_config_loader_class.return_value = mock_config_loader

        mock_trainer = Mock()
        mock_trainer.train.return_value = True

        with patch(
            "core.dataset_manager.DatasetManager"
        ) as mock_dataset_manager_class:
            mock_dataset_manager = Mock()
            mock_dataset_manager.validate_and_prepare_data.return_value = True
            mock_dataset_manager_class.return_value = mock_dataset_manager

            with patch("core.transfer_learning.TransferLearningManager"):
                with patch(
                    "training.data_loaders.create_data_loaders"
                ) as mock_create_loaders:
                    mock_create_loaders.return_value = (Mock(), Mock())

                    pipeline = TrainingPipeline(self.mock_args)
                    pipeline.setup_environment()

                    result = pipeline.run_training(mock_trainer, None)

                    self.assertTrue(result)
                    mock_trainer.train.assert_called_once()
                    logger.info("Training execution success verified")

    @patch("core.trainer.PokemonSpriteTrainer")
    @patch("training.config_loader.ConfigurationLoader")
    def test_run_training_with_wandb(
        self, mock_config_loader_class, mock_trainer_class
    ):
        """Test training execution with WandB logging."""
        mock_config_loader = Mock()
        mock_config_loader.load_and_validate_configs.return_value = (
            Mock(),
            Mock(),
        )
        mock_config_loader.get_data_root_dir.return_value = Path("/tmp")
        mock_config_loader_class.return_value = mock_config_loader

        mock_trainer = Mock()
        mock_trainer.train.return_value = True
        mock_wandb_run = Mock()

        with patch(
            "core.dataset_manager.DatasetManager"
        ) as mock_dataset_manager_class:
            mock_dataset_manager = Mock()
            mock_dataset_manager.validate_and_prepare_data.return_value = True
            mock_dataset_manager_class.return_value = mock_dataset_manager

            with patch("core.transfer_learning.TransferLearningManager"):
                with patch(
                    "training.data_loaders.create_data_loaders"
                ) as mock_create_loaders:
                    mock_create_loaders.return_value = (Mock(), Mock())

                    pipeline = TrainingPipeline(self.mock_args)
                    pipeline.setup_environment()

                    result = pipeline.run_training(
                        mock_trainer, mock_wandb_run
                    )

                    self.assertTrue(result)
                    mock_trainer.train.assert_called_once()
                    logger.info("Training execution with WandB verified")

    @patch("training.pipeline.ConfigurationLoader")
    def test_run_training_failure(self, mock_config_loader_class):
        """Test training execution failure."""
        mock_config_loader = Mock()
        mock_config_loader.load_and_validate_configs.return_value = (
            Mock(),
            Mock(),
        )
        mock_config_loader.get_data_root_dir.return_value = Path("/tmp")
        mock_config_loader_class.return_value = mock_config_loader

        mock_trainer = Mock()
        mock_trainer.train.return_value = False

        with patch(
            "training.pipeline.DatasetManager"
        ) as mock_dataset_manager_class:
            mock_dataset_manager = Mock()
            mock_dataset_manager.validate_and_prepare_data.return_value = True
            mock_dataset_manager_class.return_value = mock_dataset_manager

            with patch("core.transfer_learning.TransferLearningManager"):
                with patch(
                    "training.data_loaders.create_data_loaders"
                ) as mock_create_loaders:
                    mock_create_loaders.return_value = (Mock(), Mock())

                    pipeline = TrainingPipeline(self.mock_args)
                    pipeline.setup_environment()

                    result = pipeline.run_training(mock_trainer, None)

                    self.assertFalse(result)
                    mock_trainer.train.assert_called_once()
                    logger.info("Training execution failure handled correctly")

    @patch("core.trainer.PokemonSpriteTrainer")
    @patch("training.config_loader.ConfigurationLoader")
    def test_run_training_exception(
        self, mock_config_loader_class, mock_trainer_class
    ):
        """Test training execution with exception."""
        mock_config_loader = Mock()
        mock_config_loader_class.return_value = mock_config_loader

        mock_trainer = Mock()
        mock_trainer.train.side_effect = Exception("Training error")

        pipeline = TrainingPipeline(self.mock_args)

        result = pipeline.run_training(mock_trainer, None)

        self.assertFalse(result)
        logger.info("Training execution exception handling verified")


class TestPipelineCleanup(unittest.TestCase):
    """Test TrainingPipeline cleanup functionality."""

    def setUp(self):
        """Set up test environment."""
        self.mock_args = Mock()
        self.mock_args.model = "transformer-enhanced"
        self.mock_args.config = "test"

    @patch("training.config_loader.ConfigurationLoader")
    def test_cleanup_success(self, mock_config_loader_class):
        """Test successful pipeline cleanup."""
        mock_config_loader = Mock()
        mock_config_loader_class.return_value = mock_config_loader

        pipeline = TrainingPipeline(self.mock_args)

        # Test cleanup should not raise exceptions
        try:
            pipeline.cleanup()
            cleanup_success = True
        except Exception:
            cleanup_success = False

        self.assertTrue(cleanup_success)
        logger.info("Pipeline cleanup success verified")

    @patch("wandb.finish")
    @patch("training.config_loader.ConfigurationLoader")
    def test_cleanup_with_wandb(
        self, mock_config_loader_class, mock_wandb_finish
    ):
        """Test pipeline cleanup with WandB run."""
        mock_config_loader = Mock()
        mock_config_loader_class.return_value = mock_config_loader

        pipeline = TrainingPipeline(self.mock_args)
        pipeline.wandb_run = Mock()  # Set WandB run

        pipeline.cleanup()

        mock_wandb_finish.assert_called_once()
        logger.info("Pipeline cleanup with WandB verified")

    @patch("wandb.finish")
    @patch("training.config_loader.ConfigurationLoader")
    def test_cleanup_wandb_exception(
        self, mock_config_loader_class, mock_wandb_finish
    ):
        """Test pipeline cleanup with WandB exception."""
        mock_config_loader = Mock()
        mock_config_loader_class.return_value = mock_config_loader

        mock_wandb_finish.side_effect = Exception("WandB cleanup error")

        pipeline = TrainingPipeline(self.mock_args)
        pipeline.wandb_run = Mock()

        # Should handle WandB cleanup exception gracefully
        try:
            pipeline.cleanup()
            cleanup_handled = True
        except Exception:
            cleanup_handled = False

        self.assertTrue(cleanup_handled)
        logger.info("Pipeline cleanup WandB exception handling verified")


if __name__ == "__main__":
    unittest.main(verbosity=2)
