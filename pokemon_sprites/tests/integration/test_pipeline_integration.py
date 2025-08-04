"""
Integration tests for the Pokemon sprite generation pipeline.

This module contains comprehensive integration tests that verify the
interaction between different components of the system, ensuring
that the complete pipeline works correctly end-to-end.
"""

import logging
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from config.settings import (
    TrainingConfig,
    get_data_root_dir,
    get_models_root_dir,
)
from core.logging_config import initialize_project_logging
from data.augmentation import PairedRandomHorizontalFlip
from data.loaders import create_training_dataset, find_valid_pairs
from models import Pix2PixDiscriminator, Pix2PixGenerator

# Import test utilities
from tests import TestDataFactory

# Configure test logging
initialize_project_logging("test_integration")
logger = logging.getLogger(__name__)


class TestColors:
    """ANSI color codes for professional test output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_test_result(test_name: str, success: bool, message: str = ""):
    """Print formatted test result with appropriate colors."""
    if success:
        print(
            f"{TestColors.GREEN}{TestColors.BOLD}[SUCCESS]{TestColors.RESET} {test_name}"
        )
        if message:
            print(f"          {message}")
    else:
        print(
            f"{TestColors.RED}{TestColors.BOLD}[FAIL]{TestColors.RESET} {test_name}"
        )
        if message:
            print(f"       {message}")


class TestDataLoaderModelIntegration(unittest.TestCase):
    """Test integration between data loading and model components."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.sprites_dir = self.test_dir / "sprites"
        self.artwork_dir = self.test_dir / "artwork"
        self.output_dir = self.test_dir / "output"

        self.sprites_dir.mkdir()
        self.artwork_dir.mkdir()
        self.output_dir.mkdir()

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def _create_test_image_pair(self, pokemon_id: str):
        """Helper method to create a test image pair using shared utilities."""
        sprite_path = self.sprites_dir / f"pokemon_{pokemon_id}.png"
        artwork_path = self.artwork_dir / f"pokemon_{pokemon_id}_artwork.png"

        # Create test images using shared utility
        sprite_img = TestDataFactory.create_test_image((256, 256), "red")
        artwork_img = TestDataFactory.create_test_image((256, 256), "blue")

        sprite_img.save(sprite_path)
        artwork_img.save(artwork_path)

        return sprite_path, artwork_path

    def test_data_loading_to_model_pipeline(self):
        """Test complete pipeline from data loading to model inference."""
        # Create test data
        test_ids = ["0001", "0002", "0003"]
        for pokemon_id in test_ids:
            self._create_test_image_pair(pokemon_id)

        # Step 1: Find valid pairs
        pairs = find_valid_pairs(self.sprites_dir, self.artwork_dir)
        self.assertEqual(len(pairs), 3)

        # Step 2: Create training dataset
        dataset_info = create_training_dataset(
            pairs, self.output_dir, train_split=0.8, image_size=(64, 64)
        )

        self.assertIn("train_pairs", dataset_info)
        self.assertIn("val_pairs", dataset_info)

        # Step 3: Load data for model
        train_input_dir = self.output_dir / "train" / "input"
        self.assertTrue(train_input_dir.exists())

        input_files = list(train_input_dir.glob("*.png"))
        self.assertGreater(len(input_files), 0)

        # Step 4: Test model with loaded data
        model = Pix2PixGenerator(
            input_channels=4, output_channels=4, ngf=32
        )  # Updated for ARGB support
        model.eval()

        # Load and process a test image
        test_image = Image.open(input_files[0])
        test_tensor = (
            torch.from_numpy(np.array(test_image)).permute(2, 0, 1).float()
            / 255.0
        )
        test_tensor = test_tensor.unsqueeze(0)  # Add batch dimension

        # Test model inference
        with torch.no_grad():
            output = model(test_tensor)

        self.assertEqual(output.shape, test_tensor.shape)
        self.assertFalse(torch.isnan(output).any())

        print_test_result(
            "test_data_loading_to_model_pipeline",
            True,
            f"Pipeline: {len(pairs)} pairs -> dataset -> model inference",
        )

    def test_augmentation_model_integration(self):
        """Test integration between data augmentation and model training."""
        # Create test data
        self._create_test_image_pair("0001")

        # Load test images
        sprite_img = Image.open(self.sprites_dir / "pokemon_0001.png")
        artwork_img = Image.open(self.artwork_dir / "pokemon_0001_artwork.png")

        # Apply augmentation
        augmentation = PairedRandomHorizontalFlip(p=1.0)  # Always flip
        aug_artwork, aug_sprite = augmentation(artwork_img, sprite_img)

        # Convert to tensors
        def img_to_tensor(img):
            return (
                torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
                / 255.0
            )

        aug_artwork_tensor = img_to_tensor(aug_artwork).unsqueeze(0)
        aug_sprite_tensor = img_to_tensor(aug_sprite).unsqueeze(0)

        # Test model with augmented data
        generator = Pix2PixGenerator(
            input_channels=3, output_channels=3, ngf=32
        )
        discriminator = Pix2PixDiscriminator(
            input_channels=6, ndf=32, n_layers=2
        )

        # Generator forward pass
        with torch.no_grad():
            fake_sprite = generator(aug_artwork_tensor)

        # Discriminator forward pass
        with torch.no_grad():
            real_output = discriminator(aug_artwork_tensor, aug_sprite_tensor)
            fake_output = discriminator(aug_artwork_tensor, fake_sprite)

        # Verify outputs
        self.assertEqual(fake_sprite.shape, aug_sprite_tensor.shape)
        self.assertFalse(torch.isnan(fake_sprite).any())
        self.assertFalse(torch.isnan(real_output).any())
        self.assertFalse(torch.isnan(fake_output).any())

        print_test_result(
            "test_augmentation_model_integration",
            True,
            "Augmentation -> Generator -> Discriminator pipeline working",
        )

    def test_config_driven_pipeline(self):
        """Test configuration-driven pipeline setup."""
        # Create test configuration
        config = TrainingConfig(
            batch_size=2, epochs=5, learning_rate=0.0002, device="cpu"
        )

        # Create test data
        for i in range(config.batch_size + 1):  # Create more than batch size
            self._create_test_image_pair(f"{i+1:04d}")

        # Setup pipeline based on config
        pairs = find_valid_pairs(self.sprites_dir, self.artwork_dir)
        dataset_info = create_training_dataset(
            pairs, self.output_dir, image_size=(64, 64)
        )

        # Initialize models based on config
        generator = Pix2PixGenerator(
            input_channels=3, output_channels=3, ngf=32
        )
        discriminator = Pix2PixDiscriminator(
            input_channels=6, ndf=32, n_layers=2
        )

        # Setup optimizers based on config
        gen_optimizer = torch.optim.Adam(
            generator.parameters(), lr=config.learning_rate, betas=(0.5, 0.999)
        )
        disc_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=config.learning_rate,
            betas=(0.5, 0.999),
        )

        # Verify setup
        self.assertEqual(
            gen_optimizer.param_groups[0]["lr"], config.learning_rate
        )
        self.assertEqual(
            disc_optimizer.param_groups[0]["lr"], config.learning_rate
        )
        self.assertGreater(len(pairs), 0)
        self.assertIn("train_pairs", dataset_info)

        print_test_result(
            "test_config_driven_pipeline",
            True,
            f"Config-driven setup: {len(pairs)} pairs, lr={config.learning_rate}",
        )

    def test_batch_processing_integration(self):
        """Test batch processing through the entire pipeline."""
        # Create test data
        batch_size = 4
        for i in range(batch_size * 2):  # Create enough for multiple batches
            self._create_test_image_pair(f"{i+1:04d}")

        # Load and process data
        pairs = find_valid_pairs(self.sprites_dir, self.artwork_dir)
        create_training_dataset(pairs, self.output_dir, image_size=(64, 64))

        # Simulate batch loading
        train_input_dir = self.output_dir / "train" / "input"
        train_target_dir = self.output_dir / "train" / "target"

        input_files = sorted(list(train_input_dir.glob("*.png")))
        target_files = sorted(list(train_target_dir.glob("*.png")))

        # Create batches
        input_batch = []
        target_batch = []

        for i in range(min(batch_size, len(input_files))):
            input_img = Image.open(input_files[i])
            target_img = Image.open(target_files[i])

            input_tensor = (
                torch.from_numpy(np.array(input_img)).permute(2, 0, 1).float()
                / 255.0
            )
            target_tensor = (
                torch.from_numpy(np.array(target_img)).permute(2, 0, 1).float()
                / 255.0
            )

            input_batch.append(input_tensor)
            target_batch.append(target_tensor)

        input_batch = torch.stack(input_batch)
        target_batch = torch.stack(target_batch)

        # Test models with batch
        generator = Pix2PixGenerator(
            input_channels=4, output_channels=4, ngf=32
        )
        discriminator = Pix2PixDiscriminator(
            input_channels=8, ndf=32, n_layers=2  # 4 input + 4 target channels
        )

        with torch.no_grad():
            fake_batch = generator(input_batch)

            real_scores = discriminator(input_batch, target_batch)
            fake_scores = discriminator(input_batch, fake_batch)

        # Verify batch processing
        self.assertEqual(fake_batch.shape[0], input_batch.shape[0])
        self.assertEqual(real_scores.shape[0], input_batch.shape[0])
        self.assertEqual(fake_scores.shape[0], input_batch.shape[0])

        print_test_result(
            "test_batch_processing_integration",
            True,
            f"Batch processing: {input_batch.shape[0]} samples",
        )


class TestTrainingPipelineIntegration(unittest.TestCase):
    """Test integration of complete training pipeline components."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.device = torch.device("cpu")  # Use CPU for testing

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_loss_computation_integration(self):
        """Test loss computation with generated and real data."""
        # Create models
        generator = Pix2PixGenerator(
            input_channels=3, output_channels=3, ngf=32
        )
        discriminator = Pix2PixDiscriminator(
            input_channels=6, ndf=32, n_layers=2
        )

        # Create synthetic batch
        batch_size = 2
        input_images = torch.randn(batch_size, 3, 64, 64)
        target_images = torch.randn(batch_size, 3, 64, 64)

        # Generator forward pass
        fake_images = generator(input_images)

        # Discriminator forward pass
        real_scores = discriminator(input_images, target_images)
        fake_scores = discriminator(input_images, fake_images)

        # Compute losses
        criterion = nn.BCEWithLogitsLoss()
        l1_criterion = nn.L1Loss()

        # Discriminator loss
        real_labels = torch.ones_like(real_scores)
        fake_labels = torch.zeros_like(fake_scores)

        disc_real_loss = criterion(real_scores, real_labels)
        disc_fake_loss = criterion(fake_scores, fake_labels)
        disc_loss = (disc_real_loss + disc_fake_loss) * 0.5

        # Generator loss
        gen_adv_loss = criterion(fake_scores, torch.ones_like(fake_scores))
        gen_l1_loss = l1_criterion(fake_images, target_images)
        gen_loss = gen_adv_loss + 100 * gen_l1_loss  # L1 weight

        # Verify losses
        self.assertFalse(torch.isnan(disc_loss))
        self.assertFalse(torch.isnan(gen_loss))
        self.assertGreater(disc_loss.item(), 0)
        self.assertGreater(gen_loss.item(), 0)

        print_test_result(
            "test_loss_computation_integration",
            True,
            f"Losses computed: D={disc_loss.item():.4f}, G={gen_loss.item():.4f}",
        )

    def test_optimizer_integration(self):
        """Test optimizer integration with model training."""
        # Create models
        generator = Pix2PixGenerator(
            input_channels=3, output_channels=3, ngf=32
        )
        discriminator = Pix2PixDiscriminator(
            input_channels=6, ndf=32, n_layers=2
        )

        # Create optimizers
        gen_optimizer = torch.optim.Adam(
            generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        disc_optimizer = torch.optim.Adam(
            discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )

        # Create synthetic data
        input_images = torch.randn(1, 3, 64, 64)
        target_images = torch.randn(1, 3, 64, 64)

        # Training step simulation
        generator.train()
        discriminator.train()

        # Get initial parameter values
        initial_gen_param = next(generator.parameters()).clone()
        initial_disc_param = next(discriminator.parameters()).clone()

        # Forward pass
        fake_images = generator(input_images)

        # Discriminator training
        disc_optimizer.zero_grad()

        # Fixed discriminator call
        # Fixed discriminator call

        real_scores = discriminator(input_images, target_images)
        fake_scores = discriminator(
            input_images, fake_images.detach()
        )  # Detach to avoid graph issues

        disc_loss = (
            nn.BCEWithLogitsLoss()(real_scores, torch.ones_like(real_scores))
            + nn.BCEWithLogitsLoss()(
                fake_scores, torch.zeros_like(fake_scores)
            )
        ) * 0.5

        disc_loss.backward()
        disc_optimizer.step()

        # Generator training
        gen_optimizer.zero_grad()

        # Fixed discriminator call
        fake_scores = discriminator(input_images, fake_images)

        gen_loss = nn.BCEWithLogitsLoss()(
            fake_scores, torch.ones_like(fake_scores)
        ) + 100 * nn.L1Loss()(fake_images, target_images)

        gen_loss.backward()
        gen_optimizer.step()

        # Verify parameters updated
        final_gen_param = next(generator.parameters())
        final_disc_param = next(discriminator.parameters())

        self.assertFalse(torch.equal(initial_gen_param, final_gen_param))
        self.assertFalse(torch.equal(initial_disc_param, final_disc_param))

        print_test_result(
            "test_optimizer_integration",
            True,
            "Parameters updated after training step",
        )

    def test_model_state_management(self):
        """Test model state saving and loading integration."""
        # Create models
        generator = Pix2PixGenerator(
            input_channels=3, output_channels=3, ngf=32
        )
        discriminator = Pix2PixDiscriminator(
            input_channels=6, ndf=32, n_layers=2
        )

        # Get initial states
        generator.state_dict()
        discriminator.state_dict()

        # Modify models (simulate training)
        input_tensor = torch.randn(1, 3, 64, 64)
        target_tensor = torch.randn(1, 3, 64, 64)

        fake_output = generator(input_tensor)
        loss = nn.L1Loss()(fake_output, target_tensor)
        loss.backward()

        # Save states
        gen_checkpoint_path = self.test_dir / "generator.pth"
        disc_checkpoint_path = self.test_dir / "discriminator.pth"

        torch.save(generator.state_dict(), gen_checkpoint_path)
        torch.save(discriminator.state_dict(), disc_checkpoint_path)

        # Create new models and load states
        new_generator = Pix2PixGenerator(
            input_channels=3, output_channels=3, ngf=32
        )
        new_discriminator = Pix2PixDiscriminator(
            input_channels=6, ndf=32, n_layers=2
        )

        new_generator.load_state_dict(torch.load(gen_checkpoint_path))
        new_discriminator.load_state_dict(torch.load(disc_checkpoint_path))

        # Verify loaded states match
        loaded_gen_state = new_generator.state_dict()
        loaded_disc_state = new_discriminator.state_dict()

        for key in generator.state_dict():
            self.assertTrue(
                torch.equal(generator.state_dict()[key], loaded_gen_state[key])
            )

        for key in discriminator.state_dict():
            self.assertTrue(
                torch.equal(
                    discriminator.state_dict()[key], loaded_disc_state[key]
                )
            )

        print_test_result(
            "test_model_state_management",
            True,
            "Model states saved and loaded correctly",
        )

    def test_training_loop_integration(self):
        """Test simplified training loop integration."""
        # Create models
        generator = Pix2PixGenerator(
            input_channels=3, output_channels=3, ngf=16
        )  # Smaller for speed
        discriminator = Pix2PixDiscriminator(
            input_channels=6, ndf=16, n_layers=2
        )

        # Create optimizers
        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

        # Create synthetic dataset
        num_samples = 4
        dataset = []
        for _ in range(num_samples):
            input_img = torch.randn(3, 64, 64)
            target_img = torch.randn(3, 64, 64)
            dataset.append((input_img, target_img))

        # Training loop (simplified)
        generator.train()
        discriminator.train()

        num_epochs = 2
        losses = {"gen": [], "disc": []}

        for epoch in range(num_epochs):
            epoch_gen_loss = 0
            epoch_disc_loss = 0

            for input_img, target_img in dataset:
                input_batch = input_img.unsqueeze(0)
                target_batch = target_img.unsqueeze(0)

                # Train discriminator
                disc_optimizer.zero_grad()

                fake_batch = generator(input_batch)

                # Fixed discriminator call
                # Fixed discriminator call

                real_scores = discriminator(input_batch, target_batch)
                fake_scores = discriminator(
                    input_batch, fake_batch.detach()
                )  # Detach to avoid graph issues

                disc_loss = (
                    nn.BCEWithLogitsLoss()(
                        real_scores, torch.ones_like(real_scores)
                    )
                    + nn.BCEWithLogitsLoss()(
                        fake_scores, torch.zeros_like(fake_scores)
                    )
                ) * 0.5

                disc_loss.backward()
                disc_optimizer.step()

                # Train generator
                gen_optimizer.zero_grad()

                # Fixed discriminator call
                fake_scores = discriminator(input_batch, fake_batch)

                gen_loss = nn.BCEWithLogitsLoss()(
                    fake_scores, torch.ones_like(fake_scores)
                ) + 10 * nn.L1Loss()(fake_batch, target_batch)

                gen_loss.backward()
                gen_optimizer.step()

                epoch_gen_loss += gen_loss.item()
                epoch_disc_loss += disc_loss.item()

            losses["gen"].append(epoch_gen_loss / num_samples)
            losses["disc"].append(epoch_disc_loss / num_samples)

        # Verify training progressed
        self.assertEqual(len(losses["gen"]), num_epochs)
        self.assertEqual(len(losses["disc"]), num_epochs)

        # Losses should be finite
        for loss_val in losses["gen"] + losses["disc"]:
            self.assertFalse(np.isnan(loss_val))
            self.assertFalse(np.isinf(loss_val))

        print_test_result(
            "test_training_loop_integration",
            True,
            f"Training completed: {num_epochs} epochs, "
            f"final G loss: {losses['gen'][-1]:.4f}, "
            f"final D loss: {losses['disc'][-1]:.4f}",
        )


class TestDirectoryStructureIntegration(unittest.TestCase):
    """Test integration with directory structures and file management."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_directory_structure_setup(self):
        """Test complete directory structure setup."""
        # Test models directory
        models_dir = get_models_root_dir()
        self.assertTrue(Path(models_dir).exists())

        # Test data directory - create if it doesn't exist for CI environments
        data_dir = get_data_root_dir()
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        self.assertTrue(Path(data_dir).exists())

        # Create project structure
        project_dirs = [
            "data/train/input",
            "data/train/target",
            "data/val/input",
            "data/val/target",
            "models/checkpoints",
            "logs",
            "results",
        ]

        for dir_path in project_dirs:
            full_path = self.test_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            self.assertTrue(full_path.exists())

        print_test_result(
            "test_directory_structure_setup",
            True,
            f"Created {len(project_dirs)} project directories",
        )

    def test_checkpoint_management_integration(self):
        """Test checkpoint saving and directory management."""
        # Create checkpoint directory
        checkpoint_dir = self.test_dir / "checkpoints"
        checkpoint_dir.mkdir()

        # Create model
        model = Pix2PixGenerator(input_channels=3, output_channels=3, ngf=32)

        # Save checkpoint with metadata
        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "epoch": 10,
            "loss": 0.5,
            "optimizer_state_dict": torch.optim.Adam(
                model.parameters()
            ).state_dict(),
            "config": {"batch_size": 16, "learning_rate": 0.0002},
        }

        checkpoint_path = checkpoint_dir / "model_epoch_10.pth"
        torch.save(checkpoint_data, checkpoint_path)

        # Verify checkpoint
        self.assertTrue(checkpoint_path.exists())

        # Load and verify checkpoint
        loaded_checkpoint = torch.load(checkpoint_path)

        self.assertIn("model_state_dict", loaded_checkpoint)
        self.assertIn("epoch", loaded_checkpoint)
        self.assertIn("config", loaded_checkpoint)
        self.assertEqual(loaded_checkpoint["epoch"], 10)

        print_test_result(
            "test_checkpoint_management_integration",
            True,
            f"Checkpoint saved and verified: {checkpoint_path.name}",
        )

    def test_logging_integration(self):
        """Test logging integration with file management."""
        # Create logs directory
        logs_dir = self.test_dir / "logs"
        logs_dir.mkdir()

        # Test log file creation
        log_file = logs_dir / "training.log"

        # Create logger
        test_logger = logging.getLogger("test_integration")
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.INFO)

        # Write test logs
        test_messages = [
            "Training started",
            "Epoch 1 completed",
            "Model checkpoint saved",
            "Training completed",
        ]

        for message in test_messages:
            test_logger.info(message)

        # Verify log file
        self.assertTrue(log_file.exists())

        with open(log_file, "r") as f:
            log_content = f.read()

        for message in test_messages:
            self.assertIn(message, log_content)

        print_test_result(
            "test_logging_integration",
            True,
            f"Log file created with {len(test_messages)} entries",
        )


if __name__ == "__main__":
    print(f"\n{TestColors.BLUE}{TestColors.BOLD}{'='*70}{TestColors.RESET}")
    print(
        f"{TestColors.BLUE}{TestColors.BOLD}Running Integration Tests for Pokemon Sprite Pipeline{TestColors.RESET}"
    )
    print(f"{TestColors.BLUE}{TestColors.BOLD}{'='*70}{TestColors.RESET}\n")

    # Run tests with detailed output
    unittest.main(verbosity=2, exit=False)

    print(f"\n{TestColors.BLUE}{TestColors.BOLD}{'='*70}{TestColors.RESET}")
    print(
        f"{TestColors.BLUE}{TestColors.BOLD}Integration Tests Completed{TestColors.RESET}"
    )
    print(f"{TestColors.BLUE}{TestColors.BOLD}{'='*70}{TestColors.RESET}")
