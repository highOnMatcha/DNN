"""
Advanced unit tests for train.py module focusing on main training pipeline functionality.

This module tests the comprehensive training pipeline including data loading,
model initialization, training workflows, and command-line interface components.
"""

import argparse
import logging
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import torch
from PIL import Image

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from train import (
    setup_wandb,
    load_and_validate_configs,
    create_data_loaders,
    _setup_argument_parser,
    _initialize_experiment_logging,
    _setup_wandb_tracking,
    _create_trainer,
    _determine_max_generation_count,
    find_missing_sprites,
    generate_missing_sprites,
    PokemonDataset,
    TransferLearningManager,
    DataEnhancer,
    CurriculumTrainingManager,
)

logger = logging.getLogger(__name__)


def print_test_result(test_name: str, passed: bool, message: str = ""):
    """Print colored test results."""
    status = "✓ PASS" if passed else "✗ FAIL"
    color = "\033[92m" if passed else "\033[91m"
    print(f"{color}{status}\033[0m {test_name}: {message}")


class TestTrainingPipelineComponents(unittest.TestCase):
    """Test core training pipeline components."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_argument_parser_setup(self):
        """Test command-line argument parser setup."""
        parser = _setup_argument_parser()
        
        # Verify parser is created
        self.assertIsInstance(parser, argparse.ArgumentParser)
        
        # Test parsing valid arguments
        test_args = ["--model", "test_model", "--config", "development"]
        args = parser.parse_args(test_args)
        
        self.assertEqual(args.model, "test_model")
        self.assertEqual(args.config, "development")
        
        print_test_result("test_argument_parser_setup", True,
                         "Argument parser created and tested")

    @patch('train.initialize_project_logging')
    @patch('train.log_system_info')
    def test_experiment_logging_initialization(self, mock_log_sys, mock_init_log):
        """Test experiment logging initialization."""
        # Create mock args
        mock_args = Mock()
        mock_args.model = "test_model"
        mock_args.config = "development"
        mock_args.log_level = "INFO"
        
        experiment_id = _initialize_experiment_logging(mock_args)
        
        # Verify experiment ID is generated
        self.assertIsInstance(experiment_id, str)
        self.assertIn("pokemon_sprites", experiment_id)
        
        # Verify logging functions were called
        self.assertTrue(mock_init_log.called)
        self.assertTrue(mock_log_sys.called)
        
        print_test_result("test_experiment_logging_initialization", True,
                         f"Experiment ID: {experiment_id[:20]}...")

    @patch('train.setup_wandb')
    def test_wandb_tracking_setup(self, mock_setup_wandb):
        """Test WandB tracking setup."""
        # Mock configurations
        mock_training_config = Mock()
        mock_training_config.wandb_log = True
        mock_training_config.wandb_project = "test_project"
        
        mock_model_config = Mock()
        mock_model_config.name = "test_model"
        mock_model_config.architecture = "unet"
        
        mock_args = Mock()
        mock_args.no_wandb = False
        mock_args.model = "test_model"
        mock_args.config = "development"
        
        # Mock return value
        mock_setup_wandb.return_value = Mock()
        
        wandb_run = _setup_wandb_tracking(mock_args, mock_training_config, mock_model_config)
        
        # Verify WandB setup was called
        self.assertTrue(mock_setup_wandb.called)
        
        print_test_result("test_wandb_tracking_setup", True,
                         "WandB tracking setup tested")

    def test_max_generation_count_determination(self):
        """Test maximum generation count logic."""
        # Mock args for different configurations
        mock_args_test = Mock()
        mock_args_test.max_generate = None
        mock_args_test.config = "test"
        
        mock_args_dev = Mock()
        mock_args_dev.max_generate = None
        mock_args_dev.config = "development"
        
        mock_args_custom = Mock()
        mock_args_custom.max_generate = 25
        mock_args_custom.config = "production"
        
        # Mock missing artwork list
        missing_artwork = ["artwork1", "artwork2", "artwork3"]
        
        # Test different scenarios
        count_test = _determine_max_generation_count(mock_args_test, missing_artwork)
        count_dev = _determine_max_generation_count(mock_args_dev, missing_artwork)
        count_custom = _determine_max_generation_count(mock_args_custom, missing_artwork)
        count_prod = _determine_max_generation_count(Mock(max_generate=None, config="production"), missing_artwork)
        
        self.assertEqual(count_test, 10)  # test config limit
        self.assertEqual(count_dev, 50)   # development config limit
        self.assertEqual(count_custom, 25)  # custom limit
        self.assertEqual(count_prod, 3)   # production = all missing
        
        print_test_result("test_max_generation_count_determination", True,
                         f"Test: {count_test}, Dev: {count_dev}, Custom: {count_custom}")


class TestWandBIntegration(unittest.TestCase):
    """Test WandB integration functionality."""

    @patch.dict('os.environ', {'WANDB_API_KEY': 'test_key'})
    @patch('train.wandb.login')
    @patch('train.wandb.init')
    def test_wandb_setup_success(self, mock_init, mock_login):
        """Test successful WandB setup."""
        # Mock successful login and init
        mock_login.return_value = True
        mock_run = Mock()
        mock_init.return_value = mock_run
        
        result = setup_wandb(
            project_name="test_project",
            model_name="test_model",
            config_type="test"
        )
        
        # Verify WandB functions were called
        self.assertTrue(mock_login.called)
        self.assertTrue(mock_init.called)
        self.assertEqual(result, mock_run)
        
        print_test_result("test_wandb_setup_success", True,
                         "WandB setup successful path tested")

    @patch.dict('os.environ', {}, clear=True)
    @patch('train.wandb.init')
    @patch('train.wandb.login')
    @patch('train.logger')
    def test_wandb_setup_no_api_key(self, mock_logger, mock_login, mock_init):
        """Test WandB setup without API key."""
        # Mock WandB to avoid actual initialization and Sentry scope issues
        mock_login.return_value = False  # Simulate login failure
        mock_init.return_value = None    # Return None on init failure
        
        result = setup_wandb(
            project_name="test_project",
            model_name="test_model",
            config_type="test"
        )
        
        # Should return None when login fails or no API key
        self.assertIsNone(result)
        
        print_test_result("test_wandb_setup_no_api_key", True,
                         "WandB setup handles missing API key scenario")

    @patch.dict('os.environ', {'WANDB_API_KEY': 'test_key'})
    @patch('train.wandb.login')
    def test_wandb_setup_login_failure(self, mock_login):
        """Test WandB setup with login failure."""
        # Mock login failure
        mock_login.side_effect = Exception("Login failed")
        
        result = setup_wandb(
            project_name="test_project",
            model_name="test_model",
            config_type="test"
        )
        
        # Should return None on failure
        self.assertIsNone(result)
        
        print_test_result("test_wandb_setup_login_failure", True,
                         "WandB setup handles login failure")


class TestPokemonDataset(unittest.TestCase):
    """Test PokemonDataset class functionality."""

    def setUp(self):
        """Set up test dataset structure."""
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create test data structure
        self.train_input_dir = self.test_dir / "train" / "input"
        self.train_target_dir = self.test_dir / "train" / "target"
        self.val_input_dir = self.test_dir / "val" / "input"
        self.val_target_dir = self.test_dir / "val" / "target"
        
        for dir_path in [self.train_input_dir, self.train_target_dir, 
                        self.val_input_dir, self.val_target_dir]:
            dir_path.mkdir(parents=True)
        
        # Create test images
        for i in range(3):
            # Training images
            input_img = Image.new("RGB", (64, 64), (255, 0, 0))
            target_img = Image.new("RGB", (64, 64), (0, 255, 0))
            input_img.save(self.train_input_dir / f"pokemon_{i:03d}.png")
            target_img.save(self.train_target_dir / f"pokemon_{i:03d}.png")
            
            # Validation images
            input_img.save(self.val_input_dir / f"pokemon_{i:03d}.png")
            target_img.save(self.val_target_dir / f"pokemon_{i:03d}.png")

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_pokemon_dataset_initialization(self):
        """Test PokemonDataset initialization."""
        dataset = PokemonDataset(
            data_dir=str(self.test_dir),
            split="train",
            image_size=64,
            augmentation_level="none"
        )
        
        # Verify dataset properties
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.split, "train")
        self.assertEqual(dataset.image_size, 64)
        
        print_test_result("test_pokemon_dataset_initialization", True,
                         f"Dataset initialized with {len(dataset)} samples")

    def test_pokemon_dataset_getitem(self):
        """Test PokemonDataset __getitem__ functionality."""
        dataset = PokemonDataset(
            data_dir=str(self.test_dir),
            split="train",
            image_size=64,
            augmentation_level="none"
        )
        
        # Test getting an item
        input_tensor, target_tensor = dataset[0]
        
        # Verify tensor properties
        self.assertIsInstance(input_tensor, torch.Tensor)
        self.assertIsInstance(target_tensor, torch.Tensor)
        self.assertEqual(input_tensor.shape, (3, 64, 64))
        self.assertEqual(target_tensor.shape, (3, 64, 64))
        
        print_test_result("test_pokemon_dataset_getitem", True,
                         f"Item shape: {input_tensor.shape}")

    def test_pokemon_dataset_raw_sample(self):
        """Test PokemonDataset raw sample access."""
        dataset = PokemonDataset(
            data_dir=str(self.test_dir),
            split="train",
            image_size=64,
            augmentation_level="none"
        )
        
        # Test getting raw samples
        input_img, target_img = dataset.get_raw_sample(0)
        
        # Verify PIL Images
        self.assertIsInstance(input_img, Image.Image)
        self.assertIsInstance(target_img, Image.Image)
        
        print_test_result("test_pokemon_dataset_raw_sample", True,
                         f"Raw sample size: {input_img.size}")

    def test_pokemon_dataset_validation_split(self):
        """Test PokemonDataset validation split."""
        val_dataset = PokemonDataset(
            data_dir=str(self.test_dir),
            split="val",
            image_size=32,
            augmentation_level="none"
        )
        
        self.assertEqual(val_dataset.split, "val")
        self.assertEqual(len(val_dataset), 3)
        
        print_test_result("test_pokemon_dataset_validation_split", True,
                         f"Validation dataset: {len(val_dataset)} samples")


class TestDataEnhancer(unittest.TestCase):
    """Test DataEnhancer functionality."""

    def setUp(self):
        """Set up test environment for data enhancement."""
        self.enhancer = DataEnhancer(target_size=128)

    def test_data_enhancer_initialization(self):
        """Test DataEnhancer initialization."""
        self.assertEqual(self.enhancer.target_size, 128)
        print_test_result("test_data_enhancer_initialization", True,
                         f"Target size: {self.enhancer.target_size}")

    def test_extract_dominant_colors(self):
        """Test dominant color extraction."""
        # Create test image array
        test_image = torch.randint(0, 255, (32, 32, 3), dtype=torch.uint8).numpy()
        
        colors = self.enhancer._extract_dominant_colors(test_image, n_colors=5)
        
        # Verify colors are extracted
        self.assertIsInstance(colors, list)
        self.assertLessEqual(len(colors), 5)
        
        # Verify colors are RGB tuples
        for color in colors:
            self.assertIsInstance(color, tuple)
            self.assertEqual(len(color), 3)
        
        print_test_result("test_extract_dominant_colors", True,
                         f"Extracted {len(colors)} colors")

    def test_color_palette_swap(self):
        """Test color palette swapping."""
        # Create test images - simple single color images to test the fix
        source_img = Image.new("RGB", (64, 64), (255, 0, 0))
        target_img = Image.new("RGB", (64, 64), (0, 255, 0))
        
        # Test color palette swap - this should no longer generate clustering warnings
        result_img = self.enhancer.color_palette_swap(source_img, target_img)
        
        # Verify result is PIL Image
        self.assertIsInstance(result_img, Image.Image)
        self.assertEqual(result_img.size, (64, 64))
        
        print_test_result("test_color_palette_swap", True,
                         f"Color swap result size: {result_img.size}")

    def test_remap_colors(self):
        """Test color remapping functionality."""
        # Create test image array
        test_image = torch.randint(0, 255, (32, 32, 3), dtype=torch.uint8).numpy()
        target_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        
        # Test color remapping
        result = self.enhancer._remap_colors(test_image, target_colors)
        
        # Verify result shape matches input
        self.assertEqual(result.shape, test_image.shape)
        self.assertIsInstance(result, type(test_image))
        
        print_test_result("test_remap_colors", True,
                         f"Remapped image shape: {result.shape}")


class TestTransferLearningManager(unittest.TestCase):
    """Test TransferLearningManager functionality."""

    def setUp(self):
        """Set up transfer learning test environment."""
        self.device = torch.device("cpu")
        self.manager = TransferLearningManager(self.device)
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_transfer_learning_manager_initialization(self):
        """Test TransferLearningManager initialization."""
        self.assertEqual(self.manager.device, self.device)
        self.assertIsInstance(self.manager.pretrained_urls, dict)
        self.assertGreater(len(self.manager.pretrained_urls), 0)
        
        print_test_result("test_transfer_learning_manager_initialization", True,
                         f"Available pretrained models: {len(self.manager.pretrained_urls)}")

    def test_load_pretrained_weights_nonexistent(self):
        """Test loading weights from nonexistent file."""
        # Create simple model
        model = torch.nn.Linear(10, 5)
        
        # Test with nonexistent file
        nonexistent_path = self.test_dir / "nonexistent.pth"
        result = self.manager.load_pretrained_weights(model, nonexistent_path)
        
        # Should return False for nonexistent file
        self.assertFalse(result)
        
        print_test_result("test_load_pretrained_weights_nonexistent", True,
                         "Handled nonexistent pretrained weights file")

    def test_load_pretrained_weights_valid(self):
        """Test loading weights from valid file."""
        # Create simple model
        model = torch.nn.Linear(10, 5)
        
        # Create test checkpoint
        checkpoint_path = self.test_dir / "test_checkpoint.pth"
        test_state_dict = {
            "weight": torch.randn(5, 10),
            "bias": torch.randn(5)
        }
        torch.save({"generator": test_state_dict}, checkpoint_path)
        
        # Test loading
        result = self.manager.load_pretrained_weights(model, checkpoint_path)
        
        # Should return True for successful loading
        self.assertTrue(result)
        
        print_test_result("test_load_pretrained_weights_valid", True,
                         "Successfully loaded compatible pretrained weights")


class TestCurriculumTrainingManager(unittest.TestCase):
    """Test CurriculumTrainingManager functionality."""

    def setUp(self):
        """Set up curriculum training test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = {"augmentation": "standard"}
        self.manager = CurriculumTrainingManager(self.config, self.test_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_curriculum_manager_initialization(self):
        """Test CurriculumTrainingManager initialization."""
        self.assertEqual(self.manager.config, self.config)
        self.assertEqual(self.manager.data_dir, self.test_dir)
        self.assertEqual(self.manager.input_scales, [128, 192, 256])
        self.assertEqual(self.manager.current_scale_idx, 0)
        
        print_test_result("test_curriculum_manager_initialization", True,
                         f"Scales: {self.manager.input_scales}")

    def test_curriculum_advancement(self):
        """Test curriculum advancement logic."""
        # Initially at first scale
        initial_scale = self.manager.input_scales[self.manager.current_scale_idx]
        self.assertEqual(initial_scale, 128)
        
        # Advance curriculum
        can_advance_1 = self.manager.advance_curriculum()
        self.assertTrue(can_advance_1)
        self.assertEqual(self.manager.current_scale_idx, 1)
        
        # Advance again
        can_advance_2 = self.manager.advance_curriculum()
        self.assertTrue(can_advance_2)
        self.assertEqual(self.manager.current_scale_idx, 2)
        
        # Try to advance beyond last scale
        can_advance_3 = self.manager.advance_curriculum()
        self.assertFalse(can_advance_3)
        self.assertEqual(self.manager.current_scale_idx, 2)  # Should stay at last
        
        print_test_result("test_curriculum_advancement", True,
                         f"Advanced through {len(self.manager.input_scales)} scales")


class TestMissingSpriteFunctionality(unittest.TestCase):
    """Test missing sprite detection and generation functionality."""

    def setUp(self):
        """Set up test environment for sprite functionality."""
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create test artwork directory
        self.artwork_dir = self.test_dir / "artwork"
        self.artwork_dir.mkdir()
        
        # Create test sprite directory
        self.sprite_dir = self.test_dir / "sprites"
        self.sprite_dir.mkdir()
        
        # Create test artwork files
        for i in range(1, 6):  # 5 artwork files
            artwork_img = Image.new("RGB", (256, 256), (255, 0, 0))
            artwork_img.save(self.artwork_dir / f"pokemon_{i:03d}_artwork.png")
        
        # Create sprite files for only some Pokemon (1, 3, 5)
        for i in [1, 3, 5]:
            sprite_img = Image.new("RGB", (64, 64), (0, 255, 0))
            sprite_img.save(self.sprite_dir / f"pokemon_{i:03d}_bw.png")

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_find_missing_sprites(self):
        """Test finding Pokemon artwork without corresponding sprites."""
        missing_sprites = find_missing_sprites(self.artwork_dir, self.sprite_dir)
        
        # Should find artwork files 002 and 004 as missing sprites
        self.assertIsInstance(missing_sprites, list)
        self.assertEqual(len(missing_sprites), 2)  # Pokemon 2 and 4 should be missing
        
        # Verify missing files are correct
        missing_ids = []
        for missing_file in missing_sprites:
            # Extract ID from filename
            if "_artwork" in missing_file.stem:
                pokemon_id = missing_file.stem.split("_")[1]
                missing_ids.append(pokemon_id)
        
        self.assertIn("002", missing_ids)
        self.assertIn("004", missing_ids)
        
        print_test_result("test_find_missing_sprites", True,
                         f"Found {len(missing_sprites)} missing sprites")


if __name__ == "__main__":
    # Run tests with detailed output
    unittest.main(verbosity=2)
