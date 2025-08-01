#!/usr/bin/env python3
"""
Pokemon sprite generation training pipeline with comprehensive logging and WandB integration.

This module provides a comprehensive training pipeline for image-to-image translation models
with experiment tracking, model evaluation, and generation testing capabilities.
It supports both development and production training configurations with
extensive logging and monitoring.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, List

# Add the src directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import torch
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Import logging configuration
from core.logging_config import initialize_project_logging, get_logger, log_system_info, TrainingProgressLogger

from core.trainer import PokemonSpriteTrainer
from core.memory_efficient_trainer import MemoryEfficientPokemonTrainer
from config.settings import (
    get_model_config, 
    get_training_config, 
    create_experiment_config,
    list_available_models,
    get_available_training_configs,
    get_data_root_dir
)
from data.augmentation import get_augmentation_config, AUGMENTATION_PRESETS

# Initialize module logger
logger = get_logger(__name__)


class PokemonDataset(Dataset):
    """Dataset for Pokemon artwork to sprite translation with advanced augmentation."""
    
    def __init__(self, data_dir: str, split: str = "train", image_size: int = 64,
                 augmentation_level: str = "standard"):
        """
        Initialize Pokemon dataset.
        
        Args:
            data_dir: Path to training data directory.
            split: Data split ("train" or "val").
            image_size: Target image size.
            augmentation_level: Augmentation level ("light", "standard", "production", "none").
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.augmentation_level = augmentation_level
        
        # Data paths
        self.input_dir = self.data_dir / split / "input"
        self.target_dir = self.data_dir / split / "target"
        
        if not (self.input_dir.exists() and self.target_dir.exists()):
            raise ValueError(f"Dataset directories not found: {self.input_dir}, {self.target_dir}")
        
        # Get image files
        self.input_files = sorted(list(self.input_dir.glob("*.png")))
        self.target_files = sorted(list(self.target_dir.glob("*.png")))
        
        if len(self.input_files) != len(self.target_files):
            raise ValueError(f"Mismatch in number of input ({len(self.input_files)}) and target ({len(self.target_files)}) images")
        
        # Setup augmentation pipeline
        if split == "train":
            self.augmentation = get_augmentation_config(augmentation_level, image_size)
            self.augmentation.set_dataset(self)  # For augmentations that need dataset reference
        else:
            self.augmentation = get_augmentation_config("none", image_size)
        
        # Setup basic transforms (applied after augmentation)
        self.basic_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
        
        logger.info(f"Loaded {len(self.input_files)} {split} samples")
        logger.info(f"Using '{augmentation_level}' augmentation for {split} split")
    
    def __len__(self):
        return len(self.input_files)
    
    def get_raw_sample(self, idx: int) -> Tuple[Image.Image, Image.Image]:
        """Get raw PIL images without transforms (used by some augmentations)."""
        input_img = Image.open(self.input_files[idx]).convert('RGB')
        target_img = Image.open(self.target_files[idx]).convert('RGB')
        return input_img, target_img
    
    def __getitem__(self, idx):
        # Load images
        input_img, target_img = self.get_raw_sample(idx)
        
        # Apply augmentation pipeline (only for training)
        if self.split == "train":
            input_img, target_img = self.augmentation(input_img, target_img)
        
        # Apply basic transforms (resize, normalize)
        input_tensor = self.basic_transform(input_img)
        target_tensor = self.basic_transform(target_img)
        
        return input_tensor, target_tensor


def setup_wandb(project_name: str = "pokemon-sprite-generation", model_name: str = "unknown", 
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
            tags=[model_name, config_type, "image-to-image", "pokemon"],
            notes=f"Training {model_name} using {config_type} configuration",
            config={
                "pipeline_version": "1.0",
                "training_type": "image_to_image",
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


def load_and_validate_configs(model_name: str, config_type: str):
    """Load and validate model and training configurations."""
    logger.info("Loading configurations...")
    
    try:
        model_config, training_config = create_experiment_config(model_name, config_type)
        
        logger.info(f"Model: {model_config.name}")
        logger.info(f"Architecture: {model_config.architecture}")
        logger.info(f"Output directory: {model_config.output_dir}")
        logger.info(f"Training config: {config_type}")
        logger.info(f"Epochs: {training_config.epochs}")
        logger.info(f"Batch size: {training_config.batch_size}")
        logger.info(f"Learning rate: {training_config.learning_rate}")
        logger.info(f"Image size: {training_config.image_size}")
        
        return model_config, training_config
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.info("Available models:")
        available_models = list_available_models()
        for arch, models in available_models.items():
            logger.info(f"  {arch}: {', '.join(models)}")
        
        logger.info("Available training configs:")
        available_configs = get_available_training_configs()
        logger.info(f"  {', '.join(available_configs)}")
        
        raise


def create_data_loaders(training_config, config_type: str = "development", 
                       max_samples: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    logger.info("Setting up data loaders...")
    
    # Get data directory
    data_root = get_data_root_dir()
    training_data_dir = Path(data_root) / "training_data"
    
    if not training_data_dir.exists():
        logger.error(f"Training data directory not found: {training_data_dir}")
        logger.info("Please run the data preparation notebook first")
        raise FileNotFoundError(f"Training data directory not found: {training_data_dir}")
    
    # Determine augmentation level based on training config
    if config_type in AUGMENTATION_PRESETS:
        augmentation_level = AUGMENTATION_PRESETS[config_type]
    else:
        augmentation_level = config_type  # Use directly if it's a valid augmentation level
    
    logger.info(f"Using '{augmentation_level}' augmentation level for '{config_type}' configuration")
    
    # Create datasets
    train_dataset = PokemonDataset(
        str(training_data_dir), 
        split="train",
        image_size=training_config.image_size,
        augmentation_level=augmentation_level
    )
    
    val_dataset = PokemonDataset(
        str(training_data_dir),
        split="val", 
        image_size=training_config.image_size,
        augmentation_level="none"
    )
    
    # Limit samples if specified
    if max_samples:
        if len(train_dataset) > max_samples:
            train_dataset.input_files = train_dataset.input_files[:max_samples]
            train_dataset.target_files = train_dataset.target_files[:max_samples]
            logger.info(f"Limited training samples to {max_samples}")
        
        val_samples = max_samples // 4  # 25% for validation
        if len(val_dataset) > val_samples:
            val_dataset.input_files = val_dataset.input_files[:val_samples]
            val_dataset.target_files = val_dataset.target_files[:val_samples]
            logger.info(f"Limited validation samples to {val_samples}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        drop_last=False
    )
    
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


def find_missing_sprites(artwork_dir: Path, sprite_dir: Path) -> List[Path]:
    """Find Pokemon artwork that don't have corresponding BW sprites."""
    artwork_files = list(artwork_dir.glob("*.png"))
    sprite_files = {f.stem.replace("_bw", "_artwork") for f in sprite_dir.glob("*.png")}
    
    missing_sprites = []
    for artwork_file in artwork_files:
        if artwork_file.stem not in sprite_files:
            missing_sprites.append(artwork_file)
    
    logger.info(f"Found {len(missing_sprites)} Pokemon without BW sprites")
    return missing_sprites


def generate_missing_sprites(trainer: PokemonSpriteTrainer, missing_artwork: List[Path], 
                           output_dir: Path, max_generate: int = 50) -> None:
    """Generate sprites for Pokemon that don't have BW sprites."""
    if not missing_artwork:
        logger.info("No missing sprites to generate")
        return
        
    logger.info(f"Generating sprites for {min(len(missing_artwork), max_generate)} Pokemon without BW sprites...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the generator model
    if isinstance(trainer.models, dict):
        if "generator" in trainer.models:
            generator = trainer.models["generator"]
        elif "generator_A2B" in trainer.models:
            generator = trainer.models["generator_A2B"]
        else:
            logger.warning("No suitable generator found")
            return
    else:
        generator = trainer.models
    
    generator.eval()
    
    # Setup transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((trainer.training_config.image_size, trainer.training_config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    generated_count = 0
    with torch.no_grad():
        for artwork_path in missing_artwork[:max_generate]:
            try:
                # Load and preprocess artwork
                artwork_img = Image.open(artwork_path).convert('RGB')
                input_tensor = transform(artwork_img).unsqueeze(0).to(trainer.device)
                
                # Generate sprite
                generated_sprite = generator(input_tensor)
                
                # Denormalize and convert back to PIL
                generated_sprite = (generated_sprite + 1) / 2.0  # [-1, 1] -> [0, 1]
                generated_sprite = generated_sprite.clamp(0, 1)
                generated_sprite = generated_sprite.squeeze(0).cpu()
                
                # Convert to PIL and save
                generated_pil = transforms.ToPILImage()(generated_sprite)
                
                # Extract Pokemon ID from filename
                pokemon_id = artwork_path.stem.split('_')[1]
                output_path = output_dir / f"pokemon_{pokemon_id}_generated_sprite.png"
                generated_pil.save(output_path)
                
                generated_count += 1
                
                if generated_count % 10 == 0:
                    logger.info(f"Generated {generated_count}/{min(len(missing_artwork), max_generate)} sprites")
                    
            except Exception as e:
                logger.warning(f"Failed to generate sprite for {artwork_path.name}: {e}")
                continue
    
    logger.info(f"Successfully generated {generated_count} sprites for missing Pokemon")
    logger.info(f"Generated sprites saved to: {output_dir}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Pokemon sprite generation models")
    parser.add_argument("--model", type=str, required=True,
                       help="Model configuration name")
    parser.add_argument("--config", type=str, default="development",
                       choices=["test", "development", "production", "pixel_art_optimal", "anti_overfitting"],
                       help="Training configuration type")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of training samples (for testing)")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable WandB logging")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--skip-generation", action="store_true",
                       help="Skip generating sprites for missing Pokemon after training")
    parser.add_argument("--max-generate", type=int, default=None,
                       help="Maximum number of missing sprites to generate")
    parser.add_argument("--augmentation", type=str, default=None,
                       choices=["none", "light", "standard", "production", "anti_overfitting"],
                       help="Data augmentation level (overrides config-based default)")
    parser.add_argument("--memory-efficient", action="store_true",
                       help="Use memory-efficient trainer")
    parser.add_argument("--backbone", type=str, default=None,
                       choices=["resnet50", "resnet34", "efficientnet_b0"],
                       help="Pretrained backbone for pix2pix-pretrained model")
    
    
    args = parser.parse_args()
    
    # Initialize logging
    experiment_id = f"pokemon_sprites_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    initialize_project_logging(
        project_name="pokemon_sprites",
        log_level=args.log_level,
        model_name=args.model,
        experiment_id=experiment_id,
        enable_file_logging=True,
        enable_json_logging=True
    )
    
    logger.info("="*80)
    logger.info("POKEMON SPRITE GENERATION TRAINING PIPELINE")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Experiment ID: {experiment_id}")
    
    # Log system information
    log_system_info()
    
    try:
        # Load configurations
        model_config, training_config = load_and_validate_configs(args.model, args.config)
        
        # Override max samples if specified
        if args.max_samples:
            training_config.max_samples = args.max_samples
        
        # Setup WandB
        wandb_run = None
        if not args.no_wandb and training_config.wandb_log:
            wandb_run = setup_wandb(
                project_name=training_config.wandb_project,
                model_name=args.model,
                config_type=args.config
            )
            
            if wandb_run:
                # Log configurations to WandB
                wandb_config = {
                    "model_name": model_config.name,
                    "architecture": model_config.architecture,
                    "epochs": training_config.epochs,
                    "batch_size": training_config.batch_size,
                    "learning_rate": training_config.learning_rate,
                    "image_size": training_config.image_size,
                    "max_samples": training_config.max_samples,
                }
                wandb_run.config.update(wandb_config)
        
        # Create data loaders
        augmentation_level = args.augmentation if args.augmentation else args.config
        train_loader, val_loader = create_data_loaders(training_config, augmentation_level, args.max_samples)
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        if args.memory_efficient:
            logger.info("Using memory-efficient trainer with gradient accumulation and mixed precision")
            trainer = MemoryEfficientPokemonTrainer(model_config, training_config, wandb_run)
        else:
            trainer = PokemonSpriteTrainer(model_config, training_config, wandb_run)
        
        # Start training
        start_time = time.time()
        trainer.train(train_loader, val_loader)
        training_time = time.time() - start_time
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Average time per epoch: {training_time / training_config.epochs:.2f} seconds")
        
        # Generate sprites for Pokemon without BW sprites
        if not args.skip_generation:
            logger.info("=" * 50)
            logger.info("POST-TRAINING: GENERATING MISSING SPRITES")
            logger.info("=" * 50)
            
            try:
                data_root = Path(get_data_root_dir())
                artwork_dir = data_root / "pokemon_complete" / "sugimori_artwork"
                sprite_dir = data_root / "pokemon_complete" / "black_white_sprites"
                generated_output_dir = data_root / "generated_sprites"
                
                if artwork_dir.exists() and sprite_dir.exists():
                    missing_artwork = find_missing_sprites(artwork_dir, sprite_dir)
                    
                    if missing_artwork:
                        # Determine max generation count
                        if args.max_generate is not None:
                            max_generate = args.max_generate
                        elif args.config == "test":
                            max_generate = 10
                        elif args.config == "development":
                            max_generate = 50
                        else:  # production
                            max_generate = len(missing_artwork)  # Generate all
                        
                        generate_missing_sprites(trainer, missing_artwork, generated_output_dir, max_generate)
                        
                        # Log to WandB if available
                        if wandb_run:
                            # Log some generated samples
                            generated_files = list(generated_output_dir.glob("*.png"))[:8]
                            if generated_files:
                                wandb_images = []
                                for img_path in generated_files:
                                    img = Image.open(img_path)
                                    wandb_images.append(wandb.Image(img, caption=f"Generated: {img_path.stem}"))
                                
                                wandb_run.log({"generated_missing_sprites": wandb_images})
                                logger.info(f"Logged {len(wandb_images)} generated sprites to WandB")
                    else:
                        logger.info("No missing sprites found - all Pokemon have BW sprites!")
                else:
                    logger.warning("Artwork or sprite directories not found - skipping sprite generation")
                    
            except Exception as e:
                logger.error(f"Failed to generate missing sprites: {e}", exc_info=True)
        else:
            logger.info("Sprite generation skipped (--skip-generation flag used)")
        
        # Cleanup
        if wandb_run:
            wandb_run.finish()
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if 'wandb_run' in locals() and wandb_run:
            wandb_run.finish()
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        if 'wandb_run' in locals() and wandb_run:
            wandb_run.finish()
        sys.exit(1)


if __name__ == "__main__":
    main()
