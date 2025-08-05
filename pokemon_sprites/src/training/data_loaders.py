"""
Data loading utilities for the training pipeline.

This module provides functions to create data loaders for training and validation
with proper augmentation and sampling strategies.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

from torch.utils.data import DataLoader, Dataset

from config.settings import get_data_root_dir
from core.logging_config import get_logger
from data.augmentation import get_augmentation_config

# Add src to path
current_dir = Path(__file__).parent
src_path = str(current_dir.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)


logger = get_logger(__name__)


class PokemonDataset(Dataset):
    """
    Dataset for Pokemon artwork to sprite translation with advanced augmentation.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: int = 96,
        augmentation_level: str = "conservative",
    ):
        """
        Initialize Pokemon dataset with ARGB support.

        Args:
            data_dir: Path to training data directory
            split: Data split ("train" or "val")
            image_size: Target image size (reduced to 96 for sprite handling)
            augmentation_level: Augmentation level
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.augmentation_level = augmentation_level

        # Data paths
        self.input_dir = self.data_dir / split / "input"
        self.target_dir = self.data_dir / split / "target"

        if not (self.input_dir.exists() and self.target_dir.exists()):
            raise ValueError(
                f"Dataset directories not found: {self.input_dir}, {self.target_dir}"
            )

        # Get image files
        self.input_files = sorted(list(self.input_dir.glob("*.png")))
        self.target_files = sorted(list(self.target_dir.glob("*.png")))

        if len(self.input_files) != len(self.target_files):
            raise ValueError(
                f"Mismatch in number of input ({len(self.input_files)}) "
                f"and target ({len(self.target_files)}) images"
            )

        # Setup augmentation pipeline
        if split == "train":
            self.augmentation = get_augmentation_config(
                augmentation_level, image_size
            )
            if hasattr(self.augmentation, "set_dataset"):
                self.augmentation.set_dataset(self)
        else:
            self.augmentation = get_augmentation_config("none", image_size)

        logger.info(f"Loaded {len(self.input_files)} {split} samples")

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.input_files)

    def __getitem__(self, idx: int):
        """
        Get dataset item.

        Args:
            idx: Index of the item

        Returns:
            Tuple of (input_tensor, target_tensor)
        """
        input_path = self.input_files[idx]
        target_path = self.target_files[idx]

        try:
            from PIL import Image
            from torchvision import transforms

            # Load images
            input_image = Image.open(input_path).convert("RGBA")
            target_image = Image.open(target_path).convert("RGBA")

            # Apply augmentations if available
            if self.augmentation:
                input_image, target_image = self.augmentation(
                    input_image, target_image
                )

            # Convert to tensors
            to_tensor = transforms.ToTensor()
            input_tensor = to_tensor(input_image)
            target_tensor = to_tensor(target_image)

            return input_tensor, target_tensor

        except Exception as e:
            logger.error(f"Failed to load item {idx}: {e}")
            # Return a fallback item
            return self.__getitem__(0 if idx != 0 else 1)

    def get_raw_sample(self, idx: int):
        """
        Get raw sample without augmentation.

        Args:
            idx: Index of the item

        Returns:
            Tuple of (input_image, target_image) as PIL Images
        """
        from PIL import Image

        input_path = self.input_files[idx]
        target_path = self.target_files[idx]

        input_image = Image.open(input_path).convert("RGBA")
        target_image = Image.open(target_path).convert("RGBA")

        return input_image, target_image


def create_data_loaders(
    training_config,
    augmentation_level: str = "conservative",
    max_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.

    Args:
        training_config: Training configuration object
        augmentation_level: Level of data augmentation to apply
        max_samples: Maximum number of samples to use (for testing)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    try:
        # Get data directory - prefer 96 for sprite handling
        data_root = Path(get_data_root_dir())
        target_size = getattr(training_config, "image_size", 96)
        data_dir = (
            data_root
            / "pokemon_complete"
            / "processed"
            / f"input_{target_size}"
        )

        # Fallback hierarchy: try 256, then 128, then 96
        if not data_dir.exists():
            # Try 256 first
            fallback_dir = (
                data_root / "pokemon_complete" / "processed" / "input_256"
            )
            if fallback_dir.exists():
                data_dir = fallback_dir
                target_size = 256
            else:
                # Try 128
                fallback_dir = (
                    data_root / "pokemon_complete" / "processed" / "input_128"
                )
                if fallback_dir.exists():
                    data_dir = fallback_dir
                    target_size = 128
                else:
                    # Try 96
                    fallback_dir = (
                        data_root
                        / "pokemon_complete"
                        / "processed"
                        / "input_96"
                    )
                    if fallback_dir.exists():
                        data_dir = fallback_dir
                        target_size = 96

        if not data_dir.exists():
            raise ValueError(f"Training data directory not found: {data_dir}")

        # Create datasets with ARGB support
        train_dataset = PokemonDataset(
            data_dir=str(data_dir),
            split="train",
            image_size=target_size,
            augmentation_level=augmentation_level,
        )

        val_dataset = PokemonDataset(
            data_dir=str(data_dir),
            split="val",
            image_size=target_size,
            augmentation_level="none",  # No augmentation for validation
        )

        # Apply max_samples limit if specified
        if max_samples and max_samples > 0:
            import random

            from torch.utils.data import Subset

            # Limit training samples
            if len(train_dataset) > max_samples:
                indices = random.sample(range(len(train_dataset)), max_samples)
                train_dataset = Subset(train_dataset, indices)

            # Limit validation samples proportionally
            val_ratio = 0.2  # Keep 20% for validation
            max_val_samples = max(1, int(max_samples * val_ratio))
            if len(val_dataset) > max_val_samples:
                indices = random.sample(
                    range(len(val_dataset)), max_val_samples
                )
                val_dataset = Subset(val_dataset, indices)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=getattr(training_config, "num_workers", 4),
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=getattr(training_config, "num_workers", 4),
            pin_memory=True,
            drop_last=False,
        )

        logger.info(
            f"Created data loaders: {len(train_dataset)} train, {len(val_dataset)} val samples"
        )
        return train_loader, val_loader

    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        raise


def get_dataset_statistics(data_dir: Path) -> dict:
    """
    Get statistics about the dataset.

    Args:
        data_dir: Path to dataset directory

    Returns:
        Dictionary with dataset statistics
    """
    try:
        train_input = data_dir / "train" / "input"
        data_dir / "train" / "target"
        val_input = data_dir / "val" / "input"
        data_dir / "val" / "target"

        stats = {
            "train_samples": (
                len(list(train_input.glob("*.png")))
                if train_input.exists()
                else 0
            ),
            "val_samples": (
                len(list(val_input.glob("*.png"))) if val_input.exists() else 0
            ),
            "total_samples": 0,
        }

        stats["total_samples"] = stats["train_samples"] + stats["val_samples"]

        return stats

    except Exception as e:
        logger.error(f"Failed to get dataset statistics: {e}")
        return {"train_samples": 0, "val_samples": 0, "total_samples": 0}
        return {"train_samples": 0, "val_samples": 0, "total_samples": 0}
