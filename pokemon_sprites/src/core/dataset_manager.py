"""
Dataset management and setup utilities for Pokemon sprite generation.

This module handles dataset downloading, validation, and preparation
for training pipelines.
"""

import sys
from pathlib import Path

from core.logging_config import get_logger
from data.loaders import (
    check_existing_raw_data,
    create_preprocessing_pipeline,
    download_pokemon_data_with_cache,
    find_valid_pairs,
)

# Add src to path
current_dir = Path(__file__).parent
src_path = str(current_dir.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)


logger = get_logger(__name__)


class DatasetManager:
    """Manages Pokemon sprite dataset setup and validation."""

    def __init__(self, data_root: Path):
        """
        Initialize dataset manager.

        Args:
            data_root: Root directory for dataset storage
        """
        self.data_root = data_root
        self.pokemon_data_dir = data_root / "pokemon_complete"
        self.artwork_dir = self.pokemon_data_dir / "artwork"
        self.sprites_dir = self.pokemon_data_dir / "sprites"
        self.training_data_dir = (
            self.pokemon_data_dir / "processed" / "input_96"
        )

    def is_dataset_ready(self) -> bool:
        """
        Check if training data already exists.

        Returns:
            True if dataset is ready for training
        """
        return (
            self.training_data_dir.exists()
            and (self.training_data_dir / "train").exists()
            and (self.training_data_dir / "val").exists()
        )

    def setup_dataset(self) -> bool:
        """
        Automatically download and prepare dataset if missing.

        Returns:
            True if dataset setup successful, False otherwise
        """
        if self.is_dataset_ready():
            logger.info(
                f"Training data found in {self.training_data_dir}, proceeding..."
            )
            return True

        logger.info("Training data not found, checking for raw data...")

        try:
            # Create data directories
            self.artwork_dir.mkdir(parents=True, exist_ok=True)
            self.sprites_dir.mkdir(parents=True, exist_ok=True)

            # Check if we already have sufficient raw data
            raw_data_status = check_existing_raw_data(
                self.sprites_dir, self.artwork_dir
            )

            if raw_data_status["sufficient"]:
                logger.info(
                    f"Found existing raw data: {raw_data_status['details']}"
                )
                logger.info(
                    "Skipping download, proceeding to create training dataset..."
                )
            else:
                logger.info(
                    f"Insufficient raw data: {raw_data_status['details']}"
                )
                logger.info("Downloading Pokemon data...")

                # Download and prepare data
                if not self._download_pokemon_data():
                    return False

            # Create training dataset
            return self._create_training_dataset()

        except Exception as e:
            logger.error(f"Dataset setup failed: {e}")
            return False

    def validate_and_prepare_data(self) -> bool:
        """
        Validate existing data and prepare if necessary.

        Returns:
            True if data validation and preparation successful, False otherwise
        """
        return self.setup_dataset()

    def _download_pokemon_data(self) -> bool:
        """Download Pokemon artwork and sprites."""
        pokemon_ids = list(range(1, 899))  # Gen 1-8 Pokemon

        # Download Pokemon artwork (official art)
        logger.info(f"Downloading Pokemon artwork to {self.artwork_dir}...")
        artwork_downloaded, artwork_cached, artwork_failed = (
            download_pokemon_data_with_cache(
                self.artwork_dir, pokemon_ids, sprite_type="artwork"
            )
        )

        # Rename artwork files to expected format
        self._rename_files(self.artwork_dir, "_artwork.png")

        # Download Pokemon sprites (Black/White)
        logger.info("Downloading Pokemon Black/White sprites...")
        sprites_downloaded, sprites_cached, sprites_failed = (
            download_pokemon_data_with_cache(
                self.sprites_dir, pokemon_ids, sprite_type="black-white"
            )
        )

        # Rename sprite files to expected format
        self._rename_files(self.sprites_dir, "_bw.png")

        return True

    def _rename_files(self, directory: Path, expected_suffix: str) -> None:
        """Rename files to expected format."""
        for file_path in directory.glob("*.png"):
            if not file_path.name.endswith(expected_suffix):
                pokemon_id = file_path.stem.split("_")[-1]
                new_name = f"pokemon_{pokemon_id}{expected_suffix}"
                file_path.rename(directory / new_name)

    def _create_training_dataset(self) -> bool:
        """Create the training dataset from downloaded files."""
        # Find valid pairs
        valid_pairs = find_valid_pairs(self.sprites_dir, self.artwork_dir)

        if len(valid_pairs) < 50:  # Minimum viable dataset
            logger.error(
                f"Insufficient valid pairs found: {len(valid_pairs)}. "
                "Dataset setup failed."
            )
            return False

        logger.info(f"Found {len(valid_pairs)} valid artwork-sprite pairs")

        # Create preprocessing pipeline
        logger.info("Creating training dataset...")
        try:
            processed_dir, metadata = create_preprocessing_pipeline(
                self.pokemon_data_dir
            )
            logger.info(
                f"Dataset setup complete. {metadata.get('total_pairs', 0)} pairs "
                "ready for training."
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create training dataset: {e}")
            return False

    def get_dataset_info(self) -> dict:
        """Get information about the current dataset."""
        if not self.is_dataset_ready():
            return {"status": "not_ready"}

        train_dir = self.training_data_dir / "train"
        val_dir = self.training_data_dir / "val"

        train_count = (
            len(list(train_dir.glob("*/*.png"))) if train_dir.exists() else 0
        )
        val_count = (
            len(list(val_dir.glob("*/*.png"))) if val_dir.exists() else 0
        )

        return {
            "status": "ready",
            "train_samples": train_count
            // 2,  # Divided by 2 (input/target pairs)
            "val_samples": val_count // 2,
            "total_samples": (train_count + val_count) // 2,
            "data_dir": str(self.training_data_dir),
        }
