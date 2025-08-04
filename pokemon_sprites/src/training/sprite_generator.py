"""
Post-training sprite generation utilities.

This module handles sprite generation tasks after training completion,
including missing sprite generation and comparison visualization.
"""

import sys
from pathlib import Path
from typing import List, Optional

import wandb
from PIL import Image

from config.settings import get_data_root_dir
from core.logging_config import get_logger

# Add src to path
current_dir = Path(__file__).parent
src_path = str(current_dir.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)


logger = get_logger(__name__)


def find_missing_sprites(artwork_dir: Path, sprite_dir: Path) -> List[Path]:
    """
    Find artwork files that don't have corresponding sprites.

    Args:
        artwork_dir: Directory containing artwork files
        sprite_dir: Directory containing sprite files

    Returns:
        List of artwork files without corresponding sprites
    """
    missing_sprites = []

    try:
        if not artwork_dir.exists() or not sprite_dir.exists():
            logger.warning("Artwork or sprite directory does not exist")
            return missing_sprites

        # Get existing sprite Pokemon IDs
        existing_sprite_ids = set()
        for sprite_file in sprite_dir.glob("*.png"):
            # Extract Pokemon ID from filename (e.g., "pokemon_001_bw.png" ->
            # "001")
            parts = sprite_file.stem.split("_")
            if len(parts) >= 2:
                existing_sprite_ids.add(parts[1])

        # Check artwork files for missing sprites
        for artwork_file in artwork_dir.glob("*.png"):
            parts = artwork_file.stem.split("_")
            if len(parts) >= 2:
                pokemon_id = parts[1]
                if pokemon_id not in existing_sprite_ids:
                    missing_sprites.append(artwork_file)

        logger.info(
            f"Found {len(missing_sprites)} artwork files without corresponding sprites"
        )
        return missing_sprites

    except Exception as e:
        logger.error(f"Error finding missing sprites: {e}")
        return missing_sprites


def generate_missing_sprites(
    trainer,
    missing_artwork: List[Path],
    output_dir: Path,
    max_generation_count: Optional[int] = None,
) -> Path:
    """
    Generate sprites for missing artwork using the trained model.

    Args:
        trainer: Trained model trainer
        missing_artwork: List of artwork files to generate sprites for
        output_dir: Directory to save generated sprites
        max_generation_count: Maximum number of sprites to generate

    Returns:
        Path to directory containing generated sprites
    """
    try:
        if not missing_artwork:
            logger.info("No missing sprites to generate")
            return output_dir

        # Limit generation count if specified
        if max_generation_count and max_generation_count > 0:
            missing_artwork = missing_artwork[:max_generation_count]

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating {len(missing_artwork)} missing sprites...")

        generated_count = 0
        for artwork_path in missing_artwork:
            try:
                # Load and preprocess artwork
                artwork_image = Image.open(artwork_path).convert("RGBA")

                # Generate sprite using trainer
                generated_sprite = trainer.generate_single_sprite(
                    artwork_image
                )

                # Save generated sprite
                pokemon_id = artwork_path.stem.split("_")[1]
                output_path = (
                    output_dir / f"generated_pokemon_{pokemon_id}_sprite.png"
                )
                generated_sprite.save(output_path)

                generated_count += 1

                if generated_count % 10 == 0:
                    logger.info(
                        f"Generated {generated_count}/{len(missing_artwork)} sprites"
                    )

            except Exception as e:
                logger.error(
                    f"Failed to generate sprite for {artwork_path.name}: {e}"
                )
                continue

        logger.info(f"Successfully generated {generated_count} sprites")
        return output_dir

    except Exception as e:
        logger.error(f"Sprite generation failed: {e}")
        return output_dir


def log_generated_sprites_to_wandb(
    wandb_run, generated_output_dir: Path
) -> None:
    """
    Log generated sprites to Weights & Biases.

    Args:
        wandb_run: WandB run object
        generated_output_dir: Directory containing generated sprites
    """
    try:
        if not wandb_run or not generated_output_dir.exists():
            return

        sprite_files = list(generated_output_dir.glob("*.png"))
        if not sprite_files:
            logger.info("No generated sprites to log to WandB")
            return

        # Log a sample of generated sprites
        sample_size = min(10, len(sprite_files))
        sample_files = sprite_files[:sample_size]

        images = []
        for sprite_file in sample_files:
            try:
                image = Image.open(sprite_file)
                images.append(wandb.Image(image, caption=sprite_file.stem))
            except Exception as e:
                logger.warning(
                    f"Failed to load sprite for WandB: {sprite_file.name}: {e}"
                )

        if images:
            wandb_run.log({"generated_sprites": images})
            logger.info(f"Logged {len(images)} generated sprites to WandB")

    except Exception as e:
        logger.error(f"Failed to log sprites to WandB: {e}")


def create_comparison_grid(
    original_artworks: List[Path],
    generated_sprites: List[Path],
    output_path: Path,
    grid_size: tuple = (4, 4),
) -> Optional[Path]:
    """
    Create a comparison grid showing original artwork and generated sprites.

    Args:
        original_artworks: List of original artwork files
        generated_sprites: List of generated sprite files
        output_path: Path to save the comparison grid
        grid_size: Size of the comparison grid (rows, cols)

    Returns:
        Path to saved comparison grid or None if failed
    """
    try:
        from PIL import Image

        rows, cols = grid_size
        max_pairs = min(
            len(original_artworks), len(generated_sprites), rows * cols // 2
        )

        if max_pairs == 0:
            logger.warning(
                "No artwork-sprite pairs available for comparison grid"
            )
            return None

        # Image dimensions
        img_size = 128
        grid_width = cols * img_size
        grid_height = rows * img_size

        # Create grid image
        grid_image = Image.new("RGB", (grid_width, grid_height), "white")

        pair_idx = 0
        for row in range(0, rows, 2):  # Process pairs of rows
            for col in range(cols):
                if pair_idx >= max_pairs:
                    break

                try:
                    # Load and resize artwork
                    artwork = Image.open(original_artworks[pair_idx]).convert(
                        "RGB"
                    )
                    artwork = artwork.resize(
                        (img_size, img_size), Image.Resampling.LANCZOS
                    )

                    # Load and resize sprite
                    sprite = Image.open(generated_sprites[pair_idx]).convert(
                        "RGB"
                    )
                    sprite = sprite.resize(
                        (img_size, img_size), Image.Resampling.LANCZOS
                    )

                    # Paste into grid
                    artwork_pos = (col * img_size, row * img_size)
                    sprite_pos = (col * img_size, (row + 1) * img_size)

                    grid_image.paste(artwork, artwork_pos)
                    grid_image.paste(sprite, sprite_pos)

                    pair_idx += 1

                except Exception as e:
                    logger.warning(
                        f"Failed to add pair {pair_idx} to grid: {e}"
                    )
                    pair_idx += 1
                    continue

            if pair_idx >= max_pairs:
                break

        # Save grid
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grid_image.save(output_path)

        logger.info(
            f"Created comparison grid with {pair_idx} pairs: {output_path}"
        )
        return output_path

    except Exception as e:
        logger.error(f"Failed to create comparison grid: {e}")
        return None


def handle_post_training_generation(args, trainer, wandb_run) -> None:
    """
    Handle all post-training generation tasks.

    Args:
        args: Command line arguments
        trainer: Trained model trainer
        wandb_run: WandB run object
    """
    try:
        if not hasattr(args, "generate") or not args.generate:
            logger.info("Post-training generation disabled")
            return

        logger.info("Starting post-training generation tasks...")

        # Get data directories
        data_root = Path(get_data_root_dir())
        pokemon_data_dir = data_root / "pokemon_complete"
        artwork_dir = pokemon_data_dir / "artwork"
        sprite_dir = pokemon_data_dir / "sprites"

        # Find missing sprites
        missing_artwork = find_missing_sprites(artwork_dir, sprite_dir)

        if not missing_artwork:
            logger.info("No missing sprites found")
            return

        # Determine generation count
        max_count = _determine_max_generation_count(args, missing_artwork)

        # Generate missing sprites
        output_dir = (
            pokemon_data_dir
            / "generated_sprites"
            / f"run_{trainer.experiment_id}"
        )
        generated_output_dir = generate_missing_sprites(
            trainer, missing_artwork, output_dir, max_count
        )

        # Log to WandB
        log_generated_sprites_to_wandb(wandb_run, generated_output_dir)

        # Create comparison grid
        generated_files = list(generated_output_dir.glob("*.png"))
        if generated_files and len(missing_artwork) > 0:
            comparison_path = generated_output_dir / "comparison_grid.png"
            create_comparison_grid(
                missing_artwork[: len(generated_files)],
                generated_files,
                comparison_path,
            )

        logger.info("Post-training generation completed successfully")

    except Exception as e:
        logger.error(f"Post-training generation failed: {e}")


def _determine_max_generation_count(
    args, missing_artwork: List[Path]
) -> Optional[int]:
    """
    Determine maximum number of sprites to generate.

    Args:
        args: Command line arguments
        missing_artwork: List of missing artwork files

    Returns:
        Maximum generation count or None for unlimited
    """
    if hasattr(args, "max_generate") and args.max_generate:
        return min(args.max_generate, len(missing_artwork))

    # Default limits based on configuration
    if hasattr(args, "config"):
        limits = {"test": 5, "development": 20, "production": None}  # No limit
        return limits.get(args.config, 10)

    return 10  # Default fallback
