"""
Data loading and preparation utilities for Pokémon sprite generation project.

This module contains functions for downloading, organizing, and preparing
Pokémon sprite and artwork data for machine learning training.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image


def download_pokemon_data_with_cache(
    data_dir: Path, pokemon_ids: List[int], sprite_type: str = "black-white"
) -> Tuple[int, int, List[int]]:
    """
    Download Pokémon sprites with intelligent caching.

    Args:
        data_dir: Directory to save sprites
        pokemon_ids: List of Pokémon IDs to download
        sprite_type: Type of sprite to download

    Returns:
        Tuple of (downloaded_count, cached_count, failed_ids)
    """
    data_dir.mkdir(exist_ok=True)

    downloaded_count = 0
    cached_count = 0
    failed_downloads = []

    for pokemon_id in pokemon_ids:
        sprite_path = data_dir / f"pokemon_{pokemon_id:04d}.png"

        # Check cache
        if sprite_path.exists() and sprite_path.stat().st_size > 0:
            cached_count += 1
            continue

        try:
            if sprite_type == "black-white":
                url = (
                    "https://raw.githubusercontent.com/PokeAPI/sprites/master/"
                    "sprites/pokemon/versions/generation-v/black-white/"
                    f"{pokemon_id}.png"
                )
            elif sprite_type == "artwork":
                url = (
                    "https://raw.githubusercontent.com/PokeAPI/sprites/master/"
                    "sprites/pokemon/other/official-artwork/"
                    f"{pokemon_id}.png"
                )
            else:
                url = (
                    "https://raw.githubusercontent.com/PokeAPI/sprites/master/"
                    f"sprites/pokemon/{pokemon_id}.png"
                )

            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(sprite_path, "wb") as f:
                    f.write(response.content)
                downloaded_count += 1
            else:
                failed_downloads.append(pokemon_id)

        except Exception:
            failed_downloads.append(pokemon_id)

    return downloaded_count, cached_count, failed_downloads


def find_valid_pairs(sprites_dir: Path, artwork_dir: Path) -> List[Dict]:
    """
    Find valid sprite-artwork pairs.

    Args:
        sprites_dir: Directory containing sprite files
        artwork_dir: Directory containing artwork files

    Returns:
        List of dictionaries containing pair information
    """
    if not (sprites_dir.exists() and artwork_dir.exists()):
        return []

    sprite_files = list(sprites_dir.glob("*.png"))
    valid_pairs = []

    for sprite_file in sprite_files:
        # Extract Pokémon ID from filename (handles multiple formats)
        if "_bw" in sprite_file.stem:
            # Format: pokemon_0001_bw.png -> extract "0001"
            pokemon_id = sprite_file.stem.split("_")[1]
        elif (
            sprite_file.stem.startswith("pokemon_")
            and len(sprite_file.stem.split("_")) >= 2
        ):
            # Format: pokemon_0001.png -> extract "0001"
            pokemon_id = sprite_file.stem.split("_")[1]
        else:
            # Skip files that do not match expected naming convention
            continue

        artwork_file = artwork_dir / f"pokemon_{pokemon_id}_artwork.png"

        if artwork_file.exists():
            try:
                # Verify both files are valid images
                with Image.open(sprite_file) as sprite_img:
                    with Image.open(artwork_file) as artwork_img:
                        if (
                            sprite_img.width > 0
                            and sprite_img.height > 0
                            and artwork_img.width > 0
                            and artwork_img.height > 0
                        ):
                            valid_pairs.append(
                                {
                                    "pokemon_id": pokemon_id,
                                    "sprite_path": sprite_file,
                                    "artwork_path": artwork_file,
                                }
                            )
            except Exception:
                continue

    return valid_pairs


def resize_with_padding(
    img: Image.Image, target_size: Tuple[int, int]
) -> Image.Image:
    """
    Resize image to target size with padding to maintain aspect ratio.

    Args:
        img: PIL Image to resize
        target_size: Target (width, height)

    Returns:
        Resized PIL Image with padding
    """
    img.thumbnail(target_size, Image.Resampling.LANCZOS)

    new_img = Image.new("RGB", target_size, (255, 255, 255))

    # Center the image
    x = (target_size[0] - img.width) // 2
    y = (target_size[1] - img.height) // 2

    new_img.paste(img, (x, y))
    return new_img


def process_image_pairs(
    pairs: List[Dict],
    output_dir: Path,
    target_size: Tuple[int, int] = (256, 256),
) -> int:
    """
    Process and save image pairs for training.

    Args:
        pairs: List of image pair dictionaries
        output_dir: Directory to save processed images
        target_size: Target image size

    Returns:
        Number of successfully processed pairs
    """
    # Create output directories
    input_dir = output_dir / "input_artwork"
    target_dir = output_dir / "target_sprites"

    for dir_path in [input_dir, target_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    successful_pairs = 0

    for pair in pairs:
        try:
            pokemon_id = pair["pokemon_id"]

            # Load and process images
            artwork_img = Image.open(pair["artwork_path"]).convert("RGB")
            sprite_img = Image.open(pair["sprite_path"]).convert("RGB")

            artwork_processed = resize_with_padding(artwork_img, target_size)
            sprite_processed = resize_with_padding(sprite_img, target_size)

            # Save processed images
            artwork_processed.save(input_dir / f"pokemon_{pokemon_id}.png")
            sprite_processed.save(target_dir / f"pokemon_{pokemon_id}.png")

            successful_pairs += 1

        except Exception:
            continue

    return successful_pairs


def create_train_val_split(
    pairs: List[Dict], validation_split: float = 0.15
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split pairs into training and validation sets.

    Args:
        pairs: List of image pair dictionaries
        validation_split: Fraction for validation

    Returns:
        Tuple of (train_pairs, val_pairs)
    """
    random.shuffle(pairs)

    val_size = int(len(pairs) * validation_split)
    train_pairs = pairs[val_size:]
    val_pairs = pairs[:val_size]

    return train_pairs, val_pairs


def save_dataset_metadata(output_dir: Path, dataset_info: Dict) -> None:
    """
    Save dataset metadata to JSON file.

    Args:
        output_dir: Directory to save metadata
        dataset_info: Dictionary containing dataset information
    """
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)


def calculate_image_stats(image_files: List[Path]) -> Dict:
    """
    Calculate basic statistics for a list of image files.

    Args:
        image_files: List of image file paths

    Returns:
        Dictionary containing image statistics
    """
    brightness_vals = []
    contrast_vals = []
    color_counts = []

    # Sample up to 50 images for speed
    sample_files = random.sample(image_files, min(50, len(image_files)))

    for img_path in sample_files:
        try:
            with Image.open(img_path) as img:
                img_array = np.array(img.convert("RGB"))

                brightness_vals.append(np.mean(img_array))
                contrast_vals.append(np.std(img_array))

                unique_colors = len(
                    np.unique(img_array.reshape(-1, 3), axis=0)
                )
                color_counts.append(unique_colors)

        except Exception:
            continue

    return {
        "brightness": {
            "mean": np.mean(brightness_vals) if brightness_vals else 0,
            "std": np.std(brightness_vals) if brightness_vals else 0,
        },
        "contrast": {
            "mean": np.mean(contrast_vals) if contrast_vals else 0,
            "std": np.std(contrast_vals) if contrast_vals else 0,
        },
        "color_complexity": {
            "mean": np.mean(color_counts) if color_counts else 0,
            "std": np.std(color_counts) if color_counts else 0,
        },
        "sample_size": len(brightness_vals),
    }


def create_training_dataset(
    pairs: List[Dict],
    output_dir: Path,
    train_split: float = 0.8,
    image_size: Tuple[int, int] = (64, 64),
    augment_data: bool = True,
) -> Dict:
    """
    Create a training dataset from valid pairs of artwork and sprites.

    Args:
        pairs: List of valid artwork-sprite pairs
        output_dir: Output directory for processed dataset
        train_split: Fraction of data to use for training
        image_size: Target image size (width, height)
        augment_data: Whether to apply data augmentation

    Returns:
        Dictionary containing dataset information
    """
    # Create output directories
    train_input_dir = output_dir / "train" / "input"
    train_target_dir = output_dir / "train" / "target"
    val_input_dir = output_dir / "val" / "input"
    val_target_dir = output_dir / "val" / "target"

    for dir_path in [
        train_input_dir,
        train_target_dir,
        val_input_dir,
        val_target_dir,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Shuffle pairs for random split
    random.shuffle(pairs)

    # Split data
    split_idx = int(len(pairs) * train_split)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    # Process training pairs
    print(f"Processing {len(train_pairs)} training pairs...")
    for i, pair in enumerate(train_pairs):
        # Process artwork (input)
        artwork_img = Image.open(pair["artwork_path"])
        artwork_img = resize_with_padding(artwork_img, image_size)
        artwork_output = train_input_dir / f"pokemon_{pair['pokemon_id']}.png"
        artwork_img.save(artwork_output, "PNG")

        # Process sprite (target)
        sprite_img = Image.open(pair["sprite_path"])
        sprite_img = resize_with_padding(sprite_img, image_size)
        sprite_output = train_target_dir / f"pokemon_{pair['pokemon_id']}.png"
        sprite_img.save(sprite_output, "PNG")

    # Process validation pairs
    print(f"Processing {len(val_pairs)} validation pairs...")
    for i, pair in enumerate(val_pairs):
        # Process artwork (input)
        artwork_img = Image.open(pair["artwork_path"])
        artwork_img = resize_with_padding(artwork_img, image_size)
        artwork_output = val_input_dir / f"pokemon_{pair['pokemon_id']}.png"
        artwork_img.save(artwork_output, "PNG")

        # Process sprite (target)
        sprite_img = Image.open(pair["sprite_path"])
        sprite_img = resize_with_padding(sprite_img, image_size)
        sprite_output = val_target_dir / f"pokemon_{pair['pokemon_id']}.png"
        sprite_img.save(sprite_output, "PNG")

    # Create dataset info
    dataset_info = {
        "total_pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "image_size": image_size,
        "data_paths": {
            "train_input": str(train_input_dir),
            "train_target": str(train_target_dir),
            "val_input": str(val_input_dir),
            "val_target": str(val_target_dir),
        },
        "augmentation": augment_data,
        "created_at": str(output_dir),
    }

    return dataset_info


def _find_image_subdirectories(dataset_dir: Path) -> list:
    """Find subdirectories containing image files."""
    subdirs = []
    for d in dataset_dir.iterdir():
        if d.is_dir():
            image_files = [
                f
                for f in d.glob("*")
                if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif"]
            ]
            if image_files:
                subdirs.append(d)
    return subdirs


def _load_valid_images(image_files: list, samples_needed: int):
    """Load valid images from a list of image files."""
    valid_images = []
    attempts = 0
    max_attempts = min(len(image_files), samples_needed * 3)

    while len(valid_images) < samples_needed and attempts < max_attempts:
        img_file = random.choice(image_files)
        try:
            if img_file.suffix.lower() == ".gif":
                img = Image.open(img_file)
                img = img.convert("RGB")
            else:
                img = Image.open(img_file)

            if img_file not in [vi[0] for vi in valid_images]:
                valid_images.append((img_file, img))
        except Exception:
            pass
        attempts += 1

    return valid_images


def _setup_visualization_grid(subdirs: list, samples_per_category: int):
    """Set up the matplotlib grid for visualization."""
    fig, axes = plt.subplots(
        len(subdirs),
        samples_per_category,
        figsize=(4 * samples_per_category, 3 * len(subdirs)),
    )

    if len(subdirs) == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        "Sample Images from Complete Pokémon Dataset",
        fontsize=16,
        fontweight="bold",
    )
    return fig, axes


def _display_image_or_fallback(axes, row, col, valid_images, subdir):
    """Display an image or fallback message in the given axes position."""
    if col < len(valid_images):
        img_file, img = valid_images[col]
        try:
            axes[row, col].imshow(img)
            if "_" in img_file.stem:
                pokemon_id = img_file.stem.split("_")[1]
                axes[row, col].set_title(
                    f"{subdir.name}\n#{pokemon_id}", fontsize=8
                )
            else:
                axes[row, col].set_title(
                    f"{subdir.name}\n{img_file.name}", fontsize=8
                )
            axes[row, col].axis("off")
        except Exception:
            axes[row, col].text(
                0.5,
                0.5,
                "Display\nError",
                ha="center",
                va="center",
                transform=axes[row, col].transAxes,
            )
            axes[row, col].axis("off")
    else:
        axes[row, col].text(
            0.5,
            0.5,
            "No valid\nimages",
            ha="center",
            va="center",
            transform=axes[row, col].transAxes,
        )
        axes[row, col].axis("off")


def _calculate_directory_stats(image_files: list, subdir_name: str) -> dict:
    """Calculate statistics for a directory."""
    file_count = len(image_files)
    try:
        size_mb = sum(f.stat().st_size for f in image_files if f.is_file()) / (
            1024 * 1024
        )
    except (OSError, IOError):
        size_mb = 0
    return {"files": file_count, "size_mb": size_mb}


def _print_dataset_statistics(dataset_stats: dict):
    """Print comprehensive dataset statistics."""
    print("\nDataset Statistics:")
    total_size_mb = 0
    for category, stats in dataset_stats.items():
        print(
            f"- {category.replace('_', ' ').title()}: {stats['files']} files "
            f"({stats['size_mb']:.1f} MB)"
        )
        total_size_mb += stats["size_mb"]
    print(f"- Total Size: {total_size_mb:.1f} MB")


def visualize_dataset_samples(
    dataset_dir: Path, samples_per_category: int = 3
) -> None:
    """
    Visualize samples from each category in the dataset.

    Args:
        dataset_dir: Directory containing dataset subdirectories
        samples_per_category: Number of samples to show per category
    """
    if not dataset_dir.exists():
        print("Dataset directory not found.")
        return

    subdirs = _find_image_subdirectories(dataset_dir)
    if not subdirs:
        print("No subdirectories with images found in dataset.")
        return

    fig, axes = _setup_visualization_grid(subdirs, samples_per_category)
    dataset_stats = {}

    for row, subdir in enumerate(subdirs):
        # Get image files
        image_files = [
            f
            for f in subdir.glob("*")
            if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif"]
        ]

        if not image_files:
            for col in range(samples_per_category):
                axes[row, col].text(
                    0.5,
                    0.5,
                    f"No images\nin {subdir.name}",
                    ha="center",
                    va="center",
                    transform=axes[row, col].transAxes,
                )
                axes[row, col].axis("off")
            dataset_stats[subdir.name] = {"files": 0, "size_mb": 0}
            continue

        valid_images = _load_valid_images(image_files, samples_per_category)

        # Display images
        for col in range(samples_per_category):
            _display_image_or_fallback(axes, row, col, valid_images, subdir)

        dataset_stats[subdir.name] = _calculate_directory_stats(
            image_files, subdir.name
        )

    plt.tight_layout()
    plt.show()
    _print_dataset_statistics(dataset_stats)


def get_dataset_statistics(dataset_dir: Path) -> Dict:
    """
    Get comprehensive statistics about the dataset.

    Args:
        dataset_dir: Directory containing dataset

    Returns:
        Dictionary containing dataset statistics
    """
    if not dataset_dir.exists():
        return {"error": "Dataset directory not found"}

    subdirs = [d for d in dataset_dir.iterdir() if d.is_dir()]

    dataset_info = {}
    total_size = 0

    for subdir in subdirs:
        files = list(subdir.glob("*"))
        total_files = len(files)

        if total_files > 0:
            # Calculate total size
            subdir_size = sum(f.stat().st_size for f in files if f.is_file())
            total_size += subdir_size

            # Sample a few files for detailed analysis
            sample_files = files[:5] if len(files) >= 5 else files

            sizes = []
            dimensions = []

            for file_path in sample_files:
                if file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    try:
                        img = Image.open(file_path)
                        dimensions.append((img.width, img.height))
                        sizes.append(file_path.stat().st_size)
                    except (OSError, IOError):
                        pass

            dataset_info[subdir.name] = {
                "total_files": total_files,
                "size_mb": subdir_size / (1024 * 1024),
                "avg_file_size": np.mean(sizes) if sizes else 0,
                "dimensions": dimensions,
            }

    dataset_info["total_size_mb"] = total_size / (1024 * 1024)
    return dataset_info


def analyze_image_characteristics(dataset_info: Dict) -> None:
    """
    Analyze and visualize image characteristics of the training dataset.

    Args:
        dataset_info: Dictionary containing dataset information from
                      create_training_dataset
    """
    if not dataset_info or "data_paths" not in dataset_info:
        print("No dataset information available")
        return

    print("Dataset Characteristics Analysis")
    print("=" * 40)

    train_input_dir = Path(dataset_info["data_paths"]["train_input"])
    train_target_dir = Path(dataset_info["data_paths"]["train_target"])

    # Get file lists
    input_files = list(train_input_dir.glob("*.png"))
    target_files = list(train_target_dir.glob("*.png"))

    if not input_files or not target_files:
        print("No training files found")
        return

    # Calculate statistics
    input_stats = calculate_image_stats(input_files)
    target_stats = calculate_image_stats(target_files)

    # Print statistics
    print(
        f"Artwork (Input) Statistics (sample: {input_stats['sample_size']}):"
    )
    print(
        f"- Brightness: {input_stats['brightness']['mean']:.1f} ± "
        f"{input_stats['brightness']['std']:.1f}"
    )
    print(
        f"- Contrast: {input_stats['contrast']['mean']:.1f} ± "
        f"{input_stats['contrast']['std']:.1f}"
    )
    print(
        f"- Color complexity: {input_stats['color_complexity']['mean']:.0f} ± "
        f"{input_stats['color_complexity']['std']:.0f}"
    )

    print(
        f"\nSprites (Target) Statistics "
        f"(sample: {target_stats['sample_size']}):"
    )
    print(
        f"- Brightness: {target_stats['brightness']['mean']:.1f} ± "
        f"{target_stats['brightness']['std']:.1f}"
    )
    print(
        f"- Contrast: {target_stats['contrast']['mean']:.1f} ± "
        f"{target_stats['contrast']['std']:.1f}"
    )
    print(
        f"- Color complexity: "
        f"{target_stats['color_complexity']['mean']:.0f} ± "
        f"{target_stats['color_complexity']['std']:.0f}"
    )

    # Create comparison visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        "Dataset Characteristics Comparison", fontsize=14, fontweight="bold"
    )

    categories = ["brightness", "contrast", "color_complexity"]
    titles = ["Brightness", "Contrast", "Color Complexity"]

    for idx, (category, title) in enumerate(zip(categories, titles)):
        input_mean = input_stats[category]["mean"]
        target_mean = target_stats[category]["mean"]

        axes[idx].bar(
            ["Artwork", "Sprites"],
            [input_mean, target_mean],
            color=["blue", "red"],
            alpha=0.7,
        )
        axes[idx].set_title(title)
        axes[idx].set_ylabel("Mean Value")
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Show sample comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Sample Comparison", fontsize=14, fontweight="bold")

    sample_input = Image.open(random.choice(input_files))
    sample_target = Image.open(random.choice(target_files))

    axes[0].imshow(sample_input)
    axes[0].set_title("Ken Sugimori Artwork\n(High resolution, rich colors)")
    axes[0].axis("off")

    axes[1].imshow(sample_target)
    axes[1].set_title("Black/White Sprite\n(Pixel art, simplified)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def analyze_sprites(sprite_dir: Path) -> None:
    """
    Analyze sprite files to understand their characteristics.

    Args:
        sprite_dir: Directory containing sprite files
    """
    sprite_files = list(sprite_dir.glob("*.png"))

    if len(sprite_files) == 0:
        print("No sprites to analyze!")
        return

    print(f"Analyzing {len(sprite_files)} Pokemon sprites...")

    # Lists to store sprite properties
    widths, heights, file_sizes = [], [], []
    has_transparency = []

    # Analyze first 20 files for speed
    sample_files = sprite_files[:20]

    for sprite_file in sample_files:
        try:
            img = Image.open(sprite_file)
            widths.append(img.width)
            heights.append(img.height)
            file_sizes.append(sprite_file.stat().st_size)
            has_transparency.append(img.mode in ("RGBA", "LA"))

        except Exception as e:
            print(f"Error analyzing {sprite_file}: {e}")

    if not widths:
        print("No valid sprite files found for analysis")
        return

    # Display analysis results
    print("\nSprite Analysis Results")
    print(f"Sample size: {len(widths)} sprites")
    print("Dimensions:")
    print(
        f"  Width: {np.min(widths)} - {np.max(widths)} pixels "
        f"(avg: {np.mean(widths):.1f})"
    )
    print(
        f"  Height: {np.min(heights)} - {np.max(heights)} pixels "
        f"(avg: {np.mean(heights):.1f})"
    )
    print(
        f"File sizes: {np.min(file_sizes)} - {np.max(file_sizes)} bytes "
        f"(avg: {np.mean(file_sizes):.1f})"
    )
    print(
        f"Transparency: {sum(has_transparency)}/"
        f"{len(has_transparency)} sprites have transparency"
    )

    # Create a visualization of sprite sizes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of widths and heights
    ax1.hist(widths, bins=10, alpha=0.7, label="Width", color="blue")
    ax1.hist(heights, bins=10, alpha=0.7, label="Height", color="red")
    ax1.set_xlabel("Pixels")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Sprite Dimensions")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Scatter plot of width vs height
    ax2.scatter(widths, heights, alpha=0.7, color="green")
    ax2.set_xlabel("Width (pixels)")
    ax2.set_ylabel("Height (pixels)")
    ax2.set_title("Sprite Width vs Height")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_artwork_sprite_pairs(
    dataset_dir: Path, num_pairs: int = 6
) -> int:
    """
    Visualize matched artwork-sprite pairs side by side.

    Args:
        dataset_dir: Directory containing the dataset
        num_pairs: Number of pairs to display

    Returns:
        Total number of matched pairs found
    """
    sprite_dir = dataset_dir / "black_white_sprites"
    artwork_dir = dataset_dir / "sugimori_artwork"

    if not (sprite_dir.exists() and artwork_dir.exists()):
        print("Required directories not found")
        return 0

    # Get sprite files and find matching artwork
    sprite_files = list(sprite_dir.glob("*.png"))
    matched_pairs = []

    for sprite_file in sprite_files:
        # Extract Pokemon ID from sprite filename
        if "_bw" in sprite_file.stem:
            pokemon_id = sprite_file.stem.split("_")[1]
            artwork_file = artwork_dir / f"pokemon_{pokemon_id}_artwork.png"

            if artwork_file.exists():
                try:
                    # Verify both files can be loaded
                    with Image.open(artwork_file) as artwork_img:
                        with Image.open(sprite_file) as sprite_img:
                            if (
                                artwork_img.width > 0
                                and artwork_img.height > 0
                                and sprite_img.width > 0
                                and sprite_img.height > 0
                            ):
                                matched_pairs.append(
                                    (artwork_file, sprite_file, pokemon_id)
                                )
                except Exception:
                    continue

    if not matched_pairs:
        print("No matched pairs found")
        return 0

    # Sample random pairs
    display_pairs = random.sample(
        matched_pairs, min(num_pairs, len(matched_pairs))
    )

    # Create visualization
    fig, axes = plt.subplots(2, num_pairs, figsize=(3 * num_pairs, 6))
    fig.suptitle(
        "Sugimori Artwork -> Black/White Sprite Training Pairs",
        fontsize=16,
        fontweight="bold",
    )

    for i, (artwork_file, sprite_file, pokemon_id) in enumerate(display_pairs):
        try:
            # Load and display artwork (top row)
            artwork_img = Image.open(artwork_file)
            axes[0, i].imshow(artwork_img)
            axes[0, i].set_title(f"Artwork\n#{pokemon_id}", fontsize=10)
            axes[0, i].axis("off")

            # Load and display sprite (bottom row)
            sprite_img = Image.open(sprite_file)
            axes[1, i].imshow(sprite_img)
            axes[1, i].set_title(f"B/W Sprite\n#{pokemon_id}", fontsize=10)
            axes[1, i].axis("off")

        except Exception as e:
            print(f"Error loading pair {pokemon_id}: {e}")
            for row in range(2):
                axes[row, i].text(0.5, 0.5, "Error", ha="center", va="center")
                axes[row, i].axis("off")

    plt.tight_layout()
    plt.show()

    print("\nMatched Pairs Analysis:")
    print(f"- Total matched pairs found: {len(matched_pairs)}")
    print(f"- Displayed sample: {len(display_pairs)} pairs")
    print("- Artwork resolution: Usually 475x475 pixels")
    print("- Sprite resolution: Usually 96x96 pixels")

    return len(matched_pairs)
