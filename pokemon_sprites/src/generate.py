#!/usr/bin/env python3
"""
Pokemon sprite generation inference script.

This module provides functionality to load trained models and generate
Pokemon sprites from artwork using the trained image-to-image translation
models.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from PIL import Image
from torchvision import transforms

# Add the src directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from config.settings import ModelConfig  # noqa: E402
from core.logging_config import (  # noqa: E402
    get_logger,
    initialize_project_logging,
)
from data.postprocessing import SpritePostProcessor  # noqa: E402
from models import create_model  # noqa: E402

logger = get_logger(__name__)


class PokemonSpriteGenerator:
    """Generator for creating Pokemon sprites from artwork."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        enable_postprocessing: bool = True,
    ):
        """
        Initialize the generator with a trained model and post-processing support.

        Args:
            model_path: Path to the trained model checkpoint.
            device: Device to use ("auto", "cpu", "cuda").
            enable_postprocessing: Whether to enable ARGB to P format post-processing.
        """
        self.model_path = Path(model_path)
        self.device = self._setup_device(device)
        self.enable_postprocessing = enable_postprocessing

        # Load checkpoint
        self.checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model_config = ModelConfig(**self.checkpoint["model_config"])

        # Create and load model
        self.model = self._load_model()
        self.model.eval()

        # Setup post-processing if enabled
        if self.enable_postprocessing:
            self.postprocessor = SpritePostProcessor()
            logger.info(
                "Post-processing enabled for ARGB to P format conversion"
            )
        else:
            self.postprocessor = None

        # Determine input channels and setup transforms accordingly
        input_channels = self.model_config.parameters.get("generator", {}).get(
            "input_channels", 4
        )
        self.use_rgba = input_channels == 4

        # Setup transforms for ARGB support
        image_size = self.model_config.parameters.get("image_size", 128)

        if self.use_rgba:
            # RGBA transforms - no normalization for alpha channel
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                ]
            )

            self.inverse_transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                ]
            )
        else:
            # RGB transforms with normalization
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

            self.inverse_transform = transforms.Compose(
                [
                    transforms.Normalize([-1, -1, -1], [2, 2, 2]),
                    transforms.ToPILImage(),
                ]
            )

        logger.info(f"Generator initialized with {self.model_config.name}")
        logger.info(f"Architecture: {self.model_config.architecture}")
        logger.info(
            f"Input channels: {input_channels} ({'RGBA' if self.use_rgba else 'RGB'})"
        )
        logger.info(f"Image size: {image_size}x{image_size}")
        logger.info(f"Device: {self.device}")

    def _setup_device(self, device: str) -> torch.device:
        """Setup compute device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_model(self) -> torch.nn.Module:
        """Load the trained model."""
        models = create_model(self.checkpoint["model_config"])

        if isinstance(models, dict):
            # For GAN architectures, use the generator
            if "generator" in models:
                model = models["generator"]
                state_dict = self.checkpoint["model_state"]["generator"]
            elif "generator_A2B" in models:
                model = models["generator_A2B"]
                state_dict = self.checkpoint["model_state"]["generator_A2B"]
            else:
                raise ValueError("No generator found in model dictionary")
        else:
            # For single models like U-Net
            model = models
            state_dict = self.checkpoint["model_state"]

        # Handle DataParallel
        if any(key.startswith("module.") for key in state_dict.keys()):
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith("module.") else k
                new_state_dict[name] = v
            state_dict = new_state_dict

        model.load_state_dict(state_dict)
        model.to(self.device)

        return model

    def generate_sprite(
        self, artwork_path: Union[str, Path], return_formats: list = ["rgba"]
    ) -> Dict[str, Any]:
        """
        Generate a sprite from artwork with ARGB support and post-processing.

        Args:
            artwork_path: Path to the input artwork image.
            return_formats: List of formats to return ("rgba", "p")

        Returns:
            Dictionary containing generated sprites in requested formats and analysis.
        """
        artwork_path = Path(artwork_path)

        if not artwork_path.exists():
            raise FileNotFoundError(f"Artwork file not found: {artwork_path}")

        # Load and preprocess image with ARGB support
        if self.use_rgba:
            artwork = Image.open(artwork_path).convert("RGBA")
        else:
            artwork = Image.open(artwork_path).convert("RGB")

        # Apply transforms and ensure tensor output
        if hasattr(self.transform, "__call__"):
            transformed = self.transform(artwork)
            if isinstance(transformed, Image.Image):
                # If transform returns PIL Image, convert to tensor manually
                import torchvision.transforms.functional as TF

                tensor = TF.to_tensor(transformed)
                # Normalize to [-1, 1]
                tensor = (tensor - 0.5) / 0.5
                input_tensor = tensor.unsqueeze(0).to(self.device)
            else:
                # Transform already returns tensor
                input_tensor = transformed.unsqueeze(0).to(self.device)
        else:
            # Fallback to basic tensor conversion
            import torchvision.transforms.functional as TF

            tensor = TF.to_tensor(artwork)
            # Normalize to [-1, 1]
            tensor = (tensor - 0.5) / 0.5
            input_tensor = tensor.unsqueeze(0).to(self.device)

        # Generate sprite
        with torch.no_grad():
            generated_tensor = self.model(input_tensor)

            if not self.use_rgba:
                generated_tensor = torch.clamp(generated_tensor, -1, 1)

        # Convert back to PIL Image
        generated_sprite = self.inverse_transform(
            generated_tensor.squeeze(0).cpu()
        )

        # Ensure RGBA format for post-processing compatibility
        if not isinstance(generated_sprite, Image.Image):
            # Convert tensor to PIL Image if needed
            if isinstance(generated_sprite, torch.Tensor):
                generated_sprite = transforms.ToPILImage()(generated_sprite)

        if generated_sprite.mode != "RGBA":
            generated_sprite = generated_sprite.convert("RGBA")

        result: Dict[str, Any] = {"artwork_path": str(artwork_path)}

        # Apply post-processing if enabled
        if self.enable_postprocessing and self.postprocessor:
            processed_results = self.postprocessor.process_single_sprite(
                generated_sprite, return_formats=return_formats
            )
            result.update(processed_results)
        else:
            # Return RGBA sprite without post-processing
            if "rgba" in return_formats:
                result["rgba"] = generated_sprite

        return result

    def generate_batch(
        self,
        artwork_paths: list,
        output_dir: Union[str, Path],
        return_formats: list = ["rgba", "p"],
    ) -> list:
        """
        Generate sprites for multiple artworks with post-processing support.

        Args:
            artwork_paths: List of paths to artwork images.
            output_dir: Directory to save generated sprites.
            return_formats: List of formats to return ("rgba", "p")

        Returns:
            List of output paths for generated sprites.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_paths = []

        for i, artwork_path in enumerate(artwork_paths):
            try:
                artwork_path = Path(artwork_path)
                sprite_results = self.generate_sprite(
                    artwork_path, return_formats
                )

                # Save RGBA sprite
                if "rgba" in sprite_results:
                    rgba_output_path = (
                        output_dir
                        / f"{artwork_path.stem}_generated_sprite.png"
                    )
                    sprite_results["rgba"].save(rgba_output_path)
                    output_paths.append(rgba_output_path)

                # Save P format sprite if available
                if "p" in sprite_results:
                    p_output_path = (
                        output_dir
                        / f"{artwork_path.stem}_generated_sprite_p.png"
                    )
                    sprite_results["p"].save(p_output_path)

                logger.info(
                    f"Generated sprite {i+1}/{len(artwork_paths)}: "
                    f"Formats: {list(sprite_results.keys())}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to generate sprite for {artwork_path}: {e}"
                )
                output_paths.append(None)

        return output_paths

    def create_comparison(
        self,
        artwork_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Image.Image:
        """
        Create a side-by-side comparison of artwork and generated sprite.

        Args:
            artwork_path: Path to the input artwork.
            output_path: Optional path to save the comparison image.

        Returns:
            Comparison image as PIL Image.
        """
        artwork_path = Path(artwork_path)

        # Load original artwork
        artwork = Image.open(artwork_path).convert("RGB")

        # Generate sprite
        sprite_result = self.generate_sprite(artwork_path)

        # Extract PIL Image from result (use final processed image if available, otherwise base)
        if self.enable_postprocessing and "processed" in sprite_result:
            sprite = sprite_result["processed"]
        else:
            sprite = sprite_result["base"]

        # Resize images to same height for comparison
        target_height = 256
        artwork_width = int(artwork.width * target_height / artwork.height)
        sprite_width = int(sprite.width * target_height / sprite.height)

        artwork_resized = artwork.resize(
            (artwork_width, target_height), Image.Resampling.LANCZOS
        )
        sprite_resized = sprite.resize(
            (sprite_width, target_height), Image.Resampling.NEAREST
        )  # Pixel art should use nearest neighbor

        # Create comparison image
        total_width = artwork_width + sprite_width + 20  # 20px padding
        comparison = Image.new("RGB", (total_width, target_height), "white")

        # Paste images
        comparison.paste(artwork_resized, (0, 0))
        comparison.paste(sprite_resized, (artwork_width + 20, 0))

        # Save if output path provided
        if output_path:
            comparison.save(output_path)
            logger.info(f"Comparison saved: {output_path}")

        return comparison


def main():
    """Main generation function."""
    parser = argparse.ArgumentParser(
        description="Generate Pokemon sprites from artwork"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input artwork image or directory",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for inference",
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Create side-by-side comparison images",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing multiple images",
    )

    args = parser.parse_args()

    # Initialize logging
    initialize_project_logging(
        project_name="pokemon_sprites_inference",
        log_level="INFO",
        model_name="generator",
        enable_file_logging=False,
    )

    logger.info("=" * 60)
    logger.info("POKEMON SPRITE GENERATION")
    logger.info("=" * 60)

    try:
        # Initialize generator
        logger.info(f"Loading model from: {args.model}")
        generator = PokemonSpriteGenerator(args.model, args.device)

        # Setup input and output paths
        input_path = Path(args.input)
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        if input_path.is_file():
            # Single image
            logger.info(f"Processing single image: {input_path}")

            if args.comparison:
                comparison_path = (
                    output_path / f"{input_path.stem}_comparison.png"
                )
                generator.create_comparison(input_path, comparison_path)
            else:
                sprite_result = generator.generate_sprite(input_path)
                sprite_path = output_path / f"{input_path.stem}_sprite.png"

                # Save the appropriate image based on post-processing setting
                if (
                    generator.enable_postprocessing
                    and "processed" in sprite_result
                ):
                    sprite_result["processed"].save(sprite_path)
                    logger.info(f"Generated processed sprite: {sprite_path}")
                else:
                    sprite_result["base"].save(sprite_path)
                    logger.info(f"Generated base sprite: {sprite_path}")

                # Also save other formats if available
                if "palette" in sprite_result:
                    palette_path = (
                        output_path / f"{input_path.stem}_palette.png"
                    )
                    sprite_result["palette"].save(palette_path)
                    logger.info(f"Generated palette sprite: {palette_path}")

        elif input_path.is_dir():
            # Multiple images
            logger.info(f"Processing directory: {input_path}")

            # Find image files
            image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
            image_files = [
                f
                for f in input_path.iterdir()
                if f.suffix.lower() in image_extensions
            ]

            if not image_files:
                logger.error(f"No image files found in {input_path}")
                return

            logger.info(f"Found {len(image_files)} image files")

            if args.comparison:
                for img_file in image_files:
                    comparison_path = (
                        output_path / f"{img_file.stem}_comparison.png"
                    )
                    generator.create_comparison(img_file, comparison_path)
            else:
                output_paths = generator.generate_batch(
                    image_files, output_path
                )
                successful = sum(1 for p in output_paths if p is not None)
                logger.info(
                    f"Successfully generated {successful}/"
                    f"{len(image_files)} sprites"
                )

        else:
            logger.error(f"Input path does not exist: {input_path}")
            return

        logger.info("Generation completed successfully!")

    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
