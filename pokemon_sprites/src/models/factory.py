"""
Model factory function and high-level model creation interface.

This module provides the main create_model function that instantiates
different model architectures based on configuration parameters.
"""

import sys
from pathlib import Path
from typing import Dict, Union

import torch.nn as nn

from .discriminators import CycleGANDiscriminator, Pix2PixDiscriminator
from .generators import (
    CycleGANGenerator,
    Pix2PixGenerator,
    PretrainedBackboneGenerator,
    UNet,
    ViTCLIPGenerator,
)

# Import after path modification to avoid issues
try:
    from core.logging_config import get_logger
except ImportError:
    import logging

    def get_logger(name):
        return logging.getLogger(name)


# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


logger = get_logger(__name__)


def create_model(config) -> Union[nn.Module, Dict[str, nn.Module]]:
    """
    Create a model based on configuration.

    Args:
        config: Model configuration (dict or ModelConfig object).

    Returns:
        PyTorch model instance or dictionary of models for multi-model
        architectures.
    """
    # Handle both dict and ModelConfig objects
    if hasattr(config, "architecture"):
        architecture = config.architecture
        params = config.parameters
    else:
        architecture = config.get("architecture", "unet")
        params = config.get("parameters", {})

    if architecture == "unet":
        return UNet(
            input_channels=params.get("input_channels", 3),
            output_channels=params.get("output_channels", 3),
            features=params.get("features", [64, 128, 256, 512]),
            dropout=params.get("dropout", 0.1),
            attention=params.get("attention", False),
        )

    elif architecture == "pix2pix":
        generator = Pix2PixGenerator(
            input_channels=params.get("generator", {}).get(
                "input_channels", 3
            ),
            output_channels=params.get("generator", {}).get(
                "output_channels", 3
            ),
            ngf=params.get("generator", {}).get("ngf", 64),
            n_blocks=params.get("generator", {}).get("n_blocks", 6),
            norm_layer=params.get("generator", {}).get("norm_layer", "batch"),
            dropout=params.get("generator", {}).get("dropout", 0.5),
        )

        discriminator = Pix2PixDiscriminator(
            input_channels=params.get("discriminator", {}).get(
                "input_channels", 6
            ),
            ndf=params.get("discriminator", {}).get("ndf", 64),
            n_layers=params.get("discriminator", {}).get("n_layers", 3),
            norm_layer=params.get("discriminator", {}).get(
                "norm_layer", "batch"
            ),
        )

        return {"generator": generator, "discriminator": discriminator}

    elif architecture == "pix2pix_pretrained":
        # Pretrained backbone generator with regular discriminator
        generator = PretrainedBackboneGenerator(
            backbone=params.get("backbone", "resnet50"),
            freeze_backbone=params.get("freeze_backbone", True),
            output_channels=params.get("generator", {}).get(
                "output_channels", 3
            ),
            decoder_features=params.get("generator", {}).get(
                "decoder_features", [512, 256, 128, 64]
            ),
            dropout=params.get("generator", {}).get("dropout", 0.3),
        )

        discriminator = Pix2PixDiscriminator(
            input_channels=params.get("discriminator", {}).get(
                "input_channels", 6
            ),
            ndf=params.get("discriminator", {}).get("ndf", 64),
            n_layers=params.get("discriminator", {}).get("n_layers", 3),
            norm_layer=params.get("discriminator", {}).get(
                "norm_layer", "instance"
            ),
        )

        return {"generator": generator, "discriminator": discriminator}

    elif architecture == "pix2pix_vit":
        generator = ViTCLIPGenerator(
            vit_model=params.get("generator", {}).get(
                "vit_model", "vit_large_patch16_224"
            ),
            use_clip=params.get("generator", {}).get("use_clip", True),
            output_channels=params.get("generator", {}).get(
                "output_channels", 3
            ),
            decoder_features=params.get("generator", {}).get(
                "decoder_features", [1024, 512, 256, 128]
            ),
            dropout=params.get("generator", {}).get("dropout", 0.2),
        )

        discriminator = Pix2PixDiscriminator(
            input_channels=params.get("discriminator", {}).get(
                "input_channels", 6
            ),
            ndf=params.get("discriminator", {}).get("ndf", 64),
            n_layers=params.get("discriminator", {}).get("n_layers", 4),
            norm_layer=params.get("discriminator", {}).get(
                "norm_layer", "instance"
            ),
        )

        return {"generator": generator, "discriminator": discriminator}

    elif architecture == "cyclegan":
        generator_A2B = CycleGANGenerator(
            input_channels=params.get("generator", {}).get(
                "input_channels", 3
            ),
            output_channels=params.get("generator", {}).get(
                "output_channels", 3
            ),
            ngf=params.get("generator", {}).get("ngf", 64),
            n_blocks=params.get("generator", {}).get("n_blocks", 9),
            norm_layer=params.get("generator", {}).get(
                "norm_layer", "instance"
            ),
        )

        generator_B2A = CycleGANGenerator(
            input_channels=params.get("generator", {}).get(
                "input_channels", 3
            ),
            output_channels=params.get("generator", {}).get(
                "output_channels", 3
            ),
            ngf=params.get("generator", {}).get("ngf", 64),
            n_blocks=params.get("generator", {}).get("n_blocks", 9),
            norm_layer=params.get("generator", {}).get(
                "norm_layer", "instance"
            ),
        )

        discriminator_A = CycleGANDiscriminator(
            input_channels=params.get("discriminator", {}).get(
                "input_channels", 3
            ),
            ndf=params.get("discriminator", {}).get("ndf", 64),
            n_layers=params.get("discriminator", {}).get("n_layers", 3),
            norm_layer=params.get("discriminator", {}).get(
                "norm_layer", "instance"
            ),
        )

        discriminator_B = CycleGANDiscriminator(
            input_channels=params.get("discriminator", {}).get(
                "input_channels", 3
            ),
            ndf=params.get("discriminator", {}).get("ndf", 64),
            n_layers=params.get("discriminator", {}).get("n_layers", 3),
            norm_layer=params.get("discriminator", {}).get(
                "norm_layer", "instance"
            ),
        )

        return {
            "generator_A2B": generator_A2B,
            "generator_B2A": generator_B2A,
            "discriminator_A": discriminator_A,
            "discriminator_B": discriminator_B,
        }

    else:
        raise ValueError(f"Unknown architecture: {architecture}")
