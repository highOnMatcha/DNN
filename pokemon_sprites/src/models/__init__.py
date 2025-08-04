"""
Neural network models for Pokemon sprite generation.

This package provides a comprehensive collection of model architectures
for image-to-image translation, specifically optimized for Pokemon sprite
generation tasks. The models are organized into logical components:

- generators/: Generator architectures (U-Net, Pix2Pix, CycleGAN, etc.)
- discriminators/: Discriminator architectures for adversarial training
- components/: Reusable building blocks (attention, blocks, encoders)
- utils: Utility functions for model analysis and parameter counting
- factory: Model creation interface

Example usage:
    from models import create_model

    config = {
        "architecture": "pix2pix",
        "parameters": {
            "generator": {"ngf": 64, "n_blocks": 6},
            "discriminator": {"ndf": 64, "n_layers": 3}
        }
    }
    models = create_model(config)
"""

# Import all model classes for direct access
from .components import (
    AttentionBlock,
    ConvBlock,
    ResBlock,
    SelfAttention,
    TransformerBottleneck,
    UNetDecoder,
    UNetEncoder,
)
from .discriminators import CycleGANDiscriminator, Pix2PixDiscriminator

# Import main factory function
from .factory import create_model
from .generators import (
    CycleGANGenerator,
    Pix2PixGenerator,
    PretrainedBackboneGenerator,
    UNet,
    ViTCLIPGenerator,
)

# Import utility functions
from .utils import (
    analyze_model_architectures,
    count_parameters,
    count_total_parameters,
)

__all__ = [
    # Factory function
    "create_model",
    # Utility functions
    "analyze_model_architectures",
    "count_parameters",
    "count_total_parameters",
    # Components
    "AttentionBlock",
    "ConvBlock",
    "ResBlock",
    "SelfAttention",
    "TransformerBottleneck",
    "UNetDecoder",
    "UNetEncoder",
    # Discriminators
    "CycleGANDiscriminator",
    "Pix2PixDiscriminator",
    # Generators
    "CycleGANGenerator",
    "Pix2PixGenerator",
    "PretrainedBackboneGenerator",
    "UNet",
    "ViTCLIPGenerator",
]
