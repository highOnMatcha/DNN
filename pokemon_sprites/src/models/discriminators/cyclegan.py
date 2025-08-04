"""
CycleGAN discriminator implementation.

This module implements the PatchGAN discriminator for CycleGAN
cycle-consistent adversarial networks.
"""

import sys
from pathlib import Path

import torch.nn as nn

from ..components import ConvBlock

try:
    from core.logging_config import get_logger
except ImportError:
    import logging

    def get_logger(name):
        return logging.getLogger(name)


# Add src to path - must be before other local imports
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)


logger = get_logger(__name__)


class CycleGANDiscriminator(nn.Module):
    """CycleGAN PatchGAN Discriminator."""

    def __init__(
        self,
        input_channels: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer: str = "instance",
    ):
        super().__init__()

        model_layers = [
            ConvBlock(input_channels, ndf, 4, 2, 1, "none", "leaky_relu", 0.0)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            model_layers.append(
                ConvBlock(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    4,
                    2,
                    1,
                    norm_layer,
                    "leaky_relu",
                    0.0,
                )
            )

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        model_layers.append(
            ConvBlock(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                4,
                1,
                1,
                norm_layer,
                "leaky_relu",
                0.0,
            )
        )

        # Final prediction layer
        final_conv = nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1)

        # Combine all layers
        all_layers = model_layers + [final_conv]
        self.model = nn.Sequential(*all_layers)

    def forward(self, x):
        return self.model(x)
