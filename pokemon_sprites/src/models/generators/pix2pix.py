"""
Pix2Pix generator implementation.

This module implements the U-Net based generator for Pix2Pix
image-to-image translation with skip connections.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components import ConvBlock, ResBlock

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


class Pix2PixGenerator(nn.Module):
    """Pix2Pix Generator (U-Net with skip connections)."""

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        ngf: int = 64,
        n_blocks: int = 6,
        norm_layer: str = "batch",
        dropout: float = 0.5,
    ):
        super().__init__()

        # Encoder layers with proper downsampling
        self.encoder = nn.ModuleList(
            [
                ConvBlock(
                    input_channels, ngf, 4, 2, 1, "none", "leaky_relu", 0.0
                ),  # Input -> ngf
                ConvBlock(
                    ngf, ngf * 2, 4, 2, 1, norm_layer, "leaky_relu", 0.0
                ),  # ngf -> ngf*2
                ConvBlock(
                    ngf * 2, ngf * 4, 4, 2, 1, norm_layer, "leaky_relu", 0.0
                ),  # ngf*2 -> ngf*4
                ConvBlock(
                    ngf * 4, ngf * 8, 4, 2, 1, norm_layer, "leaky_relu", 0.0
                ),  # ngf*4 -> ngf*8
            ]
        )

        # Bottleneck (residual blocks instead of downsampling)
        # Use residual blocks to add depth without changing spatial dimensions
        bottleneck_layers = []
        for _ in range(max(0, n_blocks - 4)):
            bottleneck_layers.append(ResBlock(ngf * 8, norm_layer, dropout))
        self.bottleneck = nn.ModuleList(bottleneck_layers)

        # Decoder with transposed convolutions
        self.decoder = nn.ModuleList(
            [
                # Upsample from ngf*8 to ngf*4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
                ConvBlock(
                    ngf * 8, ngf * 4, 3, 1, 1, norm_layer, "relu", dropout
                ),
                # Upsample from ngf*4 to ngf*2
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
                ConvBlock(ngf * 4, ngf * 2, 3, 1, 1, norm_layer, "relu", 0.0),
                # Upsample from ngf*2 to ngf
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
                ConvBlock(ngf * 2, ngf, 3, 1, 1, norm_layer, "relu", 0.0),
                # Final layer to output
                nn.ConvTranspose2d(ngf, output_channels, 4, 2, 1),
                nn.Tanh(),
            ]
        )

    def forward(self, x):
        # Store skip connections from encoder
        skips = []

        # Encoder
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)

        # Bottleneck (residual blocks - do not change spatial dimensions)
        for layer in self.bottleneck:
            x = layer(x)

        # Decoder with skip connections
        # Skip connections are used in reverse order (excluding the last
        # encoder output)
        skips = skips[::-1][
            1:
        ]  # Reverse and skip the last one (which is the bottleneck input)

        skip_idx = 0
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.ConvTranspose2d):
                x = layer(x)
                # Add skip connection if available
                if skip_idx < len(skips):
                    skip = skips[skip_idx]
                    # Ensure spatial dimensions match
                    if x.shape[2:] != skip.shape[2:]:
                        x = F.interpolate(
                            x,
                            size=skip.shape[2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    x = torch.cat([x, skip], dim=1)
                    skip_idx += 1
            else:
                x = layer(x)

        return x
