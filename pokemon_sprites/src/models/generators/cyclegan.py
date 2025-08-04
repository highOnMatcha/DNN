"""
CycleGAN generator implementation.

This module implements the ResNet-based generator for CycleGAN
cycle-consistent adversarial networks.
"""

import sys
from pathlib import Path

import torch.nn as nn

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


class CycleGANGenerator(nn.Module):
    """CycleGAN Generator with ResNet blocks."""

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        ngf: int = 64,
        n_blocks: int = 9,
        norm_layer: str = "instance",
    ):
        super().__init__()

        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            ConvBlock(input_channels, ngf, 7, 1, 0, norm_layer, "relu", 0.0),
        ]

        # Downsampling
        for i in range(2):
            mult = 2**i
            model.append(
                ConvBlock(
                    ngf * mult,
                    ngf * mult * 2,
                    3,
                    2,
                    1,
                    norm_layer,
                    "relu",
                    0.0,
                )
            )

        # ResNet blocks
        mult = 2**2
        for i in range(n_blocks):
            model.append(ResBlock(ngf * mult, norm_layer))

        # Upsampling
        for i in range(2):
            mult = 2 ** (2 - i)
            model.append(
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), 3, 2, 1, 1)
            )
            model.append(
                ConvBlock(
                    int(ngf * mult / 2),
                    int(ngf * mult / 2),
                    3,
                    1,
                    1,
                    norm_layer,
                    "relu",
                    0.0,
                )
            )

        # Output layer
        model.extend(
            [
                nn.ReflectionPad2d(3),
                nn.Conv2d(ngf, output_channels, 7, 1, 0),
                nn.Tanh(),
            ]
        )

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
