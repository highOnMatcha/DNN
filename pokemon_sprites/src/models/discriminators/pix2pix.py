"""
Pix2Pix discriminator implementation.

This module implements the PatchGAN discriminator for Pix2Pix
image-to-image translation with improved training stability.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add src to path - must be before other local imports
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from core.logging_config import get_logger
except ImportError:
    import logging

    def get_logger(name):
        return logging.getLogger(name)


logger = get_logger(__name__)


class Pix2PixDiscriminator(nn.Module):
    """Balanced PatchGAN Discriminator for better training stability."""

    def __init__(
        self,
        input_channels: int = 6,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer: str = "batch",
        use_spectral_norm: bool = False,
    ):
        super().__init__()

        layers = []

        # First layer - no normalization
        if use_spectral_norm:
            layers.append(
                nn.utils.spectral_norm(nn.Conv2d(input_channels, ndf, 4, 2, 1))
            )
        else:
            layers.append(nn.Conv2d(input_channels, ndf, 4, 2, 1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Subsequent layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)

            conv_layer = nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1)
            if use_spectral_norm:
                conv_layer = nn.utils.spectral_norm(conv_layer)
            layers.append(conv_layer)

            # Add normalization
            if norm_layer == "batch":
                layers.append(nn.BatchNorm2d(ndf * nf_mult))
            elif norm_layer == "instance":
                layers.append(nn.InstanceNorm2d(ndf * nf_mult))

            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Final layers
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)

        conv_layer = nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1)
        if use_spectral_norm:
            conv_layer = nn.utils.spectral_norm(conv_layer)
        layers.append(conv_layer)

        if norm_layer == "batch":
            layers.append(nn.BatchNorm2d(ndf * nf_mult))
        elif norm_layer == "instance":
            layers.append(nn.InstanceNorm2d(ndf * nf_mult))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Final prediction layer
        final_layer = nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1)
        if use_spectral_norm:
            final_layer = nn.utils.spectral_norm(final_layer)
        layers.append(final_layer)

        self.model = nn.Sequential(*layers)

    def forward(self, input_img, target_img):
        x = torch.cat([input_img, target_img], dim=1)
        return self.model(x)
