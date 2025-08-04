"""
ViT/CLIP-based generator optimized for pixel art generation.

This module implements a simplified CNN-based generator optimized
for pixel art generation with transformer attention in the bottleneck.
"""

import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from ..components import ResBlock, SelfAttention, TransformerBottleneck

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


class ViTCLIPGenerator(nn.Module):
    """Simplified CNN-based generator optimized for pixel art generation."""

    def __init__(
        self,
        vit_model: str = "vit_base_patch16_224",
        use_clip: bool = False,
        output_channels: int = 3,
        decoder_features: Optional[List[int]] = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if decoder_features is None:
            decoder_features = [256, 128, 64, 32]

        logger.info("Creating simplified pixel-art optimized generator...")

        # Simple but effective encoder for pixel art
        self.encoder = nn.ModuleList(
            [
                # 64x64 -> 32x32
                nn.Sequential(
                    nn.Conv2d(3, decoder_features[3], 4, 2, 1),
                    nn.BatchNorm2d(decoder_features[3]),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                # 32x32 -> 16x16
                nn.Sequential(
                    nn.Conv2d(
                        decoder_features[3], decoder_features[2], 4, 2, 1
                    ),
                    nn.BatchNorm2d(decoder_features[2]),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(dropout * 0.5),
                ),
                # 16x16 -> 8x8
                nn.Sequential(
                    nn.Conv2d(
                        decoder_features[2], decoder_features[1], 4, 2, 1
                    ),
                    nn.BatchNorm2d(decoder_features[1]),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(dropout),
                ),
                # 8x8 -> 4x4
                nn.Sequential(
                    nn.Conv2d(
                        decoder_features[1], decoder_features[0], 4, 2, 1
                    ),
                    nn.BatchNorm2d(decoder_features[0]),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(dropout),
                ),
            ]
        )

        # Enhanced bottleneck with transformer attention for global feature
        # learning
        self.bottleneck = nn.Sequential(
            ResBlock(decoder_features[0], "batch", dropout * 0.5),
            TransformerBottleneck(
                channels=decoder_features[0],
                num_heads=8,
                num_layers=4,
                dropout=dropout * 0.3,
                spatial_size=4,  # 4x4 spatial size at bottleneck
            ),
            nn.Dropout2d(dropout * 0.2),
            ResBlock(decoder_features[0], "batch", dropout * 0.3),
            SelfAttention(decoder_features[0]),
            nn.Dropout2d(dropout * 0.1),
            ResBlock(decoder_features[0], "batch", dropout * 0.2),
        )

        # Decoder with skip connections
        self.decoder = nn.ModuleList(
            [
                # 4x4 -> 8x8
                nn.Sequential(
                    nn.ConvTranspose2d(
                        decoder_features[0], decoder_features[1], 4, 2, 1
                    ),
                    nn.BatchNorm2d(decoder_features[1]),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout * 0.5),
                ),
                # 8x8 -> 16x16 (with skip connection)
                nn.Sequential(
                    nn.ConvTranspose2d(
                        decoder_features[1] * 2, decoder_features[2], 4, 2, 1
                    ),
                    nn.BatchNorm2d(decoder_features[2]),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout * 0.3),
                ),
                # 16x16 -> 32x32 (with skip connection)
                nn.Sequential(
                    nn.ConvTranspose2d(
                        decoder_features[2] * 2, decoder_features[3], 4, 2, 1
                    ),
                    nn.BatchNorm2d(decoder_features[3]),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout * 0.1),
                ),
                # 32x32 -> 64x64 (with skip connection)
                nn.Sequential(
                    nn.ConvTranspose2d(
                        decoder_features[3] * 2, output_channels, 4, 2, 1
                    ),
                    nn.Tanh(),
                ),
            ]
        )

        logger.info(
            f"Pixel-art Generator created with "
            f"{len(self.encoder)} encoder layers"
        )
        logger.info(f"  Encoder layers: {len(self.encoder)}")
        logger.info(f"  Decoder layers: {len(self.decoder)}")
        logger.info(f"  Features: {decoder_features}")
        logger.info(f"  Dropout: {dropout}")

    def forward(self, x):
        # Store skip connections
        skip_connections = []

        # Encoder
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections (reverse order, excluding last)
        skip_connections = skip_connections[:-1][::-1]

        for i, layer in enumerate(self.decoder):
            x = layer(x)
            # Add skip connection for all but the last layer
            if i < len(self.decoder) - 1 and i < len(skip_connections):
                skip = skip_connections[i]
                if x.shape[2:] == skip.shape[2:]:
                    x = torch.cat([x, skip], dim=1)

        return x
