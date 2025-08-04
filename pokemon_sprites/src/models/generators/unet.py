"""
U-Net generator implementation.

This module implements a standard U-Net architecture for
image-to-image translation tasks.
"""

import sys
from pathlib import Path
from typing import List, Optional

import torch.nn as nn

from ..components import AttentionBlock, UNetDecoder, UNetEncoder

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


class UNet(nn.Module):
    """U-Net model for image-to-image translation."""

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        features: Optional[List[int]] = None,
        dropout: float = 0.1,
        attention: bool = False,
    ):
        super().__init__()

        if features is None:
            features = [64, 128, 256, 512]

        self.encoder = UNetEncoder(input_channels, features)
        self.decoder = UNetDecoder(features, output_channels)

        # Optional attention in bottleneck
        if attention:
            self.attention = AttentionBlock(features[-1])
        else:
            self.attention = nn.Identity()

    def forward(self, x):
        # Encoder
        x, skip_connections = self.encoder(x)

        # Bottleneck with optional attention
        x = self.attention(x)

        # Decoder
        x = self.decoder(x, skip_connections)

        return x
