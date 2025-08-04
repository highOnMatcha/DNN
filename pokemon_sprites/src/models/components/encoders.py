"""
Encoder and decoder components for U-Net architecture.

This module provides the encoder and decoder building blocks
specifically designed for U-Net style architectures.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock


class UNetEncoder(nn.Module):
    """U-Net encoder (downsampling path)."""

    def __init__(self, input_channels: int, features: List[int]):
        super().__init__()

        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input layer
        self.downs.append(
            ConvBlock(input_channels, features[0], norm_type="none")
        )

        # Downsampling layers
        for i in range(len(features) - 1):
            self.downs.append(
                nn.Sequential(
                    ConvBlock(features[i], features[i + 1]),
                    ConvBlock(features[i + 1], features[i + 1]),
                )
            )

    def forward(self, x):
        skip_connections = []

        for i, down in enumerate(self.downs):
            x = down(x)
            if i < len(self.downs) - 1:  # Do not pool the last layer
                skip_connections.append(x)
                x = self.pool(x)

        return x, skip_connections


class UNetDecoder(nn.Module):
    """U-Net decoder (upsampling path)."""

    def __init__(self, features: List[int], output_channels: int):
        super().__init__()

        self.ups = nn.ModuleList()

        # Upsampling layers
        for i in range(len(features) - 1, 0, -1):
            self.ups.append(
                nn.ConvTranspose2d(features[i], features[i - 1], 2, 2)
            )
            self.ups.append(
                nn.Sequential(
                    ConvBlock(features[i], features[i - 1]),
                    ConvBlock(features[i - 1], features[i - 1]),
                )
            )

        # Output layer
        self.final_conv = nn.Conv2d(features[0], output_channels, 1)

    def forward(self, x, skip_connections):
        skip_connections = skip_connections[::-1]  # Reverse order

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)  # Upsampling
            if i // 2 < len(skip_connections):
                skip = skip_connections[i // 2]
                # Ensure spatial dimensions match
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(
                        x,
                        size=skip.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)  # Convolution layers

        return torch.tanh(self.final_conv(x))
