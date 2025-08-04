"""
Basic building blocks for neural network architectures.

This module provides reusable convolutional and residual blocks
used across different model architectures.
"""

import torch.nn as nn


class ConvBlock(nn.Module):
    """Basic convolutional block with normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm_type: str = "batch",
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )

        # Normalization
        if norm_type == "batch":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == "instance":
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm_type == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")

        # Activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unknown activation type: {activation}")

        # Dropout
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class ResBlock(nn.Module):
    """Residual block for generator networks."""

    def __init__(
        self, channels: int, norm_type: str = "batch", dropout: float = 0.0
    ):
        super().__init__()

        self.block = nn.Sequential(
            ConvBlock(channels, channels, 3, 1, 1, norm_type, "relu", dropout),
            ConvBlock(channels, channels, 3, 1, 1, norm_type, "none", 0.0),
        )

    def forward(self, x):
        return x + self.block(x)
