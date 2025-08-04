"""
Common building blocks and components for neural network models.

This module provides reusable components like attention mechanisms,
convolutional blocks, and residual blocks used across different
model architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Self-attention mechanism for better feature learning."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Generate query, key, value
        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, height * width)
        v = self.value(x).view(batch_size, -1, height * width)

        # Attention
        attention = torch.bmm(q, k)
        attention = F.softmax(attention, dim=2)

        # Apply attention
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        # Residual connection
        return self.gamma * out + x


class AttentionBlock(nn.Module):
    """Self-attention block for improved feature learning."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Compute attention
        query = (
            self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        )
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)

        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        return self.gamma * out + x


class TransformerBottleneck(nn.Module):
    """Transformer-based bottleneck for enhanced global feature learning."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        spatial_size: int = 4,
    ):
        super().__init__()

        self.spatial_size = spatial_size
        self.channels = channels

        # Position encoding for spatial locations
        self.pos_embedding = nn.Parameter(
            torch.randn(1, spatial_size * spatial_size, channels) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=channels * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Flatten spatial dimensions
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Add positional encoding
        x_flat = x_flat + self.pos_embedding

        # Apply transformer
        x_out = self.transformer(x_flat)

        # Reshape back to spatial
        x_out = x_out.transpose(1, 2).view(batch_size, channels, height, width)

        return x + x_out
