"""
Model architectures for Pokemon sprite generation.

This module contains implementations of different image-to-image translation
architectures including Pix2Pix, U-Net, CycleGAN, and diffusion models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


# ============================================================================
# Utility layers and functions
# ============================================================================

class ConvBlock(nn.Module):
    """Basic convolutional block with normalization and activation."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, norm_type: str = "batch",
                 activation: str = "relu", dropout: float = 0.0):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
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
    
    def __init__(self, channels: int, norm_type: str = "batch", dropout: float = 0.0):
        super().__init__()
        
        self.block = nn.Sequential(
            ConvBlock(channels, channels, 3, 1, 1, norm_type, "relu", dropout),
            ConvBlock(channels, channels, 3, 1, 1, norm_type, "none", 0.0)
        )
    
    def forward(self, x):
        return x + self.block(x)


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
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)
        
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        return self.gamma * out + x


# ============================================================================
# U-Net Architecture
# ============================================================================

class UNetEncoder(nn.Module):
    """U-Net encoder (downsampling path)."""
    
    def __init__(self, input_channels: int, features: List[int]):
        super().__init__()
        
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Input layer
        self.downs.append(ConvBlock(input_channels, features[0], norm_type="none"))
        
        # Downsampling layers
        for i in range(len(features) - 1):
            self.downs.append(nn.Sequential(
                ConvBlock(features[i], features[i+1]),
                ConvBlock(features[i+1], features[i+1])
            ))
    
    def forward(self, x):
        skip_connections = []
        
        for i, down in enumerate(self.downs):
            x = down(x)
            if i < len(self.downs) - 1:  # Don't pool the last layer
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
            self.ups.append(nn.ConvTranspose2d(features[i], features[i-1], 2, 2))
            self.ups.append(nn.Sequential(
                ConvBlock(features[i], features[i-1]),
                ConvBlock(features[i-1], features[i-1])
            ))
        
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
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)  # Convolution layers
        
        return torch.tanh(self.final_conv(x))


class UNet(nn.Module):
    """U-Net model for image-to-image translation."""
    
    def __init__(self, input_channels: int = 3, output_channels: int = 3,
                 features: List[int] = [64, 128, 256, 512], dropout: float = 0.1,
                 attention: bool = False):
        super().__init__()
        
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


# ============================================================================
# Pix2Pix Generator and Discriminator
# ============================================================================

class Pix2PixGenerator(nn.Module):
    """Pix2Pix Generator (U-Net with skip connections)."""
    
    def __init__(self, input_channels: int = 3, output_channels: int = 3,
                 ngf: int = 64, n_blocks: int = 6, norm_layer: str = "batch",
                 dropout: float = 0.5):
        super().__init__()
        
        # Encoder layers with proper downsampling
        self.encoder = nn.ModuleList([
            ConvBlock(input_channels, ngf, 4, 2, 1, "none", "leaky_relu", 0.0),  # Input -> ngf
            ConvBlock(ngf, ngf * 2, 4, 2, 1, norm_layer, "leaky_relu", 0.0),    # ngf -> ngf*2
            ConvBlock(ngf * 2, ngf * 4, 4, 2, 1, norm_layer, "leaky_relu", 0.0), # ngf*2 -> ngf*4
            ConvBlock(ngf * 4, ngf * 8, 4, 2, 1, norm_layer, "leaky_relu", 0.0), # ngf*4 -> ngf*8
        ])
        
        # Bottleneck (residual blocks instead of downsampling)
        # Use residual blocks to add depth without changing spatial dimensions
        bottleneck_layers = []
        for _ in range(max(0, n_blocks - 4)):
            bottleneck_layers.append(
                ResBlock(ngf * 8, norm_layer, dropout)
            )
        self.bottleneck = nn.ModuleList(bottleneck_layers)
        
        # Decoder with transposed convolutions
        self.decoder = nn.ModuleList([
            # Upsample from ngf*8 to ngf*4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            ConvBlock(ngf * 8, ngf * 4, 3, 1, 1, norm_layer, "relu", dropout),
            
            # Upsample from ngf*4 to ngf*2
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            ConvBlock(ngf * 4, ngf * 2, 3, 1, 1, norm_layer, "relu", 0.0),
            
            # Upsample from ngf*2 to ngf
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            ConvBlock(ngf * 2, ngf, 3, 1, 1, norm_layer, "relu", 0.0),
            
            # Final layer to output
            nn.ConvTranspose2d(ngf, output_channels, 4, 2, 1),
            nn.Tanh()
        ])
    
    def forward(self, x):
        # Store skip connections from encoder
        skips = []
        
        # Encoder
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)
        
        # Bottleneck (residual blocks - don't change spatial dimensions)
        for layer in self.bottleneck:
            x = layer(x)
        
        # Decoder with skip connections
        # Skip connections are used in reverse order (excluding the last encoder output)
        skips = skips[::-1][1:]  # Reverse and skip the last one (which is the bottleneck input)
        
        skip_idx = 0
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.ConvTranspose2d):
                x = layer(x)
                # Add skip connection if available
                if skip_idx < len(skips):
                    skip = skips[skip_idx]
                    # Ensure spatial dimensions match
                    if x.shape[2:] != skip.shape[2:]:
                        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                    x = torch.cat([x, skip], dim=1)
                    skip_idx += 1
            else:
                x = layer(x)
        
        return x


class Pix2PixDiscriminator(nn.Module):
    """Pix2Pix PatchGAN Discriminator."""
    
    def __init__(self, input_channels: int = 6, ndf: int = 64, n_layers: int = 3,
                 norm_layer: str = "batch"):
        super().__init__()
        
        layers = [
            ConvBlock(input_channels, ndf, 4, 2, 1, "none", "leaky_relu", 0.0)
        ]
        
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.append(
                ConvBlock(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1, norm_layer, "leaky_relu", 0.0)
            )
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.append(
            ConvBlock(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1, norm_layer, "leaky_relu", 0.0)
        )
        
        layers.append(nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, input_img, target_img):
        x = torch.cat([input_img, target_img], dim=1)
        return self.model(x)


# ============================================================================
# CycleGAN Generator and Discriminator
# ============================================================================

class CycleGANGenerator(nn.Module):
    """CycleGAN Generator with ResNet blocks."""
    
    def __init__(self, input_channels: int = 3, output_channels: int = 3,
                 ngf: int = 64, n_blocks: int = 9, norm_layer: str = "instance"):
        super().__init__()
        
        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            ConvBlock(input_channels, ngf, 7, 1, 0, norm_layer, "relu", 0.0)
        ]
        
        # Downsampling
        for i in range(2):
            mult = 2 ** i
            model.append(
                ConvBlock(ngf * mult, ngf * mult * 2, 3, 2, 1, norm_layer, "relu", 0.0)
            )
        
        # ResNet blocks
        mult = 2 ** 2
        for i in range(n_blocks):
            model.append(ResBlock(ngf * mult, norm_layer))
        
        # Upsampling
        for i in range(2):
            mult = 2 ** (2 - i)
            model.append(
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), 3, 2, 1, 1)
            )
            model.append(
                ConvBlock(int(ngf * mult / 2), int(ngf * mult / 2), 3, 1, 1, norm_layer, "relu", 0.0)
            )
        
        # Output layer
        model.extend([
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_channels, 7, 1, 0),
            nn.Tanh()
        ])
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


class CycleGANDiscriminator(nn.Module):
    """CycleGAN PatchGAN Discriminator."""
    
    def __init__(self, input_channels: int = 3, ndf: int = 64, n_layers: int = 3,
                 norm_layer: str = "instance"):
        super().__init__()
        
        layers = [
            ConvBlock(input_channels, ndf, 4, 2, 1, "none", "leaky_relu", 0.0)
        ]
        
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.append(
                ConvBlock(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1, norm_layer, "leaky_relu", 0.0)
            )
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.append(
            ConvBlock(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1, norm_layer, "leaky_relu", 0.0)
        )
        
        layers.append(nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


# ============================================================================
# Model Factory Functions
# ============================================================================

def create_model(config: dict) -> nn.Module:
    """
    Create a model based on configuration.
    
    Args:
        config: Model configuration dictionary.
        
    Returns:
        PyTorch model instance.
    """
    architecture = config.get("architecture", "unet")
    params = config.get("parameters", {})
    
    if architecture == "unet":
        return UNet(
            input_channels=params.get("input_channels", 3),
            output_channels=params.get("output_channels", 3),
            features=params.get("features", [64, 128, 256, 512]),
            dropout=params.get("dropout", 0.1),
            attention=params.get("attention", False)
        )
    
    elif architecture == "pix2pix":
        generator = Pix2PixGenerator(
            input_channels=params.get("generator", {}).get("input_channels", 3),
            output_channels=params.get("generator", {}).get("output_channels", 3),
            ngf=params.get("generator", {}).get("ngf", 64),
            n_blocks=params.get("generator", {}).get("n_blocks", 6),
            norm_layer=params.get("generator", {}).get("norm_layer", "batch"),
            dropout=params.get("generator", {}).get("dropout", 0.5)
        )
        
        discriminator = Pix2PixDiscriminator(
            input_channels=params.get("discriminator", {}).get("input_channels", 6),
            ndf=params.get("discriminator", {}).get("ndf", 64),
            n_layers=params.get("discriminator", {}).get("n_layers", 3),
            norm_layer=params.get("discriminator", {}).get("norm_layer", "batch")
        )
        
        return {"generator": generator, "discriminator": discriminator}
    
    elif architecture == "cyclegan":
        generator_A2B = CycleGANGenerator(
            input_channels=params.get("generator", {}).get("input_channels", 3),
            output_channels=params.get("generator", {}).get("output_channels", 3),
            ngf=params.get("generator", {}).get("ngf", 64),
            n_blocks=params.get("generator", {}).get("n_blocks", 9),
            norm_layer=params.get("generator", {}).get("norm_layer", "instance")
        )
        
        generator_B2A = CycleGANGenerator(
            input_channels=params.get("generator", {}).get("input_channels", 3),
            output_channels=params.get("generator", {}).get("output_channels", 3),
            ngf=params.get("generator", {}).get("ngf", 64),
            n_blocks=params.get("generator", {}).get("n_blocks", 9),
            norm_layer=params.get("generator", {}).get("norm_layer", "instance")
        )
        
        discriminator_A = CycleGANDiscriminator(
            input_channels=params.get("discriminator", {}).get("input_channels", 3),
            ndf=params.get("discriminator", {}).get("ndf", 64),
            n_layers=params.get("discriminator", {}).get("n_layers", 3),
            norm_layer=params.get("discriminator", {}).get("norm_layer", "instance")
        )
        
        discriminator_B = CycleGANDiscriminator(
            input_channels=params.get("discriminator", {}).get("input_channels", 3),
            ndf=params.get("discriminator", {}).get("ndf", 64),
            n_layers=params.get("discriminator", {}).get("n_layers", 3),
            norm_layer=params.get("discriminator", {}).get("norm_layer", "instance")
        )
        
        return {
            "generator_A2B": generator_A2B,
            "generator_B2A": generator_B2A,
            "discriminator_A": discriminator_A,
            "discriminator_B": discriminator_B
        }
    
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def count_parameters(model) -> int:
    """Count the number of trainable parameters in a model."""
    if isinstance(model, dict):
        return sum(count_parameters(m) for m in model.values())
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
