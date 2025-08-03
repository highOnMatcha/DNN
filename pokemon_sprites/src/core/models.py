"""
Model architectures for Pokemon sprite generation.

This module contains implementations of different image-to-image translation
architectures including Pix2Pix, U-Net, CycleGAN, and diffusion models.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ============================================================================
# Pretrained backbone models
# ============================================================================


class PretrainedBackboneGenerator(nn.Module):
    """Generator using pretrained backbone features."""

    def __init__(
        self,
        backbone: str = "resnet50",
        freeze_backbone: bool = True,
        output_channels: int = 3,
        decoder_features: List[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if decoder_features is None:
            decoder_features = [512, 256, 128, 64]

        # Setup pretrained backbone
        self.backbone_name = backbone
        self.freeze_backbone = freeze_backbone

        if backbone == "resnet50":
            backbone_model = models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(
                *list(backbone_model.children())[:-2]
            )  # Remove avgpool and fc
            backbone_features = 2048
        elif backbone == "resnet34":
            backbone_model = models.resnet34(pretrained=True)
            self.backbone = nn.Sequential(
                *list(backbone_model.children())[:-2]
            )
            backbone_features = 512
        elif backbone == "efficientnet_b0":
            backbone_model = models.efficientnet_b0(pretrained=True)
            self.backbone = backbone_model.features
            backbone_features = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Adaptive pooling to standard size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        # Decoder network
        self.decoder = nn.ModuleList()

        # First layer: flatten backbone features
        flat_features = backbone_features * 8 * 8
        self.feature_processor = nn.Sequential(
            nn.Linear(flat_features, decoder_features[0] * 4 * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Convolutional decoder layers
        current_features = decoder_features[0]
        current_size = 4

        for next_features in decoder_features[1:]:
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        current_features, next_features, 4, 2, 1
                    ),
                    nn.BatchNorm2d(next_features),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout * 0.5),  # Less dropout in later layers
                )
            )
            current_features = next_features
            current_size *= 2

        # Final layer to output channels
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(current_features, output_channels, 4, 2, 1),
            nn.Tanh(),
        )

        print(f"PretrainedBackboneGenerator created: {backbone}")
        print(f"  Backbone: {backbone} (frozen: {freeze_backbone})")
        print(f"  Backbone features: {backbone_features}")
        print(f"  Final output size: {current_size * 2}x{current_size * 2}")

    def forward(self, x):
        # Extract features with backbone
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)

        # Adaptive pooling
        features = self.adaptive_pool(features)  # [B, backbone_features, 8, 8]

        # Flatten and process
        batch_size = features.size(0)
        features = features.view(batch_size, -1)  # [B, backbone_features * 64]
        features = self.feature_processor(
            features
        )  # [B, decoder_features[0] * 16]

        # Reshape for conv layers
        features = features.view(
            batch_size, -1, 4, 4
        )  # [B, decoder_features[0], 4, 4]

        # Decode through conv layers
        x = features
        for decoder_layer in self.decoder:
            x = decoder_layer(x)

        # Final output
        x = self.final_layer(x)

        return x


class ViTCLIPGenerator(nn.Module):
    """Simplified CNN-based generator optimized for pixel art generation."""

    def __init__(
        self,
        vit_model: str = "vit_base_patch16_224",
        use_clip: bool = False,
        output_channels: int = 3,
        decoder_features: List[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if decoder_features is None:
            decoder_features = [256, 128, 64, 32]

        print("Creating simplified pixel-art optimized generator...")

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

        print(
            f"Pixel-art Generator created with "
            f"{len(self.encoder)} encoder layers"
        )
        print(f"  Encoder layers: {len(self.encoder)}")
        print(f"  Decoder layers: {len(self.decoder)}")
        print(f"  Features: {decoder_features}")
        print(f"  Dropout: {dropout}")

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
        self.channels = channels
        self.spatial_size = spatial_size
        self.seq_length = spatial_size * spatial_size

        # Positional encoding for spatial positions
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.seq_length, channels)
        )

        # Transformer layers
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

        # Layer norm for input
        self.input_norm = nn.LayerNorm(channels)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [B, C, H, W] where H=W=spatial_size
        B, C, H, W = x.shape

        # Flatten spatial dimensions: [B, C, H, W] -> [B, H*W, C]
        x_flat = x.view(B, C, H * W).transpose(1, 2)

        # Add positional encoding
        x_pos = x_flat + self.pos_encoding
        x_pos = self.input_norm(x_pos)
        x_pos = self.dropout(x_pos)

        # Apply transformer
        x_transformed = self.transformer(x_pos)

        # Reshape back to spatial: [B, H*W, C] -> [B, C, H, W]
        x_out = x_transformed.transpose(1, 2).view(B, C, H, W)

        # Residual connection
        return x + x_out


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


# ============================================================================
# Model factory function
# ===============================================================================================================
# Utility layers and functions
# ============================================================================


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


class UNet(nn.Module):
    """U-Net model for image-to-image translation."""

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        features: List[int] = [64, 128, 256, 512],
        dropout: float = 0.1,
        attention: bool = False,
    ):
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


# ============================================================================
# CycleGAN Generator and Discriminator
# ============================================================================


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

        layers = [
            ConvBlock(input_channels, ndf, 4, 2, 1, "none", "leaky_relu", 0.0)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers.append(
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
        layers.append(
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

        layers.append(nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ============================================================================
# Model Factory Functions
# ============================================================================


def create_model(config) -> nn.Module:
    """
    Create a model based on configuration.

    Args:
        config: Model configuration (dict or ModelConfig object).

    Returns:
        PyTorch model instance.
    """
    # Handle both dict and ModelConfig objects
    if hasattr(config, "architecture"):
        architecture = config.architecture
        params = config.parameters
    else:
        architecture = config.get("architecture", "unet")
        params = config.get("parameters", {})

    if architecture == "unet":
        return UNet(
            input_channels=params.get("input_channels", 3),
            output_channels=params.get("output_channels", 3),
            features=params.get("features", [64, 128, 256, 512]),
            dropout=params.get("dropout", 0.1),
            attention=params.get("attention", False),
        )

    elif architecture == "pix2pix":
        generator = Pix2PixGenerator(
            input_channels=params.get("generator", {}).get(
                "input_channels", 3
            ),
            output_channels=params.get("generator", {}).get(
                "output_channels", 3
            ),
            ngf=params.get("generator", {}).get("ngf", 64),
            n_blocks=params.get("generator", {}).get("n_blocks", 6),
            norm_layer=params.get("generator", {}).get("norm_layer", "batch"),
            dropout=params.get("generator", {}).get("dropout", 0.5),
        )

        discriminator = Pix2PixDiscriminator(
            input_channels=params.get("discriminator", {}).get(
                "input_channels", 6
            ),
            ndf=params.get("discriminator", {}).get("ndf", 64),
            n_layers=params.get("discriminator", {}).get("n_layers", 3),
            norm_layer=params.get("discriminator", {}).get(
                "norm_layer", "batch"
            ),
        )

        return {"generator": generator, "discriminator": discriminator}

    elif architecture == "pix2pix_pretrained":
        # Pretrained backbone generator with regular discriminator
        generator = PretrainedBackboneGenerator(
            backbone=params.get("backbone", "resnet50"),
            freeze_backbone=params.get("freeze_backbone", True),
            output_channels=params.get("generator", {}).get(
                "output_channels", 3
            ),
            decoder_features=params.get("generator", {}).get(
                "decoder_features", [512, 256, 128, 64]
            ),
            dropout=params.get("generator", {}).get("dropout", 0.3),
        )

        discriminator = Pix2PixDiscriminator(
            input_channels=params.get("discriminator", {}).get(
                "input_channels", 6
            ),
            ndf=params.get("discriminator", {}).get("ndf", 64),
            n_layers=params.get("discriminator", {}).get("n_layers", 3),
            norm_layer=params.get("discriminator", {}).get(
                "norm_layer", "instance"
            ),
        )

        return {"generator": generator, "discriminator": discriminator}

    elif architecture == "pix2pix_vit":
        generator = ViTCLIPGenerator(
            vit_model=params.get("generator", {}).get(
                "vit_model", "vit_large_patch16_224"
            ),
            use_clip=params.get("generator", {}).get("use_clip", True),
            output_channels=params.get("generator", {}).get(
                "output_channels", 3
            ),
            decoder_features=params.get("generator", {}).get(
                "decoder_features", [1024, 512, 256, 128]
            ),
            dropout=params.get("generator", {}).get("dropout", 0.2),
        )

        discriminator = Pix2PixDiscriminator(
            input_channels=params.get("discriminator", {}).get(
                "input_channels", 6
            ),
            ndf=params.get("discriminator", {}).get("ndf", 64),
            n_layers=params.get("discriminator", {}).get("n_layers", 4),
            norm_layer=params.get("discriminator", {}).get(
                "norm_layer", "instance"
            ),
        )

        return {"generator": generator, "discriminator": discriminator}

    elif architecture == "cyclegan":
        generator_A2B = CycleGANGenerator(
            input_channels=params.get("generator", {}).get(
                "input_channels", 3
            ),
            output_channels=params.get("generator", {}).get(
                "output_channels", 3
            ),
            ngf=params.get("generator", {}).get("ngf", 64),
            n_blocks=params.get("generator", {}).get("n_blocks", 9),
            norm_layer=params.get("generator", {}).get(
                "norm_layer", "instance"
            ),
        )

        generator_B2A = CycleGANGenerator(
            input_channels=params.get("generator", {}).get(
                "input_channels", 3
            ),
            output_channels=params.get("generator", {}).get(
                "output_channels", 3
            ),
            ngf=params.get("generator", {}).get("ngf", 64),
            n_blocks=params.get("generator", {}).get("n_blocks", 9),
            norm_layer=params.get("generator", {}).get(
                "norm_layer", "instance"
            ),
        )

        discriminator_A = CycleGANDiscriminator(
            input_channels=params.get("discriminator", {}).get(
                "input_channels", 3
            ),
            ndf=params.get("discriminator", {}).get("ndf", 64),
            n_layers=params.get("discriminator", {}).get("n_layers", 3),
            norm_layer=params.get("discriminator", {}).get(
                "norm_layer", "instance"
            ),
        )

        discriminator_B = CycleGANDiscriminator(
            input_channels=params.get("discriminator", {}).get(
                "input_channels", 3
            ),
            ndf=params.get("discriminator", {}).get("ndf", 64),
            n_layers=params.get("discriminator", {}).get("n_layers", 3),
            norm_layer=params.get("discriminator", {}).get(
                "norm_layer", "instance"
            ),
        )

        return {
            "generator_A2B": generator_A2B,
            "generator_B2A": generator_B2A,
            "discriminator_A": discriminator_A,
            "discriminator_B": discriminator_B,
        }

    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def count_parameters(model) -> int:
    """Count the number of trainable parameters in a model."""
    if isinstance(model, dict):
        return sum(count_parameters(m) for m in model.values())
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model) -> int:
    """Count the total number of parameters in a model (including frozen)."""
    if isinstance(model, dict):
        return sum(count_total_parameters(m) for m in model.values())
    return sum(p.numel() for p in model.parameters())


def analyze_model_architectures(config_path=None):
    """
    Analyze available model architectures for Pokemon sprite generation.

    Evaluates model configurations for artwork-to-sprite translation,
    focusing on ARGB support, parameter efficiency, and pixel art optimization.

    Args:
        config_path: Path to model configuration file

    Returns:
        dict: Analysis results with suitability scores
    """
    import json
    from pathlib import Path

    from core.logging_config import get_logger

    logger = get_logger("model_analyzer")

    if config_path is None:
        config_file_path = (
            Path(__file__).parent.parent / "config" / "model_configs.json"
        )
    else:
        config_file_path = Path(config_path)

    if not config_file_path.exists():
        logger.error(f"Model configuration file not found: {config_file_path}")
        return {}

    try:
        with open(config_file_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return {}

    logger.info("Analyzing curated model architectures for sprite generation")

    model_configs = config.get("pix2pix_models", {})
    if not model_configs:
        logger.warning("No model configurations found")
        return {}

    logger.info(f"Found {len(model_configs)} model configurations")

    architecture_analysis = {}
    analysis_results = {"models": {}, "recommendations": {}, "summary": {}}

    for model_name, model_config in model_configs.items():
        logger.info(f"Analyzing model: {model_name}")

        params = model_config.get("parameters", {})
        generator = params.get("generator", {})
        discriminator = params.get("discriminator", {})

        # Calculate parameter estimates
        ngf = generator.get("ngf", 64)
        ndf = discriminator.get("ndf", 64)
        n_blocks = generator.get("n_blocks", 9)
        d_layers = discriminator.get("n_layers", 3)

        # Architecture-specific parameter calculation
        if "transformer" in model_name:
            transformer_layers = generator.get("transformer_layers", 4)
            attention_heads = generator.get("attention_heads", 8)
            transformer_params = (
                transformer_layers * attention_heads * ngf * ngf * 4
            )
            base_params = ngf * ngf * (n_blocks * 2 + 8) + ndf * ndf * (
                d_layers + 2
            )
            total_params = (base_params + transformer_params) / 1000000
        else:
            gen_params = ngf * ngf * (n_blocks * 2 + 10)
            disc_params = ndf * ndf * (d_layers + 2)
            total_params = (gen_params + disc_params) / 1000000

        # Task-specific assessment for ARGB sprite generation
        suitability_score = 0
        strengths = []
        considerations = []

        # ARGB support check (critical for transparency)
        if (
            generator.get("input_channels", 3) == 4
            and generator.get("output_channels", 3) == 4
        ):
            strengths.append("Native ARGB transparency support")
            suitability_score += 2
        else:
            considerations.append("Missing ARGB channel support")

        # Image size compatibility
        if params.get("image_size", 0) == 256:
            strengths.append("Direct 256px output compatibility")
            suitability_score += 1
        else:
            considerations.append("Non-standard image size")

        # Loss function optimization for pixel art
        if params.get("lambda_l1", 0) >= 150:
            strengths.append("Strong pixel-level accuracy emphasis")
            suitability_score += 1

        if params.get("lambda_pixel_art", 0) > 0:
            strengths.append("Specialized pixel art loss function")
            suitability_score += 1

        # Model-specific analysis
        if "lightweight" in model_name:
            strengths.extend(
                [
                    "Fast training and inference",
                    "Low memory requirements",
                    "Reduced overfitting risk",
                ]
            )
            considerations.append("May lack capacity for complex mappings")
            suitability_score += 2  # Good for baseline
            role = "BASELINE - Quick validation and debugging"

        elif "sprite-optimized" in model_name:
            if generator.get("use_attention", False):
                strengths.append("Attention mechanism for detail preservation")
                suitability_score += 1
            if discriminator.get("use_spectral_norm", False):
                strengths.append(
                    "Spectral normalization for training stability"
                )
                suitability_score += 1
            strengths.extend(
                [
                    "Optimized loss weights for pixel art",
                    "Balanced complexity for dataset size",
                ]
            )
            suitability_score += 2  # Recommended primary model
            role = "PRIMARY - Main production model"

        elif "transformer" in model_name:
            strengths.extend(
                [
                    "Long-range dependency modeling",
                    "Advanced attention mechanisms",
                    "State-of-the-art architecture",
                ]
            )
            considerations.extend(
                [
                    "Higher computational requirements",
                    "May need careful regularization",
                ]
            )
            suitability_score += 1  # Experimental but promising
            role = "ADVANCED - Experimental state-of-the-art"
        else:
            role = "GENERAL - Standard configuration"

        # Normalization preference for style transfer
        if generator.get("norm_layer", "batch") == "instance":
            strengths.append(
                "Instance normalization (better for style transfer)"
            )
            suitability_score += 1

        # Store detailed analysis
        model_analysis = {
            "description": model_config.get("description", "No description"),
            "architecture": model_config.get("architecture", "pix2pix"),
            "role": role,
            "estimated_parameters_m": round(total_params, 1),
            "suitability_score": suitability_score,
            "max_score": 12,
            "strengths": strengths,
            "considerations": considerations,
            "generator_config": {
                "channels": f"{generator.get('input_channels', 3)} -> {generator.get('output_channels', 3)}",
                "base_features": ngf,
                "blocks": n_blocks,
                "normalization": generator.get("norm_layer", "instance"),
                "dropout": generator.get("dropout", 0.3),
            },
            "discriminator_config": {
                "input_channels": discriminator.get("input_channels", 6),
                "base_features": ndf,
                "layers": d_layers,
                "spectral_norm": discriminator.get("use_spectral_norm", False),
            },
            "training_config": {
                "image_size": params.get("image_size", 256),
                "l1_weight": params.get("lambda_l1", 100),
                "perceptual_weight": params.get("lambda_perceptual", 0),
                "pixel_art_weight": params.get("lambda_pixel_art", 0),
            },
        }

        analysis_results["models"][model_name] = model_analysis
        architecture_analysis[model_name] = suitability_score

        logger.info(f"Model {model_name}: Score {suitability_score}/12")

    # Generate recommendations
    sorted_models = sorted(
        architecture_analysis.items(), key=lambda x: x[1], reverse=True
    )

    recommendations = []
    for i, (model_name, score) in enumerate(sorted_models, 1):
        model_info = analysis_results["models"][model_name]
        use_case = (
            "Start here for quick results"
            if i == 1
            else (
                "Optimize after baseline"
                if i == 2
                else "Experiment if resources allow"
            )
        )

        recommendations.append(
            {
                "rank": i,
                "model_name": model_name,
                "role": model_info["role"],
                "score": f"{score}/12",
                "use_case": use_case,
                "parameters_m": model_info["estimated_parameters_m"],
            }
        )

    analysis_results["recommendations"] = recommendations

    # Summary statistics
    analysis_results["summary"] = {
        "total_models": len(model_configs),
        "argb_compatible": sum(
            1
            for m in analysis_results["models"].values()
            if "4 -> 4" in m["generator_config"]["channels"]
        ),
        "pixel_art_optimized": sum(
            1
            for m in analysis_results["models"].values()
            if m["training_config"]["pixel_art_weight"] > 0
        ),
        "recommended_sequence": [r["model_name"] for r in recommendations[:3]],
    }

    logger.info("Model architecture analysis completed")
    logger.info(
        f"Recommended training sequence: {' -> '.join(analysis_results['summary']['recommended_sequence'])}"
    )

    return analysis_results
