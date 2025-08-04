"""
Pretrained backbone generator for Pokemon sprite generation.

This module implements a generator that uses pretrained CNN backbones
(ResNet, EfficientNet) for feature extraction followed by a decoder
network for image generation.
"""

import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torchvision.models as models

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


class PretrainedBackboneGenerator(nn.Module):
    """Generator using pretrained backbone features."""

    def __init__(
        self,
        backbone: str = "resnet50",
        freeze_backbone: bool = True,
        output_channels: int = 3,
        decoder_features: Optional[List[int]] = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if decoder_features is None:
            decoder_features = [512, 256, 128, 64]

        # Setup pretrained backbone
        self.backbone_name = backbone
        self.freeze_backbone = freeze_backbone

        if backbone == "resnet50":
            backbone_model = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT
            )
            self.backbone = nn.Sequential(
                *list(backbone_model.children())[:-2]
            )  # Remove avgpool and fc
            backbone_features = 2048
        elif backbone == "resnet34":
            backbone_model = models.resnet34(
                weights=models.ResNet34_Weights.DEFAULT
            )
            self.backbone = nn.Sequential(
                *list(backbone_model.children())[:-2]
            )
            backbone_features = 512
        elif backbone == "efficientnet_b0":
            backbone_model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT
            )
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

        logger.info(f"PretrainedBackboneGenerator created: {backbone}")
        logger.info(f"  Backbone: {backbone} (frozen: {freeze_backbone})")
        logger.info(f"  Backbone features: {backbone_features}")
        logger.info(
            f"  Final output size: {current_size * 2}x{current_size * 2}"
        )

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
