"""
Improved perceptual loss function for Pokemon sprite generation.

This module implements an enhanced perceptual loss using VGG features with
anti-blur optimizations for better visual quality in generated sprites.
"""

from typing import List

import torch
import torch.nn.functional as F
import torchvision.models as models

from .base_loss import BaseLoss


class ImprovedPerceptualLoss(BaseLoss):
    """
    Improved perceptual loss with anti-blur enhancements.

    Uses VGG16 features at multiple layers to capture perceptual similarities
    while encouraging sharp, detailed outputs through L1 loss instead of MSE.
    """

    def __init__(
        self,
        layers: List[int] = [3, 8, 15, 22],
        weights: List[float] = [1.0, 1.0, 1.0, 1.0],
    ):
        """
        Initialize improved perceptual loss.

        Args:
            layers: VGG layer indices to use for feature extraction
            weights: Weights for each layer's contribution
        """
        super().__init__("ImprovedPerceptualLoss")
        self.layers = layers
        self.weights = weights

        if len(layers) != len(weights):
            raise ValueError(
                f"Number of layers ({len(layers)}) must match number of "
                f"weights ({len(weights)})"
            )

        # Initialize VGG network
        self._setup_vgg()

        self.logger.info(
            f"Initialized ImprovedPerceptualLoss with layers={layers}, "
            f"weights={weights}"
        )

    def _setup_vgg(self):
        """Setup VGG16 network for feature extraction."""
        try:
            # Use new weights parameter instead of deprecated pretrained
            from torchvision.models import VGG16_Weights

            self.vgg = models.vgg16(
                weights=VGG16_Weights.IMAGENET1K_V1
            ).features
            self.logger.info("Using VGG16 with IMAGENET1K_V1 weights")
        except ImportError:
            # Fallback for older torchvision versions
            self.vgg = models.vgg16(pretrained=True).features
            self.logger.warning(
                "Using deprecated pretrained parameter for VGG16"
            )

        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

        # Use full precision for better quality
        self.vgg = self.vgg.float()

    def forward(
        self, generated: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute improved perceptual loss.

        Args:
            generated: Generated images [B, C, H, W]
            target: Target images [B, C, H, W]

        Returns:
            Perceptual loss value
        """
        self.validate_inputs(generated, target)

        try:
            # Ensure inputs are in [0, 1] range
            input_img = self._normalize_input(generated)
            target_img = self._normalize_input(target)

            # Resize to 224x224 for VGG
            input_img = F.interpolate(
                input_img,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )
            target_img = F.interpolate(
                target_img,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )

            loss = torch.tensor(
                0.0, device=generated.device, dtype=torch.float32
            )
            layer_losses = {}

            x = input_img
            y = target_img

            for i, layer in enumerate(list(self.vgg.children())):
                x = layer(x)
                y = layer(y)

                if i in self.layers:
                    weight = self.weights[self.layers.index(i)]
                    # Use L1 loss instead of MSE for sharper features
                    layer_loss = F.l1_loss(x, y)
                    weighted_loss = weight * layer_loss
                    loss += weighted_loss

                    layer_losses[f"layer_{i}"] = layer_loss.item()

            # Log detailed information
            self.log_loss_info(loss.item(), layer_losses)

            return loss

        except Exception as e:
            self.logger.error(f"Error computing perceptual loss: {str(e)}")
            raise

    def _normalize_input(self, image: torch.Tensor) -> torch.Tensor:
        """
        Normalize input to [0, 1] range if needed.

        Args:
            image: Input image tensor

        Returns:
            Normalized image tensor
        """
        if image.max() > 1.0 or image.min() < 0.0:
            # Assume input is in [-1, 1] range
            return (image + 1) / 2
        return image
