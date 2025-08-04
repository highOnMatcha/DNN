"""
Combined anti-blur loss function for Pokemon sprite generation.

This module implements a comprehensive loss function that combines multiple
specialized losses to combat blurriness and maintain high-quality sprite
generation.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .anti_blur_loss import AntiBlurLoss
from .base_loss import BaseLoss
from .perceptual_loss import ImprovedPerceptualLoss
from .pixel_art_loss import PixelArtLoss


class CombinedAntiBlurLoss(BaseLoss):
    """
    Combined loss function to combat blurriness comprehensively.

    Integrates L1 reconstruction loss with specialized anti-blur components
    including edge preservation, pixel-art style preservation, and perceptual
    quality.
    """

    def __init__(
        self,
        l1_weight: float = 100.0,
        anti_blur_weight: float = 10.0,
        pixel_art_weight: float = 5.0,
        perceptual_weight: float = 1.0,
    ):
        """
        Initialize combined anti-blur loss.

        Args:
            l1_weight: Weight for L1 reconstruction loss
            anti_blur_weight: Weight for anti-blur loss component
            pixel_art_weight: Weight for pixel-art loss component
            perceptual_weight: Weight for perceptual loss component
        """
        super().__init__("CombinedAntiBlurLoss")

        self.l1_weight = l1_weight
        self.anti_blur_weight = anti_blur_weight
        self.pixel_art_weight = pixel_art_weight
        self.perceptual_weight = perceptual_weight

        # Initialize loss components
        self.l1_loss = nn.L1Loss()
        self.anti_blur_loss = AntiBlurLoss()
        self.pixel_art_loss = PixelArtLoss()
        self.perceptual_loss = ImprovedPerceptualLoss()

        self.logger.info(
            f"Initialized CombinedAntiBlurLoss with weights - "
            f"L1: {l1_weight}, AntiBlur: {anti_blur_weight}, "
            f"PixelArt: {pixel_art_weight}, Perceptual: {perceptual_weight}"
        )

    def forward(
        self, generated: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined anti-blur loss.

        Args:
            generated: Generated images [B, C, H, W]
            target: Target images [B, C, H, W]

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        self.validate_inputs(generated, target)

        try:
            # Basic L1 loss
            l1 = self.l1_loss(generated, target)

            # Anti-blur components
            anti_blur = self.anti_blur_loss(generated, target)
            pixel_art = self.pixel_art_loss(generated, target)
            perceptual = self.perceptual_loss(generated, target)

            # Combine losses
            total_loss = (
                self.l1_weight * l1
                + self.anti_blur_weight * anti_blur
                + self.pixel_art_weight * pixel_art
                + self.perceptual_weight * perceptual
            )

            # Prepare loss components for logging
            loss_components = {
                "l1_loss": l1.item(),
                "anti_blur_loss": anti_blur.item(),
                "pixel_art_loss": pixel_art.item(),
                "perceptual_loss": perceptual.item(),
                "total_loss": total_loss.item(),
            }

            # Log combined loss information
            self.log_loss_info(total_loss.item(), loss_components)

            return total_loss, loss_components

        except Exception as e:
            self.logger.error(
                f"Error computing combined anti-blur loss: {str(e)}"
            )
            raise
