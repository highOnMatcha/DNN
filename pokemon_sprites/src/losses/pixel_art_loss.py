"""
Pixel art loss function for Pokemon sprite generation.

This module implements a specialized loss function designed to preserve
pixel-art characteristics including sharp edges and discrete color values.
"""

import torch
import torch.nn.functional as F

from .base_loss import BaseLoss


class PixelArtLoss(BaseLoss):
    """
    Specialized loss for pixel-art style preservation.

    Encourages sharp edges and discrete color values typical of pixel art
    by penalizing smooth gradients and promoting color quantization.
    """

    def __init__(
        self, sharpness_weight: float = 2.0, color_weight: float = 1.0
    ):
        """
        Initialize pixel art loss.

        Args:
            sharpness_weight: Weight for sharpness preservation
            color_weight: Weight for color quantization
        """
        super().__init__("PixelArtLoss")
        self.sharpness_weight = sharpness_weight
        self.color_weight = color_weight

        self.logger.info(
            f"Initialized PixelArtLoss with sharpness_weight="
            f"{sharpness_weight}, color_weight={color_weight}"
        )

    def forward(
        self, generated: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pixel-art preservation loss.

        Args:
            generated: Generated images [B, C, H, W]
            target: Target images [B, C, H, W]

        Returns:
            Pixel-art loss value
        """
        self.validate_inputs(generated, target)

        try:
            # Sharpness loss (penalize smooth gradients)
            sharpness_loss = self._sharpness_loss(generated, target)

            # Color quantization loss (encourage discrete colors)
            color_loss = self._color_quantization_loss(generated, target)

            total_loss = (
                self.sharpness_weight * sharpness_loss
                + self.color_weight * color_loss
            )

            # Log detailed information
            self.log_loss_info(
                total_loss.item(),
                {
                    "sharpness_loss": sharpness_loss.item(),
                    "color_loss": color_loss.item(),
                    "sharpness_weight": self.sharpness_weight,
                    "color_weight": self.color_weight,
                },
            )

            return total_loss

        except Exception as e:
            self.logger.error(f"Error computing pixel art loss: {str(e)}")
            raise

    def _sharpness_loss(
        self, generated: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalize smooth gradients to encourage sharp edges.

        Args:
            generated: Generated images
            target: Target images

        Returns:
            Sharpness loss
        """
        # Compute local gradients
        gen_grad_x = torch.abs(
            generated[:, :, :, 1:] - generated[:, :, :, :-1]
        )
        gen_grad_y = torch.abs(
            generated[:, :, 1:, :] - generated[:, :, :-1, :]
        )

        target_grad_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        target_grad_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

        # Encourage generated gradients to match target gradients
        loss_x = F.mse_loss(gen_grad_x, target_grad_x)
        loss_y = F.mse_loss(gen_grad_y, target_grad_y)

        return loss_x + loss_y

    def _color_quantization_loss(
        self, generated: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage discrete color values typical of pixel art.

        Args:
            generated: Generated images
            target: Target images

        Returns:
            Color quantization loss
        """
        # Quantize to limited color palette (simulate pixel art)
        levels = 16  # Reduce to 16 levels per channel

        # Quantize generated images
        gen_quantized = torch.round(generated * (levels - 1)) / (levels - 1)

        # Encourage generated images to be close to quantized version
        quantization_loss = F.mse_loss(generated, gen_quantized.detach())

        # Also compare quantized versions
        target_quantized = torch.round(target * (levels - 1)) / (levels - 1)
        palette_loss = F.mse_loss(gen_quantized, target_quantized)

        return quantization_loss + palette_loss
