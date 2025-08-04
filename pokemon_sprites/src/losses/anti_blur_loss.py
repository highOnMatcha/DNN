"""
Anti-blur loss function for Pokemon sprite generation.

This module implements a specialized loss function designed to combat
blurriness in generated images by combining edge preservation and
high-frequency enhancement.
"""

import torch
import torch.nn.functional as F

from .base_loss import BaseLoss
from .utils import (
    compute_edges,
    create_high_pass_filter,
    create_sobel_filters,
    rgb_to_grayscale,
)


class AntiBlurLoss(BaseLoss):
    """
    Specialized loss function to combat blurriness in generated images.

    Combines edge preservation using Sobel operators with high-frequency
    enhancement using FFT analysis to encourage sharp, detailed outputs.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        """
        Initialize anti-blur loss.

        Args:
            alpha: Edge preservation weight
            beta: High frequency enhancement weight
        """
        super().__init__("AntiBlurLoss")
        self.alpha = alpha
        self.beta = beta

        self.logger.info(
            f"Initialized AntiBlurLoss with alpha={alpha}, beta={beta}"
        )

    def forward(
        self, generated: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute anti-blur loss combining edge preservation and high-frequency
        enhancement.

        Args:
            generated: Generated images [B, C, H, W]
            target: Target images [B, C, H, W]

        Returns:
            Anti-blur loss value
        """
        self.validate_inputs(generated, target)

        try:
            # Edge preservation loss
            edge_loss = self._edge_loss(generated, target)

            # High frequency enhancement loss
            freq_loss = self._frequency_loss(generated, target)

            total_loss = self.alpha * edge_loss + self.beta * freq_loss

            # Log detailed information
            self.log_loss_info(
                total_loss.item(),
                {
                    "edge_loss": edge_loss.item(),
                    "freq_loss": freq_loss.item(),
                    "alpha": self.alpha,
                    "beta": self.beta,
                },
            )

            return total_loss

        except Exception as e:
            self.logger.error(f"Error computing anti-blur loss: {str(e)}")
            raise

    def _edge_loss(
        self, generated: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute edge preservation loss using Sobel operators.

        Args:
            generated: Generated images
            target: Target images

        Returns:
            Edge preservation loss
        """
        # Create Sobel filters
        sobel_x, sobel_y = create_sobel_filters(generated.device)

        # Compute edges for both images
        gen_edges = compute_edges(generated, sobel_x, sobel_y)
        target_edges = compute_edges(target, sobel_x, sobel_y)

        return F.l1_loss(gen_edges, target_edges)

    def _frequency_loss(
        self, generated: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute high-frequency enhancement loss using FFT.

        Args:
            generated: Generated images
            target: Target images

        Returns:
            High-frequency enhancement loss
        """
        # Convert to grayscale for frequency analysis
        gen_gray = rgb_to_grayscale(generated)
        target_gray = rgb_to_grayscale(target)

        # Compute FFT
        gen_fft = torch.fft.fft2(gen_gray)
        target_fft = torch.fft.fft2(target_gray)

        # Create high-pass filter
        h, w = gen_fft.shape[-2:]
        high_pass = create_high_pass_filter(
            h, w, generated.device, threshold=0.1
        )

        # Apply high-pass filter
        gen_high_freq = gen_fft * high_pass
        target_high_freq = target_fft * high_pass

        # Compute loss on high-frequency components
        return F.mse_loss(
            torch.abs(gen_high_freq), torch.abs(target_high_freq)
        )
