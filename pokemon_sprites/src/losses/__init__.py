"""
Loss functions for Pokemon sprite generation.

This module provides a comprehensive collection of loss functions designed
specifically for combating blurriness and maintaining high-quality pixel art
generation in image-to-image translation tasks.

Available Loss Functions:
    - AntiBlurLoss: Edge preservation and high-frequency enhancement
    - PixelArtLoss: Pixel-art style preservation with sharp edges
    - ImprovedPerceptualLoss: VGG-based perceptual quality assessment
    - CombinedAntiBlurLoss: Comprehensive anti-blur solution

Factory Functions:
    - create_loss: Create loss functions by name with configuration
    - get_available_losses: Get list of available loss function names
"""

from typing import Any, Dict, List

from .anti_blur_loss import AntiBlurLoss
from .base_loss import BaseLoss
from .combined_loss import CombinedAntiBlurLoss
from .perceptual_loss import ImprovedPerceptualLoss
from .pixel_art_loss import PixelArtLoss

# Export all loss classes
__all__ = [
    "BaseLoss",
    "AntiBlurLoss",
    "PixelArtLoss",
    "ImprovedPerceptualLoss",
    "CombinedAntiBlurLoss",
    "create_loss",
    "get_available_losses",
]

# Registry of available loss functions
LOSS_REGISTRY = {
    "anti_blur": AntiBlurLoss,
    "pixel_art": PixelArtLoss,
    "perceptual": ImprovedPerceptualLoss,
    "combined": CombinedAntiBlurLoss,
}


def create_loss(loss_name: str, **kwargs) -> BaseLoss:
    """
    Factory function to create loss functions by name.

    Args:
        loss_name: Name of the loss function to create
        **kwargs: Additional arguments to pass to the loss constructor

    Returns:
        Initialized loss function

    Raises:
        ValueError: If loss_name is not recognized

    Examples:
        >>> # Create anti-blur loss with custom weights
        >>> loss = create_loss("anti_blur", alpha=1.5, beta=0.8)

        >>> # Create combined loss with custom configuration
        >>> loss = create_loss(
        ...     "combined", l1_weight=50.0, perceptual_weight=2.0
        ... )
    """
    if loss_name not in LOSS_REGISTRY:
        available = ", ".join(LOSS_REGISTRY.keys())
        raise ValueError(
            f"Unknown loss function '{loss_name}'. "
            f"Available options: {available}"
        )

    loss_class = LOSS_REGISTRY[loss_name]
    return loss_class(**kwargs)


def get_available_losses() -> List[str]:
    """
    Get list of available loss function names.

    Returns:
        List of available loss function names
    """
    return list(LOSS_REGISTRY.keys())


def create_loss_from_config(config: Dict[str, Any]) -> BaseLoss:
    """
    Create loss function from configuration dictionary.

    Args:
        config: Configuration dictionary with 'type' and optional parameters

    Returns:
        Initialized loss function

    Example:
        >>> config = {
        ...     "type": "combined",
        ...     "l1_weight": 100.0,
        ...     "anti_blur_weight": 15.0,
        ...     "pixel_art_weight": 8.0,
        ...     "perceptual_weight": 2.0
        ... }
        >>> loss = create_loss_from_config(config)
    """
    if "type" not in config:
        raise ValueError("Configuration must include 'type' field")

    loss_type = config.pop("type")
    return create_loss(loss_type, **config)


# Backward compatibility aliases
AntiBlurLoss = AntiBlurLoss
PixelArtLoss = PixelArtLoss
ImprovedPerceptualLoss = ImprovedPerceptualLoss
CombinedAntiBlurLoss = CombinedAntiBlurLoss
