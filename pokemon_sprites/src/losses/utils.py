"""
Utility functions for loss computation.

This module provides shared utilities used across different loss functions,
including edge detection filters and common mathematical operations.
"""

import torch
import torch.nn.functional as F


def create_sobel_filters(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create Sobel edge detection filters for X and Y directions.

    Args:
        device: Device to create tensors on

    Returns:
        Tuple of (sobel_x, sobel_y) filters
    """
    # Sobel X filter
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32,
        device=device,
    ).view(1, 1, 3, 3)

    # Sobel Y filter
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=torch.float32,
        device=device,
    ).view(1, 1, 3, 3)

    return sobel_x, sobel_y


def compute_edges(
    image: torch.Tensor, sobel_x: torch.Tensor, sobel_y: torch.Tensor
) -> torch.Tensor:
    """
    Compute edge magnitude using Sobel operators.

    Args:
        image: Input image tensor [B, C, H, W]
        sobel_x: Sobel X filter
        sobel_y: Sobel Y filter

    Returns:
        Edge magnitude tensor
    """
    # Expand filters for all channels
    channels = image.shape[1]
    sobel_x_expanded = sobel_x.expand(channels, 1, 3, 3)
    sobel_y_expanded = sobel_y.expand(channels, 1, 3, 3)

    # Compute edges
    edges_x = F.conv2d(image, sobel_x_expanded, padding=1, groups=channels)
    edges_y = F.conv2d(image, sobel_y_expanded, padding=1, groups=channels)

    # Compute magnitude
    edges = torch.sqrt(edges_x**2 + edges_y**2 + 1e-8)

    return edges


def rgb_to_grayscale(image: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB image to grayscale using standard luminance weights.

    Args:
        image: RGB image tensor [B, C, H, W] where C=3

    Returns:
        Grayscale image tensor [B, H, W]
    """
    if image.shape[1] != 3:
        raise ValueError(f"Expected 3 channels, got {image.shape[1]}")

    # Standard luminance weights
    weights = torch.tensor(
        [0.299, 0.587, 0.114], device=image.device, dtype=image.dtype
    )
    grayscale = torch.sum(image * weights.view(1, 3, 1, 1), dim=1)

    return grayscale


def create_high_pass_filter(
    height: int, width: int, device: torch.device, threshold: float = 0.1
) -> torch.Tensor:
    """
    Create a high-pass frequency filter.

    Args:
        height: Filter height
        width: Filter width
        device: Device to create tensor on
        threshold: High-pass threshold (fraction of max frequency)

    Returns:
        High-pass filter mask
    """
    center_h, center_w = height // 2, width // 2

    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )

    distance = torch.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)
    high_pass = (distance > min(height, width) * threshold).float()

    return high_pass
