"""
Base loss class for Pokemon sprite generation.

This module provides a base class that can be extended by specific loss
implementations, ensuring consistent interfaces and logging capabilities.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BaseLoss(nn.Module, ABC):
    """
    Base class for all loss functions.

    Provides common functionality including logging and parameter validation.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize base loss.

        Args:
            name: Name for logging purposes
        """
        super().__init__()
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"pokemon_sprites.losses.{self.name}")

    @abstractmethod
    def forward(
        self, generated: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss between generated and target images.

        Args:
            generated: Generated images [B, C, H, W]
            target: Target images [B, C, H, W]

        Returns:
            Loss value
        """

    def validate_inputs(
        self, generated: torch.Tensor, target: torch.Tensor
    ) -> None:
        """
        Validate input tensors.

        Args:
            generated: Generated images tensor
            target: Target images tensor

        Raises:
            ValueError: If inputs are invalid
        """
        if generated.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: generated {generated.shape} vs target "
                f"{target.shape}"
            )

        if len(generated.shape) != 4:
            raise ValueError(
                f"Expected 4D tensor [B, C, H, W], got {len(generated.shape)}D"
            )

    def log_loss_info(
        self, loss_value: float, extra_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log loss information for debugging.

        Args:
            loss_value: Computed loss value
            extra_info: Additional information to log
        """
        log_msg = f"{self.name} loss: {loss_value:.6f}"

        if extra_info:
            extra_str = ", ".join([f"{k}: {v}" for k, v in extra_info.items()])
            log_msg += f" ({extra_str})"

        self.logger.debug(log_msg)
