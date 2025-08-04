"""
Transfer learning utilities for Pokemon sprite generation.

This module provides capabilities for loading and applying pretrained weights
to improve model training convergence.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from core.logging_config import get_logger

# Add src to path
current_dir = Path(__file__).parent
src_path = str(current_dir.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)


logger = get_logger(__name__)


class TransferLearningManager:
    """Manages transfer learning capabilities for model training."""

    def __init__(self, device: torch.device):
        """
        Initialize transfer learning manager.

        Args:
            device: PyTorch device for model operations
        """
        self.device = device
        self.pretrained_urls = {
            "edges2shoes": "https://storage.googleapis.com/edges2shoes_netG.pth",
            "facades": "https://storage.googleapis.com/facades_netG.pth",
            "maps": "https://storage.googleapis.com/maps_netG.pth",
        }

    def load_pretrained_weights(
        self, model: nn.Module, pretrained_path: Path
    ) -> bool:
        """
        Load pretrained weights with compatibility checking.

        Args:
            model: PyTorch model to load weights into
            pretrained_path: Path to pretrained weights file

        Returns:
            True if weights were successfully loaded, False otherwise
        """
        if not pretrained_path.exists():
            logger.warning(f"Pretrained weights not found: {pretrained_path}")
            return False

        try:
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            model_dict = model.state_dict()

            # Filter compatible parameters
            pretrained_dict = {
                k: v
                for k, v in checkpoint.get("generator", checkpoint).items()
                if k in model_dict and v.shape == model_dict[k].shape
            }

            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            loaded_params = len(pretrained_dict)
            total_params = len(model_dict)

            logger.info(
                f"Transfer learning: Loaded {loaded_params}/{total_params} "
                f"compatible parameters from {pretrained_path.name}"
            )
            return loaded_params > 0

        except Exception as e:
            logger.error(f"Failed to load pretrained weights: {e}")
            return False

    def apply_progressive_unfreezing(
        self, model: nn.Module, epoch: int, total_epochs: int
    ) -> None:
        """
        Apply progressive unfreezing strategy during training.

        Args:
            model: Model to apply unfreezing to
            epoch: Current training epoch
            total_epochs: Total number of training epochs
        """
        if not hasattr(model, "backbone"):
            return

        backbone = getattr(model, "backbone")
        if not isinstance(backbone, nn.Module):
            return

        # Progressive unfreezing: start unfreezing layers after 25% of training
        unfreeze_start = total_epochs * 0.25

        if epoch < unfreeze_start:
            # Keep all backbone layers frozen
            for param in backbone.parameters():
                param.requires_grad = False
        else:
            # Progressive unfreezing of backbone layers
            progress = (epoch - unfreeze_start) / (
                total_epochs - unfreeze_start
            )

            layers = list(backbone.children())
            if layers:
                layers_to_unfreeze = int(len(layers) * progress)

                for i, layer in enumerate(layers):
                    if i >= len(layers) - layers_to_unfreeze:
                        for param in layer.parameters():
                            param.requires_grad = True
                    else:
                        for param in layer.parameters():
                            param.requires_grad = False

            logger.debug(
                f"Progressive unfreezing: {progress:.2%} of backbone layers unfrozen"
            )

    def get_transfer_learning_config(self, model_name: str) -> Optional[Dict]:
        """
        Get recommended transfer learning configuration for a model.

        Args:
            model_name: Name of the model architecture

        Returns:
            Transfer learning configuration or None if not applicable
        """
        configs = {
            "lightweight-baseline": {
                "use_transfer": False,
                "progressive_unfreezing": False,
                "reason": "Lightweight model trains well from scratch",
            },
            "sprite-optimized": {
                "use_transfer": True,
                "progressive_unfreezing": True,
                "recommended_source": "edges2shoes",
                "unfreeze_after_epoch_ratio": 0.3,
                "reason": "Benefits from pretrained features for edge detection",
            },
            "transformer-enhanced": {
                "use_transfer": True,
                "progressive_unfreezing": True,
                "recommended_source": "facades",
                "unfreeze_after_epoch_ratio": 0.25,
                "reason": "Complex architecture benefits from pretrained initialization",
            },
        }

        return configs.get(model_name)

    def should_use_transfer_learning(self, model_config: Dict) -> bool:
        """
        Determine if transfer learning should be used for a given model.

        Args:
            model_config: Model configuration dictionary

        Returns:
            True if transfer learning is recommended
        """
        model_name = model_config.get("name", "")
        config = self.get_transfer_learning_config(model_name)

        if config is None:
            return False

        return config.get("use_transfer", False)
