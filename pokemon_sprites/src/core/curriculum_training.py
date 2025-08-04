"""
Curriculum learning management for Pokemon sprite generation.

This module manages progressive training with increasing input complexity
to improve model convergence and final performance.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

from core.logging_config import get_logger

# Add src to path
current_dir = Path(__file__).parent
src_path = str(current_dir.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)


logger = get_logger(__name__)


class CurriculumTrainingManager:
    """Manages curriculum learning with progressive input scaling."""

    def __init__(self, config: Dict[str, Any], data_dir: Path):
        """
        Initialize curriculum training manager.

        Args:
            config: Training configuration dictionary
            data_dir: Root data directory
        """
        self.config = config
        self.data_dir = data_dir
        self.input_scales = [128, 192, 256]
        self.current_scale_idx = 0
        self.logger = get_logger(self.__class__.__name__)

    def get_current_scale(self) -> int:
        """Get the current training scale."""
        return self.input_scales[self.current_scale_idx]

    def get_curriculum_phase_info(self) -> Dict[str, Any]:
        """
        Get information about the current curriculum phase.

        Returns:
            Dictionary with phase information
        """
        current_scale = self.get_current_scale()
        return {
            "phase": self.current_scale_idx + 1,
            "total_phases": len(self.input_scales),
            "current_scale": current_scale,
            "scales": self.input_scales,
            "progress": (self.current_scale_idx + 1) / len(self.input_scales),
        }

    def should_use_curriculum(self, model_name: str) -> bool:
        """
        Determine if curriculum learning should be used for a given model.

        Args:
            model_name: Name of the model architecture

        Returns:
            True if curriculum learning is recommended
        """
        curriculum_models = {
            "sprite-optimized": True,
            "transformer-enhanced": True,
            "lightweight-baseline": False,  # Too simple to benefit from curriculum
        }

        return curriculum_models.get(model_name, False)

    def get_phase_epochs(self, total_epochs: int) -> List[int]:
        """
        Distribute epochs across curriculum phases.

        Args:
            total_epochs: Total training epochs

        Returns:
            List of epochs per phase
        """
        if not self.should_use_curriculum(self.config.get("model_name", "")):
            return [total_epochs]

        # Progressive epoch allocation: more epochs for higher resolutions
        phase_ratios = [0.3, 0.3, 0.4]  # 30%, 30%, 40%
        phase_epochs = []

        for ratio in phase_ratios:
            epochs = max(1, int(total_epochs * ratio))
            phase_epochs.append(epochs)

        # Adjust to ensure total matches
        total_allocated = sum(phase_epochs)
        if total_allocated != total_epochs:
            phase_epochs[-1] += total_epochs - total_allocated

        return phase_epochs

    def advance_curriculum(self) -> bool:
        """
        Advance to next curriculum phase.

        Returns:
            True if advanced to next phase, False if curriculum complete
        """
        if self.current_scale_idx < len(self.input_scales) - 1:
            self.current_scale_idx += 1
            scale = self.input_scales[self.current_scale_idx]
            self.logger.info(
                f"Advanced to curriculum phase {self.current_scale_idx + 1}: {scale}px"
            )
            return True

        self.logger.info("Curriculum learning complete")
        return False

    def reset_curriculum(self) -> None:
        """Reset curriculum to the beginning."""
        self.current_scale_idx = 0
        self.logger.info("Curriculum reset to phase 1")

    def get_curriculum_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get curriculum learning configuration for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Curriculum configuration
        """
        configs = {
            "lightweight-baseline": {
                "use_curriculum": False,
                "reason": "Lightweight model converges quickly without curriculum",
            },
            "sprite-optimized": {
                "use_curriculum": True,
                "scales": [128, 192, 256],
                "epoch_ratios": [0.3, 0.3, 0.4],
                "reason": "Benefits from progressive resolution increase",
            },
            "transformer-enhanced": {
                "use_curriculum": True,
                "scales": [128, 192, 256],
                "epoch_ratios": [0.25, 0.35, 0.4],
                "reason": "Complex attention mechanisms benefit from curriculum",
            },
        }

        default_config = {
            "use_curriculum": False,
            "scales": [256],
            "epoch_ratios": [1.0],
            "reason": "Unknown model, using standard training",
        }

        return configs.get(model_name, default_config)

    def create_temp_dataset_structure(self, scale: int) -> Path:
        """
        Create temporary dataset structure for current curriculum phase.

        Args:
            scale: Target image scale

        Returns:
            Path to temporary dataset directory
        """
        scale_dir = self.data_dir / f"input_{scale}"
        temp_data_dir = scale_dir.parent / f"curriculum_temp_{scale}"

        train_input_dir = temp_data_dir / "train" / "input"
        train_target_dir = temp_data_dir / "train" / "target"
        val_input_dir = temp_data_dir / "val" / "input"
        val_target_dir = temp_data_dir / "val" / "target"

        # Create directories
        for directory in [
            train_input_dir,
            train_target_dir,
            val_input_dir,
            val_target_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        return temp_data_dir

    def cleanup_temp_structures(self) -> None:
        """Clean up temporary curriculum dataset structures."""
        for scale in self.input_scales:
            temp_dir = self.data_dir / f"curriculum_temp_{scale}"
            if temp_dir.exists():
                import shutil

                shutil.rmtree(temp_dir)
                self.logger.debug(
                    f"Cleaned up temporary directory: {temp_dir}"
                )

    def is_curriculum_complete(self) -> bool:
        """Check if curriculum learning is complete."""
        return self.current_scale_idx >= len(self.input_scales) - 1
