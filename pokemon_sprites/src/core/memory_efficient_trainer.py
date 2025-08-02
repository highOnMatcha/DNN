"""
Memory-efficient training enhancements for Pokemon sprite generation.

This module provides memory-efficient improvements including gradient
accumulation, mixed precision training, advanced loss functions, and efficient
attention mechanisms.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.amp.autocast_mode import autocast

from core.anti_blur_losses import CombinedAntiBlurLoss, ImprovedPerceptualLoss
from core.logging_config import get_logger
from core.trainer import PokemonSpriteTrainer


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features for better visual quality."""

    def __init__(
        self,
        layers: list = [3, 8, 15, 22],
        weights: list = [1.0, 1.0, 1.0, 1.0],
    ):
        super().__init__()
        # Use new weights parameter instead of deprecated pretrained
        try:
            from torchvision.models import VGG16_Weights

            self.vgg = models.vgg16(
                weights=VGG16_Weights.IMAGENET1K_V1
            ).features
        except ImportError:
            # Fallback for older torchvision versions
            self.vgg = models.vgg16(pretrained=True).features

        self.layers = layers
        self.weights = weights

        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

        # Move to half precision for memory efficiency
        self.vgg = self.vgg.half()

    def forward(
        self, input_img: torch.Tensor, target_img: torch.Tensor
    ) -> torch.Tensor:
        # Convert to half precision
        input_img = input_img.half()
        target_img = target_img.half()

        loss = torch.tensor(0.0, device=input_img.device, dtype=torch.float32)
        x = input_img
        y = target_img

        for i, layer in enumerate(list(self.vgg.children())):
            x = layer(x)
            y = layer(y)

            if i in self.layers:
                weight = self.weights[self.layers.index(i)]
                loss += weight * F.mse_loss(x, y)

        return loss


class EfficientAttention(nn.Module):
    """Lightweight attention mechanism for feature enhancement."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channels = channels
        self.reduction = reduction

        # Channel attention
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)

        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)

        avg_out = self.fc2(F.relu(self.fc1(avg_pool)))
        max_out = self.fc2(F.relu(self.fc1(max_pool)))
        channel_att = torch.sigmoid(avg_out + max_out)

        x = x * channel_att

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(
            self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        )

        x = x * spatial_att
        return x


class MemoryEfficientPokemonTrainer(PokemonSpriteTrainer):
    """Enhanced trainer with memory-efficient improvements."""

    def __init__(self, model_config, training_config, wandb_run=None):
        super().__init__(model_config, training_config, wandb_run)

        self.logger = get_logger(f"{__name__}.{model_config.name}")

        # Memory efficiency settings
        self.use_mixed_precision = getattr(
            training_config, "mixed_precision", True
        )
        self.gradient_accumulation_steps = getattr(
            training_config, "gradient_accumulation_steps", 1
        )
        self.use_perceptual_loss = getattr(
            training_config, "use_perceptual_loss", True
        )
        self.perceptual_weight = getattr(
            training_config, "perceptual_weight", 0.1
        )

        # Anti-blur settings
        self.use_anti_blur_loss = getattr(
            training_config, "use_anti_blur_loss", False
        )
        self.anti_blur_weight = getattr(
            training_config, "anti_blur_weight", 10.0
        )
        self.pixel_art_weight = getattr(
            training_config, "pixel_art_weight", 5.0
        )

        # Initialize mixed precision scaler
        if self.use_mixed_precision:
            try:
                # Try new API first (PyTorch 2.0+)
                from torch.amp.grad_scaler import GradScaler

                self.scaler = GradScaler("cuda")
            except ImportError:
                # Fallback to old API
                from torch.cuda.amp import GradScaler

                self.scaler = GradScaler()
            self.logger.info("Mixed precision training enabled")

        # Initialize loss functions
        self.combined_loss = None
        self.perceptual_loss = None

        if self.use_perceptual_loss:
            if self.use_anti_blur_loss:
                # Use combined anti-blur loss for maximum sharpness
                self.combined_loss = CombinedAntiBlurLoss(
                    l1_weight=100.0,
                    anti_blur_weight=self.anti_blur_weight,
                    pixel_art_weight=self.pixel_art_weight,
                    perceptual_weight=self.perceptual_weight,
                ).to(self.device)
                self.logger.info(
                    f"Anti-blur loss enabled with weights: "
                    f"anti_blur={self.anti_blur_weight}, "
                    f"pixel_art={self.pixel_art_weight}"
                )
            else:
                # Use improved perceptual loss only
                self.perceptual_loss = ImprovedPerceptualLoss().to(self.device)
                self.logger.info(
                    f"Improved perceptual loss enabled with weight "
                    f"{self.perceptual_weight}"
                )

        # Add attention modules to existing models
        self._add_attention_modules()

        self.logger.info(
            f"Gradient accumulation steps: {self.gradient_accumulation_steps}"
        )

    def _add_attention_modules(self):
        """Add efficient attention modules to existing models."""
        try:
            if (
                hasattr(self, "models")
                and isinstance(self.models, dict)
                and "generator" in self.models
            ):
                # For Pix2Pix, try to add attention to generator
                self.models["generator"]
                # Add attention as a separate module (will be manually
                # integrated)
                self.attention_module = EfficientAttention(64).to(self.device)
                self.logger.info("Added attention module with 64 channels")
        except Exception as e:
            self.logger.warning(f"Could not add attention modules: {e}")
            self.attention_module = None

    def _compute_perceptual_loss(
        self, generated: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute perceptual loss with memory efficiency."""
        if not self.use_perceptual_loss:
            return torch.tensor(0.0, device=self.device)

        if self.use_anti_blur_loss and self.combined_loss is not None:
            # Use combined anti-blur loss
            total_loss, loss_components = self.combined_loss(generated, target)

            # Log individual components if available
            if hasattr(self, "wandb_run") and self.wandb_run:
                self.wandb_run.log(loss_components)

            return total_loss - 100.0 * F.l1_loss(
                generated, target
            )  # Subtract base L1 to avoid double counting
        elif self.perceptual_loss is not None:
            # Use improved perceptual loss only
            return self.perceptual_loss(generated, target)
        else:
            return torch.tensor(0.0, device=self.device)

    def _train_epoch_pix2pix_efficient(
        self, train_loader, epoch: int
    ) -> Dict[str, float]:
        """
        Memory-efficient Pix2Pix training with gradient accumulation and
        mixed precision.
        """
        total_g_loss = 0.0
        total_d_loss = 0.0
        total_l1_loss = 0.0
        total_perceptual_loss = 0.0
        num_batches = len(train_loader)

        # Ensure models is a dictionary for Pix2Pix
        if not isinstance(self.models, dict):
            raise ValueError(
                "Pix2Pix training requires models to be a dictionary"
            )

        # Reset gradients
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

        for batch_idx, (input_imgs, target_imgs) in enumerate(train_loader):
            input_imgs = input_imgs.to(self.device, non_blocking=True)
            target_imgs = target_imgs.to(self.device, non_blocking=True)
            batch_size = input_imgs.size(0)

            # Mixed precision context
            with autocast("cuda", enabled=self.use_mixed_precision):
                # Train Discriminator
                with autocast("cuda", enabled=self.use_mixed_precision):
                    # Real images
                    real_output = self.models["discriminator"](
                        input_imgs, target_imgs
                    )
                    output_h, output_w = (
                        real_output.shape[2],
                        real_output.shape[3],
                    )
                    real_labels = torch.ones(
                        batch_size,
                        1,
                        output_h,
                        output_w,
                        device=self.device,
                        dtype=real_output.dtype,
                    )
                    fake_labels = torch.zeros(
                        batch_size,
                        1,
                        output_h,
                        output_w,
                        device=self.device,
                        dtype=real_output.dtype,
                    )

                    d_real_loss = self.loss_functions["adversarial"](
                        real_output, real_labels
                    )

                    # Fake images
                    generated_imgs = self.models["generator"](input_imgs)
                    fake_output = self.models["discriminator"](
                        input_imgs, generated_imgs.detach()
                    )
                    d_fake_loss = self.loss_functions["adversarial"](
                        fake_output, fake_labels
                    )

                    d_loss = (d_real_loss + d_fake_loss) * 0.5

                # Scale discriminator loss for gradient accumulation
                d_loss = d_loss / self.gradient_accumulation_steps

            # Backward pass for discriminator
            if self.use_mixed_precision:
                self.scaler.scale(d_loss).backward()
            else:
                d_loss.backward()

            # Train Generator
            with autocast("cuda", enabled=self.use_mixed_precision):
                generated_imgs = self.models["generator"](input_imgs)
                fake_output = self.models["discriminator"](
                    input_imgs, generated_imgs
                )

                output_h, output_w = fake_output.shape[2], fake_output.shape[3]
                real_labels_gen = torch.ones(
                    batch_size,
                    1,
                    output_h,
                    output_w,
                    device=self.device,
                    dtype=fake_output.dtype,
                )

                g_adversarial_loss = self.loss_functions["adversarial"](
                    fake_output, real_labels_gen
                )
                g_l1_loss = self.loss_functions["l1"](
                    generated_imgs, target_imgs
                )

                # Perceptual loss
                g_perceptual_loss = self._compute_perceptual_loss(
                    generated_imgs, target_imgs
                )

                g_loss = (
                    g_adversarial_loss
                    + self.training_config.lambda_l1 * g_l1_loss
                    + self.perceptual_weight * g_perceptual_loss
                )

                # Scale generator loss for gradient accumulation
                g_loss = g_loss / self.gradient_accumulation_steps

            # Backward pass for generator
            if self.use_mixed_precision:
                self.scaler.scale(g_loss).backward()
            else:
                g_loss.backward()

            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_mixed_precision:
                    self.scaler.step(self.optimizers["discriminator"])
                    self.scaler.step(self.optimizers["generator"])
                    self.scaler.update()
                else:
                    self.optimizers["discriminator"].step()
                    self.optimizers["generator"].step()

                # Reset gradients
                for optimizer in self.optimizers.values():
                    optimizer.zero_grad()

            # Accumulate losses (scale back)
            total_g_loss += g_loss.item() * self.gradient_accumulation_steps
            total_d_loss += d_loss.item() * self.gradient_accumulation_steps
            total_l1_loss += g_l1_loss.item()
            total_perceptual_loss += g_perceptual_loss.item()

            # Log batch progress
            if batch_idx % self.training_config.log_frequency == 0:
                self.progress_logger.log_batch(
                    epoch,
                    batch_idx,
                    num_batches,
                    {
                        "g_loss": g_loss.item()
                        * self.gradient_accumulation_steps,
                        "d_loss": d_loss.item()
                        * self.gradient_accumulation_steps,
                        "l1_loss": g_l1_loss.item(),
                        "perceptual_loss": g_perceptual_loss.item(),
                    },
                    total_epochs=self.training_config.epochs,
                )

                if self.wandb_run:
                    self.wandb_run.log(
                        {
                            "g_loss": g_loss.item()
                            * self.gradient_accumulation_steps,
                            "d_loss": d_loss.item()
                            * self.gradient_accumulation_steps,
                            "l1_loss": g_l1_loss.item(),
                            "perceptual_loss": g_perceptual_loss.item(),
                            "epoch": epoch,
                            "batch": batch_idx,
                        }
                    )

        return {
            "g_loss": total_g_loss / num_batches,
            "d_loss": total_d_loss / num_batches,
            "l1_loss": total_l1_loss / num_batches,
            "perceptual_loss": total_perceptual_loss / num_batches,
        }

    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Enhanced training epoch with memory efficiency."""
        if isinstance(self.models, dict):
            for model in self.models.values():
                model.train()
        else:
            self.models.train()

        if self.model_config.architecture == "pix2pix":
            return self._train_epoch_pix2pix_efficient(train_loader, epoch)
        else:
            # Fall back to parent implementation for other architectures
            return super().train_epoch(train_loader, epoch)

    def save_memory_stats(self):
        """Log current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            self.logger.info(
                f"GPU Memory - Allocated: {allocated:.2f}GB, "
                f"Cached: {cached:.2f}GB"
            )


def create_memory_efficient_trainer(
    model_config, training_config, wandb_run=None
):
    """Factory function to create memory-efficient trainer."""
    return MemoryEfficientPokemonTrainer(
        model_config, training_config, wandb_run
    )
