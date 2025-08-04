"""
Training orchestration for Pokemon sprite generation models.

This module provides comprehensive training functionality for image-to-image
translation models with support for different architectures, loss functions,
and training strategies including memory-efficient enhancements.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import wandb
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from config.settings import ModelConfig, TrainingConfig
from core.logging_config import (
    TrainingProgressLogger,
    get_logger,
    log_model_summary,
)
from models import count_parameters, count_total_parameters, create_model


class PokemonSpriteTrainer:
    """
    Comprehensive trainer for Pokemon sprite generation models.

    Supports different architectures (Pix2Pix, U-Net, CycleGAN) with
    appropriate loss functions, optimizers, and training strategies.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        wandb_run: Optional[Any] = None,
    ):
        """
        Initialize trainer with configurations.

        Args:
            model_config: Model architecture configuration.
            training_config: Training hyperparameters configuration.
            wandb_run: Optional WandB run for experiment tracking.
        """
        self.model_config = model_config
        self.training_config = training_config
        self.wandb_run = wandb_run

        # Setup logging
        self.logger = get_logger(f"{__name__}.{model_config.name}")
        self.progress_logger = TrainingProgressLogger()

        # Memory efficiency settings
        self.use_mixed_precision = getattr(
            training_config, "mixed_precision", True
        )
        self.gradient_accumulation_steps = getattr(
            training_config, "gradient_accumulation_steps", 1
        )
        self.use_perceptual_loss = getattr(
            training_config, "use_perceptual_loss", False
        )
        self.perceptual_weight = getattr(
            training_config, "perceptual_weight", 0.1
        )

        # Setup device
        self.device = self._setup_device()
        self.logger.info(f"Using device: {self.device}")

        # Initialize mixed precision scaler
        if self.use_mixed_precision and self.device.type == "cuda":
            try:
                # Try new API first (PyTorch 2.0+)
                from torch.amp.grad_scaler import GradScaler

                self.scaler = GradScaler("cuda")
            except ImportError:
                # Fallback to old API
                from torch.cuda.amp import GradScaler

                self.scaler = GradScaler()
            self.logger.info("Mixed precision training enabled")
        else:
            self.scaler = None

        # Initialize models
        self.models = self._create_models()
        self.optimizers = self._create_optimizers()
        self.schedulers = self._create_schedulers()

        # Setup loss functions
        self.loss_functions = self._setup_loss_functions()

        # Initialize perceptual loss if enabled
        self.perceptual_loss = None
        if self.use_perceptual_loss:
            self.perceptual_loss = self._create_perceptual_loss()
            self.logger.info(
                f"Perceptual loss enabled with weight {self.perceptual_weight}"
            )

        # Training state
        self.current_epoch = 0
        self.best_metrics = {}
        self.training_history = {"train": [], "val": []}

        # Create output directories
        self.output_dir = Path(model_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.samples_dir = self.output_dir / "samples"
        self.samples_dir.mkdir(exist_ok=True)

        self.logger.info(f"Trainer initialized for {model_config.name}")
        self.logger.info(f"Architecture: {model_config.architecture}")
        self.logger.info(f"Output directory: {self.output_dir}")
        if self.gradient_accumulation_steps > 1:
            steps = self.gradient_accumulation_steps
            msg = f"Gradient accumulation steps: {steps}"
            self.logger.info(msg)

        # Log model summary
        if model_config.architecture == "unet":
            log_model_summary(
                self.models,
                (3, training_config.image_size, training_config.image_size),
            )
        else:
            total_params = count_total_parameters(self.models)
            trainable_params = count_parameters(self.models)
            frozen_params = total_params - trainable_params

            # For pretrained models, show detailed breakdown
            if model_config.architecture in [
                "pix2pix_pretrained",
                "pix2pix_controlnet",
                "pix2pix_vit",
            ]:
                self.logger.info(
                    f"Total parameters: {total_params:,} "
                    f"(Trainable: {trainable_params:,}, "
                    f"Frozen: {frozen_params:,})"
                )
                if isinstance(self.models, dict):
                    gen_total = sum(
                        p.numel()
                        for p in self.models["generator"].parameters()
                    )
                    gen_trainable = sum(
                        p.numel()
                        for p in self.models["generator"].parameters()
                        if p.requires_grad
                    )
                    gen_frozen = gen_total - gen_trainable
                    disc_trainable = sum(
                        p.numel()
                        for p in self.models["discriminator"].parameters()
                        if p.requires_grad
                    )

                    self.logger.info(
                        f"Generator: {gen_total:,} total "
                        f"({gen_trainable:,} trainable, {gen_frozen:,} frozen)"
                    )
                    self.logger.info(
                        f"Discriminator: {disc_trainable:,} trainable"
                    )
            else:
                self.logger.info(
                    f"Total trainable parameters: {trainable_params:,}"
                )

    def _setup_device(self) -> torch.device:
        """Setup compute device based on configuration."""
        if self.training_config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.training_config.device)

        return device

    def _create_perceptual_loss(self):
        """Create perceptual loss function using VGG features."""

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

            def forward(
                self, input_img: torch.Tensor, target_img: torch.Tensor
            ) -> torch.Tensor:
                loss = torch.tensor(
                    0.0, device=input_img.device, dtype=torch.float32
                )
                x = input_img
                y = target_img

                for i, layer in enumerate(list(self.vgg.children())):
                    x = layer(x)
                    y = layer(y)

                    if i in self.layers:
                        weight = self.weights[self.layers.index(i)]
                        loss += weight * F.mse_loss(x, y)

                return loss

        return PerceptualLoss().to(self.device)

    def _compute_perceptual_loss(
        self, generated: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute perceptual loss with memory efficiency."""
        if not self.use_perceptual_loss or self.perceptual_loss is None:
            return torch.tensor(0.0, device=self.device)

        return self.perceptual_weight * self.perceptual_loss(generated, target)

    def _create_models(self) -> Union[nn.Module, Dict[str, nn.Module]]:
        """Create model(s) based on architecture."""
        models = create_model(asdict(self.model_config))

        if isinstance(models, dict):
            # Multiple models (GAN architectures)
            for name, model in models.items():
                model.to(self.device)
                if torch.cuda.device_count() > 1:
                    model = nn.DataParallel(model)
                models[name] = model
        else:
            # Single model (U-Net)
            models.to(self.device)
            if torch.cuda.device_count() > 1:
                models = nn.DataParallel(models)

        return models

    def _create_optimizers(self) -> Dict[str, optim.Optimizer]:
        """Create optimizers for model(s)."""
        optimizers = {}

        if isinstance(self.models, dict):
            # GAN architectures
            if self.model_config.architecture in [
                "pix2pix",
                "pix2pix_pretrained",
                "pix2pix_controlnet",
                "pix2pix_vit",
            ]:
                # For pretrained models, only optimize trainable parameters
                if self.model_config.architecture in [
                    "pix2pix_pretrained",
                    "pix2pix_controlnet",
                    "pix2pix_vit",
                ]:
                    gen_params = [
                        p
                        for p in self.models["generator"].parameters()
                        if p.requires_grad
                    ]
                else:
                    gen_params = self.models["generator"].parameters()

                optimizers["generator"] = optim.Adam(
                    gen_params,
                    lr=self.training_config.learning_rate,
                    betas=(
                        self.training_config.beta1,
                        self.training_config.beta2,
                    ),
                    weight_decay=self.training_config.weight_decay,
                )
                optimizers["discriminator"] = optim.Adam(
                    self.models["discriminator"].parameters(),
                    lr=self.training_config.learning_rate,
                    betas=(
                        self.training_config.beta1,
                        self.training_config.beta2,
                    ),
                    weight_decay=self.training_config.weight_decay,
                )

            elif self.model_config.architecture == "cyclegan":
                optimizers["generator"] = optim.Adam(
                    list(self.models["generator_A2B"].parameters())
                    + list(self.models["generator_B2A"].parameters()),
                    lr=self.training_config.learning_rate,
                    betas=(
                        self.training_config.beta1,
                        self.training_config.beta2,
                    ),
                    weight_decay=self.training_config.weight_decay,
                )
                optimizers["discriminator"] = optim.Adam(
                    list(self.models["discriminator_A"].parameters())
                    + list(self.models["discriminator_B"].parameters()),
                    lr=self.training_config.learning_rate,
                    betas=(
                        self.training_config.beta1,
                        self.training_config.beta2,
                    ),
                    weight_decay=self.training_config.weight_decay,
                )
        else:
            # Single model (U-Net)
            optimizers["model"] = optim.Adam(
                self.models.parameters(),
                lr=self.training_config.learning_rate,
                betas=(self.training_config.beta1, self.training_config.beta2),
                weight_decay=self.training_config.weight_decay,
            )

        return optimizers

    def _create_schedulers(self) -> Dict[str, Any]:
        """Create learning rate schedulers."""
        schedulers = {}

        for name, optimizer in self.optimizers.items():
            schedulers[name] = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10
            )

        return schedulers

    def _setup_loss_functions(self) -> Dict[str, nn.Module]:
        """Setup loss functions based on architecture."""
        loss_functions = {}

        if self.model_config.architecture == "unet":
            loss_functions["mse"] = nn.MSELoss()
            loss_functions["l1"] = nn.L1Loss()

        elif self.model_config.architecture in [
            "pix2pix",
            "pix2pix_pretrained",
            "pix2pix_controlnet",
            "pix2pix_vit",
        ]:
            loss_functions["adversarial"] = nn.BCEWithLogitsLoss()
            loss_functions["l1"] = nn.L1Loss()

        elif self.model_config.architecture == "cyclegan":
            loss_functions["adversarial"] = nn.MSELoss()
            loss_functions["cycle"] = nn.L1Loss()
            loss_functions["identity"] = nn.L1Loss()

        return loss_functions

    def train_epoch(
        self, train_loader: DataLoader, epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        if isinstance(self.models, dict):
            for model in self.models.values():
                model.train()
        else:
            self.models.train()

        epoch_losses = {}

        if self.model_config.architecture == "unet":
            epoch_losses = self._train_epoch_unet(train_loader, epoch)
        elif self.model_config.architecture in [
            "pix2pix",
            "pix2pix_pretrained",
            "pix2pix_controlnet",
            "pix2pix_vit",
        ]:
            epoch_losses = self._train_epoch_pix2pix(train_loader, epoch)
        elif self.model_config.architecture == "cyclegan":
            epoch_losses = self._train_epoch_cyclegan(train_loader, epoch)

        return epoch_losses

    def _train_epoch_unet(
        self, train_loader: DataLoader, epoch: int
    ) -> Dict[str, float]:
        """Train U-Net for one epoch."""
        total_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, (input_imgs, target_imgs) in enumerate(train_loader):
            input_imgs = input_imgs.to(self.device)
            target_imgs = target_imgs.to(self.device)

            # Forward pass
            self.optimizers["model"].zero_grad()
            generated_imgs = self.models(input_imgs)

            # Compute loss
            loss = self.loss_functions["l1"](generated_imgs, target_imgs)

            # Backward pass
            loss.backward()
            self.optimizers["model"].step()

            total_loss += loss.item()

            # Log batch progress
            if batch_idx % self.training_config.log_frequency == 0:
                self.progress_logger.log_batch(
                    epoch,
                    batch_idx,
                    num_batches,
                    {"l1_loss": loss.item()},
                    total_epochs=self.training_config.epochs,
                )

                if self.wandb_run:
                    self.wandb_run.log(
                        {
                            "batch_loss": loss.item(),
                            "epoch": epoch,
                            "batch": batch_idx,
                        }
                    )

        avg_loss = total_loss / num_batches
        return {"l1_loss": avg_loss}

    def _train_epoch_pix2pix(
        self, train_loader: DataLoader, epoch: int
    ) -> Dict[str, float]:
        """Train Pix2Pix for one epoch."""
        total_g_loss = 0.0
        total_d_loss = 0.0
        total_l1_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, (input_imgs, target_imgs) in enumerate(train_loader):
            input_imgs = input_imgs.to(self.device)
            target_imgs = target_imgs.to(self.device)
            batch_size = input_imgs.size(0)

            # Train Discriminator
            self.optimizers["discriminator"].zero_grad()

            # Real images
            real_output = self.models["discriminator"](input_imgs, target_imgs)
            # Create labels that match the discriminator output size
            output_h, output_w = real_output.shape[2], real_output.shape[3]
            real_labels = torch.ones(batch_size, 1, output_h, output_w).to(
                self.device
            )
            fake_labels = torch.zeros(batch_size, 1, output_h, output_w).to(
                self.device
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
            d_loss.backward()
            self.optimizers["discriminator"].step()

            # Train Generator
            self.optimizers["generator"].zero_grad()

            generated_imgs = self.models["generator"](input_imgs)
            fake_output = self.models["discriminator"](
                input_imgs, generated_imgs
            )

            # Use labels that match the fake_output size
            output_h, output_w = fake_output.shape[2], fake_output.shape[3]
            real_labels_gen = torch.ones(batch_size, 1, output_h, output_w).to(
                self.device
            )

            g_adversarial_loss = self.loss_functions["adversarial"](
                fake_output, real_labels_gen
            )
            g_l1_loss = self.loss_functions["l1"](generated_imgs, target_imgs)

            g_loss = (
                g_adversarial_loss + self.training_config.lambda_l1 * g_l1_loss
            )
            g_loss.backward()
            self.optimizers["generator"].step()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            total_l1_loss += g_l1_loss.item()

            # Log batch progress
            if batch_idx % self.training_config.log_frequency == 0:
                self.progress_logger.log_batch(
                    epoch,
                    batch_idx,
                    num_batches,
                    {
                        "g_loss": g_loss.item(),
                        "d_loss": d_loss.item(),
                        "l1_loss": g_l1_loss.item(),
                    },
                    total_epochs=self.training_config.epochs,
                )

                if self.wandb_run:
                    self.wandb_run.log(
                        {
                            "g_loss": g_loss.item(),
                            "d_loss": d_loss.item(),
                            "l1_loss": g_l1_loss.item(),
                            "epoch": epoch,
                            "batch": batch_idx,
                        }
                    )

        return {
            "g_loss": total_g_loss / num_batches,
            "d_loss": total_d_loss / num_batches,
            "l1_loss": total_l1_loss / num_batches,
        }

    def _train_epoch_cyclegan(
        self, train_loader: DataLoader, epoch: int
    ) -> Dict[str, float]:
        """
        Train CycleGAN for one epoch (stub - requires unpaired data).
        """
        # This would need to be implemented for unpaired training
        # For now, return dummy losses
        return {"g_loss": 0.0, "d_loss": 0.0, "cycle_loss": 0.0}

    def validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate model performance."""
        if isinstance(self.models, dict):
            for model in self.models.values():
                model.eval()
        else:
            self.models.eval()

        val_losses = {}

        with torch.no_grad():
            if self.model_config.architecture == "unet":
                val_losses = self._validate_unet(val_loader)
            elif self.model_config.architecture in [
                "pix2pix",
                "pix2pix_pretrained",
                "pix2pix_controlnet",
                "pix2pix_vit",
            ]:
                val_losses = self._validate_pix2pix(val_loader)
            elif self.model_config.architecture == "cyclegan":
                val_losses = {"val_loss": 0.0}  # Default value

            # Generate sample images
            self._generate_samples(val_loader, epoch)

        return val_losses

    def _validate_unet(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate U-Net model."""
        total_loss = 0.0
        num_batches = len(val_loader)

        for input_imgs, target_imgs in val_loader:
            input_imgs = input_imgs.to(self.device)
            target_imgs = target_imgs.to(self.device)

            generated_imgs = self.models(input_imgs)
            loss = self.loss_functions["l1"](generated_imgs, target_imgs)
            total_loss += loss.item()

        return {"val_l1_loss": total_loss / num_batches}

    def _validate_pix2pix(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate Pix2Pix model."""
        total_l1_loss = 0.0
        num_batches = len(val_loader)

        for input_imgs, target_imgs in val_loader:
            input_imgs = input_imgs.to(self.device)
            target_imgs = target_imgs.to(self.device)

            generated_imgs = self.models["generator"](input_imgs)
            l1_loss = self.loss_functions["l1"](generated_imgs, target_imgs)
            total_l1_loss += l1_loss.item()

        return {"val_l1_loss": total_l1_loss / num_batches}

    def _generate_samples(
        self, val_loader: DataLoader, epoch: int, num_samples: int = 8
    ):
        """Generate sample images for visualization."""
        if isinstance(self.models, dict):
            # For Pix2Pix models
            if "generator" in self.models:
                generator = self.models["generator"]
            # For CycleGAN models
            elif "generator_A2B" in self.models:
                generator = self.models["generator_A2B"]
            else:
                # Fallback - use the first generator we find
                generator_keys = [
                    k for k in self.models.keys() if "generator" in k
                ]
                if generator_keys:
                    generator = self.models[generator_keys[0]]
                else:
                    self.logger.warning(
                        "No generator found for sample generation"
                    )
                    return
        else:
            generator = self.models

        generator.eval()

        # Get a batch of validation data
        input_imgs, target_imgs = next(iter(val_loader))
        input_imgs = input_imgs[:num_samples].to(self.device)
        target_imgs = target_imgs[:num_samples].to(self.device)

        with torch.no_grad():
            generated_imgs = generator(input_imgs)

        # Create comparison grid
        comparison = torch.cat(
            [input_imgs, generated_imgs, target_imgs], dim=0
        )
        grid = make_grid(
            comparison, nrow=num_samples, normalize=True, scale_each=True
        )

        # Save sample
        sample_path = self.samples_dir / f"epoch_{epoch:04d}.png"
        save_image(grid, sample_path)

        # Log to wandb
        if self.wandb_run:
            self.wandb_run.log(
                {"samples": wandb.Image(str(sample_path)), "epoch": epoch}
            )

    def save_checkpoint(
        self, epoch: int, losses: Dict[str, float], is_best: bool = False
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_config": asdict(self.model_config),
            "training_config": asdict(self.training_config),
            "losses": losses,
            "training_history": self.training_history,
            "best_metrics": self.best_metrics,
        }

        # Save model state
        if isinstance(self.models, dict):
            checkpoint["model_state"] = {
                name: model.state_dict() for name, model in self.models.items()
            }
            checkpoint["optimizer_state"] = {
                name: opt.state_dict() for name, opt in self.optimizers.items()
            }
        else:
            checkpoint["model_state"] = self.models.state_dict()
            checkpoint["optimizer_state"] = self.optimizers[
                "model"
            ].state_dict()

        # Save checkpoint
        checkpoint_path = (
            self.checkpoints_dir / f"checkpoint_epoch_{epoch:04d}.pth"
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.checkpoints_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at epoch {epoch}")

        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Main training loop."""
        self.logger.info("Starting training...")
        self.progress_logger.start_training(
            self.training_config.epochs, len(train_loader)
        )

        for epoch in range(self.training_config.epochs):
            self.current_epoch = epoch
            self.progress_logger.start_epoch(
                epoch, self.training_config.epochs
            )

            # Training
            train_losses = self.train_epoch(train_loader, epoch)
            self.training_history["train"].append(train_losses)

            # Validation
            if epoch % self.training_config.eval_frequency == 0:
                val_losses = self.validate(val_loader, epoch)
                self.training_history["val"].append(val_losses)

                # Update learning rate schedulers
                for name, scheduler in self.schedulers.items():
                    if val_losses and "val_" in list(val_losses.keys())[0]:
                        scheduler.step(list(val_losses.values())[0])

                # Check if this is the best model
                is_best = (
                    self._is_best_model(val_losses) if val_losses else False
                )

                self.progress_logger.end_epoch(
                    epoch,
                    self.training_config.epochs,
                    train_losses,
                    val_losses,
                )

                # Log to wandb
                if self.wandb_run:
                    log_dict = {
                        f"train_{k}": v for k, v in train_losses.items()
                    }
                    log_dict.update(
                        {f"val_{k}": v for k, v in val_losses.items()}
                    )
                    log_dict["epoch"] = epoch
                    self.wandb_run.log(log_dict)
            else:
                is_best = False
                self.progress_logger.end_epoch(
                    epoch, self.training_config.epochs, train_losses
                )

                if self.wandb_run:
                    log_dict = {
                        f"train_{k}": v for k, v in train_losses.items()
                    }
                    log_dict["epoch"] = epoch
                    self.wandb_run.log(log_dict)

            # Save checkpoint
            if epoch % self.training_config.save_frequency == 0 or is_best:
                self.save_checkpoint(epoch, {**train_losses}, is_best)

        # Save final model
        self.save_checkpoint(
            self.training_config.epochs - 1,
            self.training_history["train"][-1],
            False,
        )

        self.progress_logger.end_training(self.best_metrics)
        self.logger.info("Training completed!")

    def _is_best_model(self, val_losses: Dict[str, float]) -> bool:
        """Check if current model is the best so far."""
        if not val_losses:
            return False

        # Use the first validation loss as the primary metric
        primary_metric = list(val_losses.keys())[0]
        current_value = val_losses[primary_metric]

        if primary_metric not in self.best_metrics:
            self.best_metrics[primary_metric] = current_value
            return True

        # Lower is better for loss metrics
        if current_value < self.best_metrics[primary_metric]:
            self.best_metrics.update(val_losses)
            return True

        return False
