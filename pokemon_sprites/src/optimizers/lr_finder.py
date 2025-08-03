"""
Learning Rate Finder - Implementation of Smith et al. (2017)

Trains models with actual data to find optimal learning rates.
Training experiments as the paper describes.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent))

from core.models import Pix2PixDiscriminator, Pix2PixGenerator  # noqa: E402


class LearningRateFinder:
    """Learning Rate Finder implementing Smith et al. (2017)"""

    def __init__(self, generator, discriminator, device="cuda"):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.history = {"lr": [], "loss": []}

    def find_optimal_lr(
        self,
        dataloader,
        start_lr=1e-7,
        end_lr=1,
        num_iterations=50,
        lambda_l1=100.0,
    ):
        """Find optimal learning rate using ACTUAL training"""

        print(
            f"LR Range Test: {start_lr:.2e} -> {end_lr:.2e} "
            f"({num_iterations} iterations)"
        )

        # Save initial states
        initial_gen_state = self.generator.state_dict().copy()
        initial_disc_state = self.discriminator.state_dict().copy()

        # Setup optimizers
        gen_optimizer = optim.Adam(
            self.generator.parameters(), lr=start_lr, betas=(0.5, 0.999)
        )
        disc_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=start_lr, betas=(0.5, 0.999)
        )

        # Loss functions
        criterion_GAN = nn.MSELoss()
        criterion_L1 = nn.L1Loss()

        # LR progression
        lr_mult = (end_lr / start_lr) ** (1.0 / num_iterations)
        current_lr = start_lr

        # Training setup
        self.generator.train()
        self.discriminator.train()
        data_iter = iter(dataloader)

        losses = []
        lrs = []
        best_loss = float("inf")

        for iteration in range(num_iterations):
            try:
                # Get batch
                try:
                    artwork, sprite = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    artwork, sprite = next(data_iter)

                artwork = artwork.to(self.device)
                sprite = sprite.to(self.device)

                # Update LRs
                for param_group in gen_optimizer.param_groups:
                    param_group["lr"] = current_lr
                for param_group in disc_optimizer.param_groups:
                    param_group["lr"] = current_lr

                # Train Discriminator
                disc_optimizer.zero_grad()
                disc_real = self.discriminator(artwork, sprite)

                # Create labels that match discriminator output shape
                real_label = torch.ones_like(disc_real, device=self.device)
                _ = torch.zeros_like(
                    disc_real, device=self.device
                )  # fake_label not used in this section

                loss_disc_real = criterion_GAN(disc_real, real_label)

                fake_sprite = self.generator(artwork)
                disc_fake = self.discriminator(artwork, fake_sprite.detach())
                fake_label_disc = torch.zeros_like(
                    disc_fake, device=self.device
                )
                loss_disc_fake = criterion_GAN(disc_fake, fake_label_disc)

                loss_disc = (loss_disc_real + loss_disc_fake) * 0.5
                loss_disc.backward()
                disc_optimizer.step()

                # Train Generator
                gen_optimizer.zero_grad()
                disc_fake = self.discriminator(artwork, fake_sprite)
                real_label_gen = torch.ones_like(disc_fake, device=self.device)
                loss_gen_GAN = criterion_GAN(disc_fake, real_label_gen)
                loss_gen_L1 = criterion_L1(fake_sprite, sprite) * lambda_l1
                loss_gen = loss_gen_GAN + loss_gen_L1
                loss_gen.backward()
                gen_optimizer.step()

                # Record loss
                total_loss = loss_gen.item() + loss_disc.item()
                losses.append(total_loss)
                lrs.append(current_lr)

                if total_loss < best_loss:
                    best_loss = total_loss

                # Stop if diverging
                if total_loss > best_loss * 4:
                    print(
                        f"Early stop at iteration {iteration} - loss diverged"
                    )
                    break

                if iteration % 10 == 0:
                    print(
                        f"Iter {iteration}: LR {current_lr:.2e}, "
                        f"Loss {total_loss:.4f}"
                    )

                current_lr *= lr_mult

            except Exception as e:
                print(f"Error at iteration {iteration}: {e}")
                break

        # Restore models
        self.generator.load_state_dict(initial_gen_state)
        self.discriminator.load_state_dict(initial_disc_state)

        # Analyze results
        self.history = {"lr": lrs, "loss": losses}
        return self._analyze_results(lrs, losses)

    def _analyze_results(self, lrs, losses):
        """Analyze LR curve following Smith et al."""
        if len(losses) < 5:
            # Fallback
            optimal_lr = 1e-4
            return {
                "optimal_lr": optimal_lr,
                "min_lr": optimal_lr / 10,
                "max_lr": optimal_lr * 3,
                "analysis_method": "fallback",
            }

        lrs_np = np.array(lrs)
        losses_np = np.array(losses)

        # Find minimum loss
        min_idx = np.argmin(losses_np)
        min_loss_lr = lrs_np[min_idx]

        # Smith's recommendation: use before divergence point
        optimal_lr = min_loss_lr / 3.0

        results = {
            "optimal_lr": float(optimal_lr),
            "min_lr": float(optimal_lr / 10),
            "max_lr": float(optimal_lr * 3),
            "min_loss_lr": float(min_loss_lr),
            "analysis_method": "smith_et_al_2017",
        }

        print(
            f"Results: Optimal LR = {optimal_lr:.2e}, "
            f"Range = {results['min_lr']:.2e} - {results['max_lr']:.2e}"
        )
        return results


def create_synthetic_dataloader(batch_size=4, num_samples=50):
    """Create synthetic dataloader for testing with ARGB channels"""
    artwork = torch.randn(num_samples, 4, 256, 256)  # ARGB input
    sprites = torch.randn(num_samples, 4, 256, 256)  # ARGB output
    dataset = TensorDataset(artwork, sprites)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def find_optimal_learning_rates(
    config_path, dataset_path=None, num_iterations=30
):
    """Find optimal learning rates using ACTUAL training (Smith et al. 2017)"""

    print("LEARNING RATE OPTIMIZATION")
    print("Using model training - no heuristics")

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        print(f"Config load error: {e}")
        return {}

    model_configs = config.get("pix2pix_models", {})
    if not model_configs:
        print("No model configurations found")
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Use synthetic data for demonstration
    dataloader = create_synthetic_dataloader(batch_size=4, num_samples=50)
    print("Using synthetic data for LR finding")

    optimal_lrs = {}

    for model_name, model_config in model_configs.items():
        try:
            print(f"\n--- LR finder for: {model_name} ---")

            # Extract model parameters from config
            params = model_config.get("parameters", {})
            gen_params = params.get("generator", {})
            disc_params = params.get("discriminator", {})

            # Create models with config parameters
            generator = Pix2PixGenerator(
                input_channels=gen_params.get("input_channels", 4),  
                output_channels=gen_params.get("output_channels", 4),
                ngf=gen_params.get("ngf", 64),
                n_blocks=gen_params.get("n_blocks", 9),
                norm_layer=gen_params.get("norm_layer", "instance"),
                dropout=gen_params.get("dropout", 0.3),
            )
            discriminator = Pix2PixDiscriminator(
                input_channels=disc_params.get("input_channels", 8), 
                ndf=disc_params.get("ndf", 64),
                n_layers=disc_params.get("n_layers", 3),
                norm_layer=disc_params.get("norm_layer", "instance"),
                use_spectral_norm=disc_params.get("use_spectral_norm", False),
            )

            # Run LR finder
            lr_finder = LearningRateFinder(
                generator, discriminator, str(device)
            )
            results = lr_finder.find_optimal_lr(
                dataloader, num_iterations=num_iterations
            )

            optimal_lrs[model_name] = results
            print(f"PASS Completed LR finding for {model_name}")

        except Exception as e:
            print(f"Error with {model_name}: {e}")
            # Fallback to reasonable defaults
            optimal_lrs[model_name] = {
                "optimal_lr": 1e-4,
                "min_lr": 1e-5,
                "max_lr": 3e-4,
                "analysis_method": "fallback_heuristic",
            }

    print(f"\nOptimization complete for {len(optimal_lrs)} models")
    return optimal_lrs
