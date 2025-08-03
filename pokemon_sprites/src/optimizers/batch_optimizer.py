"""
Batch Size Optimizer - Memory and Performance Based

Finds optimal batch sizes by testing memory usage and training speed
with actual models, using empirical measurements over theoretical calculations.
"""

import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent))

from core.models import Pix2PixDiscriminator, Pix2PixGenerator  # noqa: E402


class BatchSizeOptimizer:
    """Find optimal batch size through memory and speed testing"""

    def __init__(self, generator, discriminator, device="cuda"):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device

    def _test_single_batch_size(self, batch_size, input_size, output_size):
        """Test a single batch size for memory and performance."""
        try:
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Record initial memory
            initial_memory = (
                torch.cuda.memory_allocated(self.device) / 1024**2
                if torch.cuda.is_available()
                else 0
            )

            # Create test data
            artwork = torch.randn(batch_size, *input_size).to(self.device)
            sprite = torch.randn(batch_size, *output_size).to(self.device)

            # Setup optimizers and criteria
            gen_optimizer = torch.optim.Adam(
                self.generator.parameters(), lr=1e-4
            )
            disc_optimizer = torch.optim.Adam(
                self.discriminator.parameters(), lr=1e-4
            )
            criterion_GAN = nn.MSELoss()
            criterion_L1 = nn.L1Loss()

            # Time the training step
            start_time = time.time()

            # Train discriminator
            disc_optimizer.zero_grad()
            disc_real = self.discriminator(artwork, sprite)
            real_label = torch.ones_like(disc_real, device=self.device)
            loss_disc_real = criterion_GAN(disc_real, real_label)

            fake_sprite = self.generator(artwork)
            disc_fake = self.discriminator(artwork, fake_sprite.detach())
            fake_label = torch.zeros_like(disc_fake, device=self.device)
            loss_disc_fake = criterion_GAN(disc_fake, fake_label)

            loss_disc = (loss_disc_real + loss_disc_fake) * 0.5
            loss_disc.backward()
            disc_optimizer.step()

            # Train generator
            gen_optimizer.zero_grad()
            disc_fake = self.discriminator(artwork, fake_sprite)
            real_label_gen = torch.ones_like(disc_fake, device=self.device)
            loss_gen_GAN = criterion_GAN(disc_fake, real_label_gen)
            loss_gen_L1 = criterion_L1(fake_sprite, sprite) * 100
            loss_gen = loss_gen_GAN + loss_gen_L1
            loss_gen.backward()
            gen_optimizer.step()

            end_time = time.time()
            training_time = (end_time - start_time) * 1000  # ms

            # Record peak memory
            if torch.cuda.is_available():
                peak_memory = (
                    torch.cuda.max_memory_allocated(self.device) / 1024**2
                )
                memory_used = peak_memory - initial_memory
                torch.cuda.reset_peak_memory_stats(self.device)
            else:
                memory_used = 0

            return True, memory_used, training_time

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return False, 0, 0
            else:
                raise e

    def _test_batch_sizes(self, test_batch_sizes, input_size, output_size):
        """Test a list of batch sizes and return results."""
        results = {
            "tested_batch_sizes": [],
            "memory_usage_mb": [],
            "training_time_ms": [],
            "successful": [],
            "max_stable_batch_size": 4,
        }

        print("Batch Size | Memory (MB) | Time (ms) | Status")
        print("-" * 50)

        for batch_size in test_batch_sizes:
            try:
                success, memory_used, training_time = (
                    self._test_single_batch_size(
                        batch_size, input_size, output_size
                    )
                )

                results["tested_batch_sizes"].append(batch_size)
                results["memory_usage_mb"].append(memory_used)
                results["training_time_ms"].append(training_time)
                results["successful"].append(success)

                if success:
                    print(
                        f"{batch_size:9d} | {memory_used:10.1f} | "
                        f"{training_time:8.1f} | PASS Success"
                    )
                    results["max_stable_batch_size"] = batch_size
                else:
                    status_msg = "FAIL OOM"
                    print(
                        f"{batch_size:9d} | {'N/A':>10} | {'N/A':>8} | "
                        f"{status_msg}"
                    )
                    break

            except Exception:
                status_msg = "FAIL Error"
                print(
                    f"{batch_size:9d} | {'N/A':>10} | {'N/A':>8} | "
                    f"{status_msg}"
                )
                break

        return results

    def _calculate_recommendations(self, results):
        """Calculate optimal and recommended batch sizes from test results."""
        successful_indices = [
            i for i, success in enumerate(results["successful"]) if success
        ]

        if successful_indices:
            successful_batch_sizes = [
                results["tested_batch_sizes"][i] for i in successful_indices
            ]
            successful_times = [
                results["training_time_ms"][i] for i in successful_indices
            ]

            # Most efficient (best time per sample)
            time_per_sample = [
                time / batch_size
                for time, batch_size in zip(
                    successful_times, successful_batch_sizes
                )
            ]
            optimal_idx = successful_indices[
                time_per_sample.index(min(time_per_sample))
            ]
            results["optimal_batch_size"] = results["tested_batch_sizes"][
                optimal_idx
            ]

            # Conservative recommendation
            max_stable = results["max_stable_batch_size"]
            results["recommended_batch_size"] = max(4, int(max_stable * 0.75))
        else:
            results["optimal_batch_size"] = 4
            results["recommended_batch_size"] = 4

        return results

    def find_optimal_batch_size(
        self,
        initial_max_batch_size=64,
        input_size=(3, 256, 256),
        output_size=(4, 256, 256),
    ):
        """
        Find optimal batch size by testing memory usage and training speed.
        """
        print("Testing memory usage and training speed...")

        # Start with batch sizes optimized for 16GB GPU
        test_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        test_batch_sizes = [
            bs for bs in test_batch_sizes if bs <= initial_max_batch_size
        ]

        # Test initial batch sizes
        results = self._test_batch_sizes(
            test_batch_sizes, input_size, output_size
        )

        # Calculate recommendations
        results = self._calculate_recommendations(results)

        # Print results
        print("\nResults:")
        print(f"Max stable: {results['max_stable_batch_size']}")
        print(f"Most efficient: {results['optimal_batch_size']}")
        print(f"Recommended: {results['recommended_batch_size']}")

        return results


def optimize_batch_sizes(
    config_path, test_input_size=(4, 256, 256), test_output_size=(4, 256, 256)
):
    """Optimize batch sizes using memory testing with ARGB data"""

    print("BATCH SIZE OPTIMIZATION")
    print("Testing memory usage with ARGB models")

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

    batch_recommendations = {}

    for model_name, model_config in model_configs.items():
        try:
            print(f"\n--- Testing batch sizes for: {model_name} ---")

            # Extract model parameters from config
            params = model_config.get("parameters", {})
            gen_params = params.get("generator", {})
            disc_params = params.get("discriminator", {})

            # Create models with ARGB config parameters
            generator = Pix2PixGenerator(
                input_channels=gen_params.get("input_channels", 4),  # ARGB
                output_channels=gen_params.get("output_channels", 4),  # ARGB
                ngf=gen_params.get("ngf", 64),
                n_blocks=gen_params.get("n_blocks", 6),
                norm_layer=gen_params.get("norm_layer", "batch"),
                dropout=gen_params.get("dropout", 0.5),
            )
            discriminator = Pix2PixDiscriminator(
                input_channels=disc_params.get(
                    "input_channels", 8
                ),  # ARGB input+target
                ndf=disc_params.get("ndf", 64),
                n_layers=disc_params.get("n_layers", 3),
                norm_layer=disc_params.get("norm_layer", "instance"),
                use_spectral_norm=disc_params.get("use_spectral_norm", False),
            )

            # Test batch sizes with error handling
            optimizer = BatchSizeOptimizer(
                generator, discriminator, str(device)
            )

            try:
                results = optimizer.find_optimal_batch_size(
                    initial_max_batch_size=512,  # Optimized for 16GB GPU
                    input_size=test_input_size,
                    output_size=test_output_size,
                )

                batch_recommendations[model_name] = {
                    "recommended": results["recommended_batch_size"],
                    "optimal": results["optimal_batch_size"],
                    "max_stable": results["max_stable_batch_size"],
                    "testing_method": "memory_testing",
                    "status": "success",
                }

                print(f"PASS Completed for {model_name}")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(
                        f"GPU memory limit reached for {model_name}, using "
                        f"conservative values"
                    )
                    batch_recommendations[model_name] = {
                        "recommended": 4,
                        "optimal": 4,
                        "max_stable": 4,
                        "testing_method": "memory_limited",
                        "status": "memory_limited",
                    }
                else:
                    raise e

        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            batch_recommendations[model_name] = {
                "recommended": 4,
                "optimal": 4,
                "max_stable": 4,
                "testing_method": "fallback",
                "status": "error",
                "error": str(e),
            }

    print(
        f"\nBatch optimization completed for "
        f"{len(batch_recommendations)} models"
    )
    return batch_recommendations
