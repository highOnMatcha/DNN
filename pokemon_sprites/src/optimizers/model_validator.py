"""
Model Validator - Model Instantiation Testing

Validates model configurations by creating and testing the models,
checking configuration syntax and runtime compatibility.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent))

from core.models import Pix2PixDiscriminator, Pix2PixGenerator  # noqa: E402


def _validate_single_model(model_name, model_config, device):
    """Validate a single model configuration."""
    print(f"Validating {model_name}...")

    results = {
        "model_name": model_name,
        "config_valid": True,
        "generator_created": False,
        "discriminator_created": False,
        "forward_pass_works": False,
        "backward_pass_works": False,
        "errors": [],
        "parameter_count": {"generator": 0, "discriminator": 0},
    }

    try:
        # Extract parameters
        params = model_config.get("parameters", {})
        gen_params = params.get("generator", {})
        disc_params = params.get("discriminator", {})

        # Test generator creation
        generator = _create_generator(gen_params, device, results)
        if not results["generator_created"]:
            return results

        # Test discriminator creation
        discriminator = _create_discriminator(disc_params, device, results)
        if not results["discriminator_created"]:
            return results

        # Test forward pass
        if not _test_forward_pass(
            generator, discriminator, gen_params, device, results
        ):
            return results

        # Test backward pass
        _test_backward_pass(
            generator, discriminator, gen_params, device, results
        )

    except Exception as e:
        results["errors"].append(str(e))
        print(f"  FAIL Error: {e}")

    return results


def _create_generator(gen_params, device, results):
    """Create and validate generator model."""
    try:
        generator = Pix2PixGenerator(
            input_channels=gen_params.get("input_channels", 4),  # Fixed: ARGB
            output_channels=gen_params.get(
                "output_channels", 4
            ),  # Fixed: ARGB
            ngf=gen_params.get("ngf", 64),
            n_blocks=gen_params.get("n_blocks", 6),
            norm_layer=gen_params.get("norm_layer", "batch"),
            dropout=gen_params.get("dropout", 0.5),
        )

        # Move to device after creation
        generator = generator.to(device)

        # Test generator forward pass
        gen_params_count = sum(
            p.numel() for p in generator.parameters() if p.requires_grad
        )
        print(f"    Generator parameters: {gen_params_count:,}")

        # Test forward pass - ensure input is on same device as model
        test_input = torch.randn(
            2, 4, 256, 256, device=device
        )  # Create directly on device
        output = generator(test_input)
        print(
            f"    Forward pass successful: "
            f"{test_input.shape} -> {output.shape}"
        )

        results["generator_created"] = True
        results["parameter_count"]["generator"] = gen_params_count
        print(f"  PASS Generator created: {gen_params_count:,} parameters")
        return generator

    except Exception as e:
        print(f"  FAIL Generator error: {e}")
        results["generator_created"] = False
        results["errors"].append(f"Generator: {e}")
        return None


def _create_discriminator(disc_params, device, results):
    """Create and validate discriminator model."""
    try:
        discriminator = Pix2PixDiscriminator(
            input_channels=disc_params.get(
                "input_channels", 8
            ),  # Fixed: ARGB input+target
            ndf=disc_params.get("ndf", 64),
            n_layers=disc_params.get("n_layers", 3),
            norm_layer=disc_params.get("norm_layer", "instance"),
            use_spectral_norm=disc_params.get("use_spectral_norm", False),
        ).to(
            device
        )  # Move to device

        # Test discriminator forward pass
        disc_params_count = sum(
            p.numel() for p in discriminator.parameters() if p.requires_grad
        )
        print(f"    Discriminator parameters: {disc_params_count:,}")

        # Test forward pass with dummy input+target pair - create tensors
        # directly on device
        test_input = torch.randn(2, 4, 256, 256, device=device)  # ARGB input
        test_target = torch.randn(2, 4, 256, 256, device=device)  # ARGB target

        output = discriminator(test_input, test_target)
        print(
            f"    Forward pass successful: {test_input.shape} + "
            f"{test_target.shape} -> {output.shape}"
        )

        results["discriminator_created"] = True
        results["parameter_count"]["discriminator"] = disc_params_count
        print(
            f"  PASS Discriminator created: {disc_params_count:,} parameters"
        )
        return discriminator

    except Exception as e:
        print(f"  FAIL Discriminator error: {e}")
        results["discriminator_created"] = False
        results["errors"].append(f"Discriminator: {e}")
        return None


def _test_forward_pass(generator, discriminator, gen_params, device, results):
    """Test forward pass of both models."""
    try:
        generator.eval()
        discriminator.eval()

        test_artwork = torch.randn(
            2,
            gen_params.get("input_channels", 4),
            256,
            256,  # Fixed: ARGB default
        ).to(device)
        test_sprite = torch.randn(
            2, gen_params.get("output_channels", 4), 256, 256
        ).to(device)

        with torch.no_grad():
            fake_sprite = generator(test_artwork)
            disc_real = discriminator(test_artwork, test_sprite)
            _ = discriminator(
                test_artwork, fake_sprite
            )  # Test discriminator with fake input

        results["forward_pass_works"] = True
        print("  PASS Forward pass successful")
        print(f"    Generator: {test_artwork.shape} -> {fake_sprite.shape}")
        print(
            f"    Discriminator: {test_artwork.shape} + "
            f"{test_sprite.shape} -> {disc_real.shape}"
        )
        return True

    except Exception as e:
        results["errors"].append(f"Forward pass failed: {str(e)}")
        print(f"  FAIL Forward pass failed: {e}")
        return False


def _test_backward_pass(generator, discriminator, gen_params, device, results):
    """Test backward pass and training step."""
    try:
        generator.train()
        discriminator.train()

        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
        criterion_GAN = nn.MSELoss()
        criterion_L1 = nn.L1Loss()

        test_artwork = torch.randn(
            2, gen_params.get("input_channels", 3), 256, 256
        ).to(device)
        test_sprite = torch.randn(
            2, gen_params.get("output_channels", 4), 256, 256
        ).to(device)

        # Train discriminator
        disc_optimizer.zero_grad()
        disc_real = discriminator(test_artwork, test_sprite)
        real_label = torch.ones_like(disc_real).to(device)
        loss_disc_real = criterion_GAN(disc_real, real_label)

        fake_sprite = generator(test_artwork)
        disc_fake = discriminator(test_artwork, fake_sprite.detach())
        fake_label = torch.zeros_like(disc_fake).to(device)
        loss_disc_fake = criterion_GAN(disc_fake, fake_label)

        loss_disc = (loss_disc_real + loss_disc_fake) * 0.5
        loss_disc.backward()
        disc_optimizer.step()

        # Train generator
        gen_optimizer.zero_grad()
        disc_fake = discriminator(test_artwork, fake_sprite)
        real_label_gen = torch.ones_like(disc_fake).to(device)
        loss_gen_GAN = criterion_GAN(disc_fake, real_label_gen)
        loss_gen_L1 = criterion_L1(fake_sprite, test_sprite) * 100
        loss_gen = loss_gen_GAN + loss_gen_L1
        loss_gen.backward()
        gen_optimizer.step()

        results["backward_pass_works"] = True
        print("  PASS Backward pass successful")
        print(f"    Generator loss: {loss_gen.item():.4f}")
        print(f"    Discriminator loss: {loss_disc.item():.4f}")

    except Exception as e:
        results["errors"].append(f"Backward pass failed: {str(e)}")
        print(f"  FAIL Backward pass failed: {e}")


def validate_all_configurations(config_path):
    """Validate all model configurations by creating and testing models."""
    print("MODEL VALIDATION")
    print("Creating and testing models")

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
    print(f"Validating {len(model_configs)} model configurations...\n")

    validation_results = {}

    for model_name, model_config in model_configs.items():
        results = _validate_single_model(model_name, model_config, device)
        validation_results[model_name] = results

        # Summary for this model
        if results["forward_pass_works"] and results["backward_pass_works"]:
            print(f"  PASS {model_name}: VALID")
        else:
            print(f"  FAIL {model_name}: INVALID")
        print()

    _print_validation_summary(validation_results)
    return validation_results


def _print_validation_summary(validation_results):
    """Print validation summary report."""
    print("VALIDATION SUMMARY")
    print("=" * 50)

    valid_models = [
        name
        for name, results in validation_results.items()
        if results.get("forward_pass_works", False)
        and results.get("backward_pass_works", False)
    ]
    invalid_models = [
        name
        for name, results in validation_results.items()
        if not (
            results.get("forward_pass_works", False)
            and results.get("backward_pass_works", False)
        )
    ]

    print(f"Valid models: {len(valid_models)}")
    for model in valid_models:
        print(f"  PASS {model}")

    if invalid_models:
        print(f"Invalid models: {len(invalid_models)}")
        for model in invalid_models:
            print(f"  FAIL {model}")

    total_params = sum(
        results.get("parameter_count", {}).get("generator", 0)
        + results.get("parameter_count", {}).get("discriminator", 0)
        for results in validation_results.values()
        if results.get("forward_pass_works", False)
    )

    print(f"\nTotal parameters across all valid models: {total_params:,}")


def optimize_model_config(config_path):
    """Main function to validate all model configurations."""
    return validate_all_configurations(config_path)
