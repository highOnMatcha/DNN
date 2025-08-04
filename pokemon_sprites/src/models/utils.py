"""
Utility functions for model analysis and parameter counting.

This module provides functions for analyzing model architectures,
counting parameters, and generating recommendations for Pokemon
sprite generation tasks.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add src to path - must be before other local imports
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from core.logging_config import get_logger
except ImportError:
    import logging

    def get_logger(name):
        return logging.getLogger(name)


logger = get_logger(__name__)


def count_parameters(model) -> int:
    """Count the number of trainable parameters in a model."""
    if isinstance(model, dict):
        return sum(count_parameters(m) for m in model.values())
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model) -> int:
    """Count the total number of parameters in a model (including frozen)."""
    if isinstance(model, dict):
        return sum(count_total_parameters(m) for m in model.values())
    return sum(p.numel() for p in model.parameters())


def analyze_model_architectures(config_path=None):
    """
    Analyze available model architectures for Pokemon sprite generation.

    Evaluates model configurations for artwork-to-sprite translation,
    focusing on ARGB support, parameter efficiency, and pixel art optimization.

    Args:
        config_path: Path to model configuration file

    Returns:
        dict: Analysis results with suitability scores
    """
    config = _load_model_config(config_path)
    if not config:
        return {}

    model_configs = config.get("pix2pix_models", {})
    if not model_configs:
        return {}

    analysis_results: Dict[str, Any] = {
        "models": {},
        "recommendations": [],
        "summary": {},
    }
    architecture_analysis = {}

    for model_name, model_config in model_configs.items():
        model_analysis = _analyze_single_model(model_name, model_config)
        analysis_results["models"][model_name] = model_analysis
        architecture_analysis[model_name] = model_analysis["suitability_score"]

    # Generate recommendations and summary
    analysis_results["recommendations"] = _generate_recommendations(
        architecture_analysis, analysis_results
    )
    analysis_results["summary"] = _generate_summary(
        model_configs, analysis_results
    )

    return analysis_results


def _load_model_config(config_path=None):
    """Load and validate model configuration file."""
    if config_path is None:
        config_file_path = (
            Path(__file__).parent.parent / "config" / "model_configs.json"
        )
    else:
        config_file_path = Path(config_path)

    if not config_file_path.exists():
        logger.error(f"Model configuration file not found: {config_file_path}")
        return {}

    try:
        with open(config_file_path, "r") as f:
            config = json.load(f)
        logger.info(
            "Analyzing curated model architectures for sprite generation"
        )
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return {}


def _analyze_single_model(model_name: str, model_config: Dict[str, Any]):
    """Analyze a single model configuration."""
    logger.info(f"Analyzing model: {model_name}")

    params = model_config.get("parameters", {})
    generator = params.get("generator", {})
    discriminator = params.get("discriminator", {})

    # Calculate parameter estimates
    total_params = _calculate_model_parameters(
        model_name, generator, discriminator
    )

    # Assess model suitability
    suitability_score, strengths, considerations, role = (
        _assess_model_suitability(model_name, params, generator, discriminator)
    )

    # Create detailed analysis structure
    model_analysis = _create_model_analysis_dict(
        model_config,
        role,
        total_params,
        suitability_score,
        strengths,
        considerations,
        generator,
        discriminator,
        params,
    )

    logger.info(f"Model {model_name}: Score {suitability_score}/12")
    return model_analysis


def _calculate_model_parameters(
    model_name: str, generator: Dict[str, Any], discriminator: Dict[str, Any]
) -> float:
    """Calculate estimated parameters for model architecture."""
    ngf = generator.get("ngf", 64)
    ndf = discriminator.get("ndf", 64)
    n_blocks = generator.get("n_blocks", 9)
    d_layers = discriminator.get("n_layers", 3)

    if "transformer" in model_name:
        transformer_layers = generator.get("transformer_layers", 4)
        attention_heads = generator.get("attention_heads", 8)
        transformer_params = (
            transformer_layers * attention_heads * ngf * ngf * 4
        )
        base_params = ngf * ngf * (n_blocks * 2 + 8) + ndf * ndf * (
            d_layers + 2
        )
        total_params = (base_params + transformer_params) / 1000000
    else:
        gen_params = ngf * ngf * (n_blocks * 2 + 10)
        disc_params = ndf * ndf * (d_layers + 2)
        total_params = (gen_params + disc_params) / 1000000

    return total_params


def _assess_model_suitability(
    model_name: str,
    params: Dict[str, Any],
    generator: Dict[str, Any],
    discriminator: Dict[str, Any],
) -> Tuple[int, List[str], List[str], str]:
    """Assess model suitability for ARGB sprite generation."""
    suitability_score = 0
    strengths = []
    considerations = []

    # ARGB support check (critical for transparency)
    if (
        generator.get("input_channels", 3) == 4
        and generator.get("output_channels", 3) == 4
    ):
        strengths.append("Native ARGB transparency support")
        suitability_score += 2
    else:
        considerations.append("Missing ARGB channel support")

    # Image size and loss function assessments
    if params.get("image_size", 0) == 256:
        strengths.append("Direct 256px output compatibility")
        suitability_score += 1
    else:
        considerations.append("Non-standard image size")

    if params.get("lambda_l1", 0) >= 150:
        strengths.append("Strong pixel-level accuracy emphasis")
        suitability_score += 1

    if params.get("lambda_pixel_art", 0) > 0:
        strengths.append("Specialized pixel art loss function")
        suitability_score += 1

    # Model-specific analysis and role assignment
    role = _determine_model_role(
        model_name,
        generator,
        discriminator,
        strengths,
        considerations,
        suitability_score,
    )

    # Normalization preference for style transfer
    if generator.get("norm_layer", "batch") == "instance":
        strengths.append("Instance normalization (better for style transfer)")
        suitability_score += 1

    return suitability_score, strengths, considerations, role


def _determine_model_role(
    model_name: str,
    generator: Dict[str, Any],
    discriminator: Dict[str, Any],
    strengths: List[str],
    considerations: List[str],
    suitability_score: int,
) -> str:
    """Determine the role and characteristics of a model."""
    if "lightweight" in model_name:
        strengths.extend(
            [
                "Fast training and inference",
                "Low memory requirements",
                "Reduced overfitting risk",
            ]
        )
        considerations.append("May lack capacity for complex mappings")
        return "BASELINE - Quick validation and debugging"

    elif "sprite-optimized" in model_name:
        if generator.get("use_attention", False):
            strengths.append("Attention mechanism for detail preservation")
        if discriminator.get("use_spectral_norm", False):
            strengths.append("Spectral normalization for training stability")
        strengths.extend(
            [
                "Optimized loss weights for pixel art",
                "Balanced complexity for dataset size",
            ]
        )
        return "PRIMARY - Main production model"

    elif "transformer" in model_name:
        strengths.extend(
            [
                "Long-range dependency modeling",
                "Advanced attention mechanisms",
                "State-of-the-art architecture",
            ]
        )
        considerations.extend(
            [
                "Higher computational requirements",
                "May need careful regularization",
            ]
        )
        return "ADVANCED - Experimental state-of-the-art"

    return "GENERAL - Standard configuration"


def _create_model_analysis_dict(
    model_config: Dict[str, Any],
    role: str,
    total_params: float,
    suitability_score: int,
    strengths: List[str],
    considerations: List[str],
    generator: Dict[str, Any],
    discriminator: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Create the detailed model analysis dictionary."""
    input_ch = generator.get("input_channels", 3)
    output_ch = generator.get("output_channels", 3)

    return {
        "description": model_config.get("description", "No description"),
        "architecture": model_config.get("architecture", "pix2pix"),
        "role": role,
        "estimated_parameters_m": round(total_params, 1),
        "suitability_score": suitability_score,
        "max_score": 12,
        "strengths": strengths,
        "considerations": considerations,
        "generator_config": {
            "channels": f"{input_ch} -> {output_ch}",
            "base_features": generator.get("ngf", 64),
            "blocks": generator.get("n_blocks", 9),
            "normalization": generator.get("norm_layer", "instance"),
            "dropout": generator.get("dropout", 0.3),
        },
        "discriminator_config": {
            "input_channels": discriminator.get("input_channels", 6),
            "base_features": discriminator.get("ndf", 64),
            "layers": discriminator.get("n_layers", 3),
            "spectral_norm": discriminator.get("use_spectral_norm", False),
        },
        "training_config": {
            "image_size": params.get("image_size", 256),
            "l1_weight": params.get("lambda_l1", 100),
            "perceptual_weight": params.get("lambda_perceptual", 0),
            "pixel_art_weight": params.get("lambda_pixel_art", 0),
        },
    }


def _generate_recommendations(
    architecture_analysis, analysis_results
) -> List[Dict[str, Any]]:
    """Generate model recommendations based on analysis."""
    sorted_models = sorted(
        architecture_analysis.items(), key=lambda x: x[1], reverse=True
    )

    recommendations = []
    for i, (model_name, score) in enumerate(sorted_models, 1):
        model_info = analysis_results["models"][model_name]
        use_case = (
            "Start here for quick results"
            if i == 1
            else (
                "Optimize after baseline"
                if i == 2
                else "Experiment if resources allow"
            )
        )

        recommendations.append(
            {
                "rank": i,
                "model_name": model_name,
                "role": model_info["role"],
                "score": f"{score}/12",
                "use_case": use_case,
                "parameters_m": model_info["estimated_parameters_m"],
            }
        )

    return recommendations


def _generate_summary(model_configs, analysis_results):
    """Generate analysis summary statistics."""
    summary = {
        "total_models": len(model_configs),
        "argb_compatible": sum(
            1
            for m in analysis_results["models"].values()
            if "4 -> 4" in m["generator_config"]["channels"]
        ),
        "pixel_art_optimized": sum(
            1
            for m in analysis_results["models"].values()
            if m["training_config"]["pixel_art_weight"] > 0
        ),
        "recommended_sequence": [
            r["model_name"] for r in analysis_results["recommendations"][:3]
        ],
    }

    logger.info("Model architecture analysis completed")
    sequence_str = " -> ".join(summary["recommended_sequence"])
    logger.info(f"Recommended training sequence: {sequence_str}")

    return summary
