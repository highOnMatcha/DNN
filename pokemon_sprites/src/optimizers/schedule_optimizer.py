"""
Training Schedule Optimizer for curriculum learning and learning rate
scheduling.
"""

import math


class TrainingScheduleOptimizer:
    """
    Create optimized training schedules with curriculum learning
    """

    def __init__(self, total_samples=898, validation_split=0.15):
        self.total_samples = total_samples
        self.train_samples = int(total_samples * (1 - validation_split))
        self.val_samples = total_samples - self.train_samples

    def calculate_training_time(
        self, epochs, batch_size, samples_per_epoch=None
    ):
        """Estimate training time based on batch size and epochs"""

        if samples_per_epoch is None:
            samples_per_epoch = self.train_samples

        batches_per_epoch = math.ceil(samples_per_epoch / batch_size)
        total_batches = batches_per_epoch * epochs

        # Estimate processing time per batch (in seconds)
        # These are rough estimates - actual times depend on hardware
        time_per_batch = {
            "lightweight": 0.5,  # Simple models
            "medium": 1.0,  # Standard models
            "heavy": 2.0,  # Complex models with transformers
        }

        return {
            "batches_per_epoch": batches_per_epoch,
            "total_batches": total_batches,
            "estimated_time_per_batch": time_per_batch,
            "samples_per_epoch": samples_per_epoch,
        }

    def create_curriculum_schedule(self, model_complexity="medium"):
        """
        Create curriculum learning schedule with progressive resolution
        """

        print(
            f"Creating Curriculum Learning Schedule "
            f"({model_complexity} complexity)"
        )
        print("=" * 60)

        # Progressive resolution stages
        stages = [
            {
                "name": "Foundation Stage",
                "input_resolution": 128,
                "output_resolution": 256,
                "epochs": 30,
                "focus": "Basic shape and structure learning",
                "augmentation": "minimal",
            },
            {
                "name": "Refinement Stage",
                "input_resolution": 192,
                "output_resolution": 256,
                "epochs": 25,
                "focus": "Detail enhancement and color learning",
                "augmentation": "moderate",
            },
            {
                "name": "Polish Stage",
                "input_resolution": 256,
                "output_resolution": 256,
                "epochs": 20,
                "focus": "Fine details and precise output",
                "augmentation": "strong",
            },
        ]

        # Adjust batch sizes based on resolution and model complexity
        complexity_factors = {
            "lightweight": {"base_batch": 16, "lr_factor": 1.0},
            "medium": {"base_batch": 8, "lr_factor": 0.7},
            "heavy": {"base_batch": 4, "lr_factor": 0.5},
        }

        factor = complexity_factors.get(
            model_complexity, complexity_factors["medium"]
        )

        schedule = []
        total_epochs = 0

        for i, stage in enumerate(stages):
            # Calculate resolution-adjusted batch size
            resolution_factor = (256 / stage["input_resolution"]) ** 1.5
            batch_size = max(1, int(factor["base_batch"] * resolution_factor))

            # Learning rate schedule for this stage
            base_lr = 2e-4 * factor["lr_factor"]
            if i == 0:  # Foundation stage - higher LR
                stage_lr = base_lr
            elif i == 1:  # Refinement stage - medium LR
                stage_lr = base_lr * 0.7
            else:  # Polish stage - lower LR
                stage_lr = base_lr * 0.4

            # Training time estimation
            timing = self.calculate_training_time(
                epochs=stage["epochs"], batch_size=batch_size
            )

            stage_info = {
                **stage,
                "batch_size": batch_size,
                "learning_rate": stage_lr,
                "lr_schedule": "cosine_annealing",
                "warmup_epochs": 3 if i == 0 else 2,
                "timing": timing,
                "stage_number": i + 1,
                "cumulative_epochs": total_epochs + stage["epochs"],
            }

            schedule.append(stage_info)
            total_epochs += stage["epochs"]

            # Print stage summary
            print(f"STAGE {i+1}: {stage['name']}")
            print(
                f"  Resolution: {stage['input_resolution']}px → "
                f"{stage['output_resolution']}px"
            )
            print(f"  Epochs: {stage['epochs']}")
            print(f"  Batch size: {batch_size}")
            print(f"  Learning rate: {stage_lr:.2e}")
            print(f"  Augmentation: {stage['augmentation']}")
            print(f"  Focus: {stage['focus']}")
            print(f"  Batches per epoch: {timing['batches_per_epoch']}")
            print()

        print(
            f"TOTAL TRAINING: {total_epochs} epochs across "
            f"{len(stages)} stages"
        )

        return schedule

    def create_learning_rate_schedule(
        self, schedule_type="cosine_annealing", total_epochs=75, base_lr=2e-4
    ):
        """Create detailed learning rate schedule"""

        print(f"Learning Rate Schedule: {schedule_type}")
        print("-" * 40)

        if schedule_type == "cosine_annealing":
            # Cosine annealing with restarts
            schedule = []

            for epoch in range(total_epochs):
                # Cosine annealing formula
                lr = (
                    base_lr
                    * 0.5
                    * (1 + math.cos(math.pi * epoch / total_epochs))
                )
                schedule.append(lr)

            # Add warmup for first 5 epochs
            warmup_epochs = 5
            for i in range(min(warmup_epochs, len(schedule))):
                warmup_lr = base_lr * (i + 1) / warmup_epochs
                schedule[i] = warmup_lr

            print(f"  Base LR: {base_lr:.2e}")
            print(f"  Warmup epochs: {warmup_epochs}")
            print(f"  Min LR: {min(schedule):.2e}")
            print(f"  Max LR: {max(schedule):.2e}")

        elif schedule_type == "step_decay":
            # Step decay schedule
            schedule = []
            decay_epochs = [total_epochs // 3, 2 * total_epochs // 3]
            decay_factor = 0.3

            current_lr = base_lr
            for epoch in range(total_epochs):
                if epoch in decay_epochs:
                    current_lr *= decay_factor
                schedule.append(current_lr)

            print(f"  Base LR: {base_lr:.2e}")
            print(f"  Decay epochs: {decay_epochs}")
            print(f"  Decay factor: {decay_factor}")
            print(f"  Final LR: {schedule[-1]:.2e}")

        return schedule


def create_optimal_training_plan(config_path):
    """
    Create complete training plan integrating all optimization insights
    and update the configuration file with optimized schedules
    """
    import json
    from pathlib import Path

    print("OPTIMAL TRAINING PLAN FOR POKEMON SPRITE GENERATION")
    print("=" * 60)

    # Load configuration
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        config = json.load(f)

    models_to_train = list(config.get("pix2pix_models", {}).keys())

    training_plans = {}
    optimizer = TrainingScheduleOptimizer()

    # Create optimized training schedules for each model
    for model_name in models_to_train:
        print(f"\nPLAN FOR: {model_name.upper().replace('-', ' ')}")
        print("=" * 50)

        # Determine model complexity
        if "lightweight" in model_name:
            complexity = "lightweight"
        elif "transformer" in model_name:
            complexity = "heavy"
        else:
            complexity = "medium"

        # Create curriculum schedule
        curriculum = optimizer.create_curriculum_schedule(complexity)

        # Extract learning rates from our previous analysis
        base_lrs = {
            "lightweight-baseline": 2e-4,
            "sprite-optimized": 1.4e-4,
            "transformer-enhanced": 1e-4,
        }

        base_lr = base_lrs.get(model_name, 1.5e-4)

        # Training plan summary
        plan = {
            "model_name": model_name,
            "complexity": complexity,
            "curriculum_stages": curriculum,
            "base_learning_rate": base_lr,
            "total_epochs": sum(stage["epochs"] for stage in curriculum),
            "training_strategy": "curriculum_learning",
            "recommended_gpu_memory": (
                "8GB+" if complexity == "heavy" else "6GB+"
            ),
        }

        training_plans[model_name] = plan

        # Print execution order
        print("EXECUTION ORDER:")
        for i, stage in enumerate(curriculum, 1):
            print(
                f"  {i}. {stage['name']}: {stage['epochs']} epochs @ "
                f"{stage['input_resolution']}px"
            )

        print("\nRESOURCE REQUIREMENTS:")
        print(
            f"  Estimated total training time: {plan['total_epochs']} epochs"
        )
        print(f"  GPU memory needed: {plan['recommended_gpu_memory']}")
        print(f"  Base learning rate: {base_lr:.2e}")

    # Add optimized training schedules to config
    if "optimized_training_schedules" not in config:
        config["optimized_training_schedules"] = {}

    # Convert training plans to config format
    for model_name, plan in training_plans.items():
        optimized_schedule = {
            "description": (
                f"Optimized curriculum learning schedule for {model_name}"
            ),
            "strategy": "curriculum_learning",
            "base_learning_rate": plan["base_learning_rate"],
            "total_epochs": plan["total_epochs"],
            "gpu_memory_requirement": plan["recommended_gpu_memory"],
            "stages": [],
        }

        for i, stage in enumerate(plan["curriculum_stages"]):
            stage_config = {
                "stage_name": stage["name"],
                "stage_number": i + 1,
                "epochs": stage["epochs"],
                "input_resolution": stage["input_resolution"],
                "output_resolution": stage["output_resolution"],
                "batch_size": stage["batch_size"],
                "learning_rate": stage["learning_rate"],
                "augmentation_level": stage["augmentation"],
                "focus": stage["focus"],
                "lr_schedule": stage.get("lr_schedule", "cosine_annealing"),
                "warmup_epochs": stage.get("warmup_epochs", 3),
            }
            optimized_schedule["stages"].append(stage_config)

        config["optimized_training_schedules"][model_name] = optimized_schedule

    # Save updated configuration
    print(f"\n{'='*60}")
    print("UPDATING CONFIGURATION FILE")
    print("=" * 60)

    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(
            f"[SUCCESS] Updated {config_path} with optimized "
            f"training schedules"
        )
        print(
            f"[INFO] Added training schedules for {len(training_plans)} models"
        )

        # Verify the update
        with open(config_path, "r") as f:
            updated_config = json.load(f)

        if "optimized_training_schedules" in updated_config:
            schedules_count = len(
                updated_config["optimized_training_schedules"]
            )
            print(
                f"[SUCCESS] Verification: {schedules_count} optimized "
                f"schedules saved"
            )
        else:
            print(
                "[FAIL] Verification failed: schedules not found in "
                "updated config"
            )

    except Exception as e:
        print(f"[FAIL] Failed to update configuration file: {e}")

    # Final recommendations
    print("\n" + "=" * 60)
    print("FINAL RECOMMENDATIONS")
    print("=" * 60)
    print("1. Start with lightweight-baseline for quick validation")
    print("2. Use curriculum learning: 128px → 192px → 256px")
    print("3. Monitor validation loss at each stage")
    print("4. Save checkpoints after each curriculum stage")
    print("5. Use mixed precision training (fp16) to save memory")
    print("6. Enable gradient accumulation if batch sizes are too small")
    print("\nOptimized schedules saved to config file:")
    for model_name in training_plans.keys():
        print(
            f"  • {model_name}: "
            f"{training_plans[model_name]['total_epochs']} total epochs"
        )
    print("\nNext Steps:")
    print("- Use updated config file with optimized schedules")
    print("- Implement train.py with curriculum learning support")
    print("- Configure logging and checkpointing")
    print("- Start with the foundation stage (128px inputs)")

    return training_plans
