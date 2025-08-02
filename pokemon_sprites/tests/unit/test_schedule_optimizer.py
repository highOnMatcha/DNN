"""
Test suite for TrainingScheduleOptimizer component.
"""

import math
import sys
from pathlib import Path
from unittest.mock import patch

# [INITIALIZATION] Add src directory to Python path for module imports
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from optimizers.schedule_optimizer import TrainingScheduleOptimizer


class TestTrainingScheduleOptimizerInitialization:
    """Test suite for TrainingScheduleOptimizer initialization."""

    def test_default_initialization(self):
        """[SUCCESS] Verify default initialization parameters."""
        optimizer = TrainingScheduleOptimizer()

        assert optimizer.total_samples == 898
        assert optimizer.train_samples == int(
            898 * 0.85
        )  # 1 - 0.15 validation
        assert optimizer.val_samples == 898 - optimizer.train_samples

    def test_custom_initialization_parameters(self):
        """[SUCCESS] Verify custom initialization parameters work correctly."""
        total_samples = 1000
        validation_split = 0.2

        optimizer = TrainingScheduleOptimizer(
            total_samples=total_samples, validation_split=validation_split
        )

        assert optimizer.total_samples == total_samples
        assert optimizer.train_samples == int(total_samples * 0.8)
        assert optimizer.val_samples == total_samples - optimizer.train_samples

    def test_validation_split_edge_cases(self):
        """[SUCCESS] Verify validation split edge cases."""
        # Zero validation split
        optimizer = TrainingScheduleOptimizer(
            total_samples=100, validation_split=0.0
        )
        assert optimizer.train_samples == 100
        assert optimizer.val_samples == 0

        # Maximum validation split - let's verify the actual calculation
        optimizer = TrainingScheduleOptimizer(
            total_samples=100, validation_split=0.9
        )
        expected_train = int(100 * (1 - 0.9))  # Should be 10
        expected_val = 100 - expected_train  # Should be 90
        assert optimizer.train_samples == expected_train
        assert optimizer.val_samples == expected_val


class TestTrainingTimeCalculation:
    """Test suite for training time calculation functionality."""

    def test_training_time_calculation_basic(self):
        """[SUCCESS] Verify basic training time calculation."""
        optimizer = TrainingScheduleOptimizer(
            total_samples=800, validation_split=0.2
        )

        result = optimizer.calculate_training_time(epochs=10, batch_size=16)

        expected_train_samples = int(800 * 0.8)  # 640
        expected_batches_per_epoch = math.ceil(
            expected_train_samples / 16
        )  # 40
        expected_total_batches = expected_batches_per_epoch * 10  # 400

        assert result["batches_per_epoch"] == expected_batches_per_epoch
        assert result["total_batches"] == expected_total_batches
        assert result["samples_per_epoch"] == expected_train_samples
        assert "estimated_time_per_batch" in result

    def test_training_time_with_custom_samples(self):
        """[SUCCESS] Verify training time calculation with custom sample count."""
        optimizer = TrainingScheduleOptimizer()

        custom_samples = 500
        result = optimizer.calculate_training_time(
            epochs=5, batch_size=10, samples_per_epoch=custom_samples
        )

        expected_batches_per_epoch = math.ceil(custom_samples / 10)  # 50
        expected_total_batches = expected_batches_per_epoch * 5  # 250

        assert result["batches_per_epoch"] == expected_batches_per_epoch
        assert result["total_batches"] == expected_total_batches
        assert result["samples_per_epoch"] == custom_samples

    def test_training_time_batch_size_edge_cases(self):
        """[SUCCESS] Verify training time calculation with edge case batch sizes."""
        optimizer = TrainingScheduleOptimizer(
            total_samples=100, validation_split=0.0
        )

        # Very small batch size
        result = optimizer.calculate_training_time(epochs=1, batch_size=1)
        assert result["batches_per_epoch"] == 100
        assert result["total_batches"] == 100

        # Large batch size
        result = optimizer.calculate_training_time(epochs=1, batch_size=200)
        assert result["batches_per_epoch"] == 1  # ceil(100/200) = 1
        assert result["total_batches"] == 1

    def test_training_time_zero_epochs(self):
        """[SUCCESS] Verify training time calculation with zero epochs."""
        optimizer = TrainingScheduleOptimizer()

        result = optimizer.calculate_training_time(epochs=0, batch_size=16)

        assert result["total_batches"] == 0
        assert (
            result["batches_per_epoch"] > 0
        )  # Should still calculate per epoch


class TestCurriculumScheduleCreation:
    """Test suite for curriculum learning schedule creation."""

    @patch("builtins.print")
    def test_curriculum_schedule_default_complexity(self, mock_print):
        """[SUCCESS] Verify default curriculum schedule creation."""
        optimizer = TrainingScheduleOptimizer()

        schedule = optimizer.create_curriculum_schedule()

        # Verify schedule structure
        assert len(schedule) == 3  # Foundation, Refinement, Polish stages
        assert all("name" in stage for stage in schedule)
        assert all("input_resolution" in stage for stage in schedule)
        assert all("output_resolution" in stage for stage in schedule)
        assert all("epochs" in stage for stage in schedule)
        assert all("batch_size" in stage for stage in schedule)
        assert all("learning_rate" in stage for stage in schedule)

        # Verify stage progression
        assert schedule[0]["name"] == "Foundation Stage"
        assert schedule[1]["name"] == "Refinement Stage"
        assert schedule[2]["name"] == "Polish Stage"

        # Verify resolution progression
        assert schedule[0]["input_resolution"] == 128
        assert schedule[1]["input_resolution"] == 192
        assert schedule[2]["input_resolution"] == 256

        # Verify all output at 256px
        assert all(stage["output_resolution"] == 256 for stage in schedule)

    @patch("builtins.print")
    def test_curriculum_schedule_lightweight_complexity(self, mock_print):
        """[SUCCESS] Verify lightweight complexity curriculum schedule."""
        optimizer = TrainingScheduleOptimizer()

        schedule = optimizer.create_curriculum_schedule(
            model_complexity="lightweight"
        )

        # Lightweight should have larger batch sizes
        foundation_batch = schedule[0]["batch_size"]

        # Compare with medium complexity
        medium_schedule = optimizer.create_curriculum_schedule(
            model_complexity="medium"
        )
        medium_foundation_batch = medium_schedule[0]["batch_size"]

        assert foundation_batch >= medium_foundation_batch

    @patch("builtins.print")
    def test_curriculum_schedule_heavy_complexity(self, mock_print):
        """[SUCCESS] Verify heavy complexity curriculum schedule."""
        optimizer = TrainingScheduleOptimizer()

        schedule = optimizer.create_curriculum_schedule(
            model_complexity="heavy"
        )

        # Heavy should have smaller batch sizes and lower learning rates
        foundation_batch = schedule[0]["batch_size"]
        foundation_lr = schedule[0]["learning_rate"]

        # Compare with medium complexity
        medium_schedule = optimizer.create_curriculum_schedule(
            model_complexity="medium"
        )
        medium_foundation_batch = medium_schedule[0]["batch_size"]
        medium_foundation_lr = medium_schedule[0]["learning_rate"]

        assert foundation_batch <= medium_foundation_batch
        assert foundation_lr <= medium_foundation_lr

    @patch("builtins.print")
    def test_curriculum_schedule_unknown_complexity(self, mock_print):
        """[SUCCESS] Verify unknown complexity defaults to medium."""
        optimizer = TrainingScheduleOptimizer()

        unknown_schedule = optimizer.create_curriculum_schedule(
            model_complexity="unknown"
        )
        medium_schedule = optimizer.create_curriculum_schedule(
            model_complexity="medium"
        )

        # Should be identical to medium complexity
        assert len(unknown_schedule) == len(medium_schedule)
        for unknown_stage, medium_stage in zip(
            unknown_schedule, medium_schedule
        ):
            assert unknown_stage["batch_size"] == medium_stage["batch_size"]
            assert (
                unknown_stage["learning_rate"] == medium_stage["learning_rate"]
            )

    @patch("builtins.print")
    def test_curriculum_schedule_stage_properties(self, mock_print):
        """[SUCCESS] Verify curriculum schedule stage properties."""
        optimizer = TrainingScheduleOptimizer()

        schedule = optimizer.create_curriculum_schedule()

        total_epochs = 0
        for i, stage in enumerate(schedule):
            # Verify stage numbering
            assert stage["stage_number"] == i + 1

            # Verify timing information exists
            assert "timing" in stage
            assert "batches_per_epoch" in stage["timing"]
            assert "total_batches" in stage["timing"]

            # Verify learning rate schedule
            assert stage["lr_schedule"] == "cosine_annealing"

            # Verify warmup epochs
            expected_warmup = 3 if i == 0 else 2
            assert stage["warmup_epochs"] == expected_warmup

            # Verify cumulative epochs
            total_epochs += stage["epochs"]
            assert stage["cumulative_epochs"] == total_epochs

            # Verify augmentation levels
            assert stage["augmentation"] in ["minimal", "moderate", "strong"]

    @patch("builtins.print")
    def test_curriculum_schedule_learning_rate_progression(self, mock_print):
        """[SUCCESS] Verify learning rate decreases across stages."""
        optimizer = TrainingScheduleOptimizer()

        schedule = optimizer.create_curriculum_schedule(
            model_complexity="medium"
        )

        # Learning rates should decrease across stages
        foundation_lr = schedule[0]["learning_rate"]
        refinement_lr = schedule[1]["learning_rate"]
        polish_lr = schedule[2]["learning_rate"]

        assert foundation_lr > refinement_lr > polish_lr

    @patch("builtins.print")
    def test_curriculum_schedule_batch_size_logic(self, mock_print):
        """[SUCCESS] Verify batch size calculation logic."""
        optimizer = TrainingScheduleOptimizer()

        schedule = optimizer.create_curriculum_schedule(
            model_complexity="medium"
        )

        # Batch sizes should generally increase as resolution decreases
        # (earlier stages have lower resolution, should have larger batches)
        foundation_batch = schedule[0]["batch_size"]  # 128px resolution
        refinement_batch = schedule[1]["batch_size"]  # 192px resolution
        polish_batch = schedule[2]["batch_size"]  # 256px resolution

        # Foundation (lowest res) should have highest batch size
        assert foundation_batch >= refinement_batch
        assert refinement_batch >= polish_batch

        # All batch sizes should be at least 1
        assert all(stage["batch_size"] >= 1 for stage in schedule)


class TestTrainingScheduleOptimizerIntegration:
    """Test suite for TrainingScheduleOptimizer integration scenarios."""

    @patch("builtins.print")
    def test_full_schedule_workflow(self, mock_print):
        """[SUCCESS] Verify complete schedule creation workflow."""
        optimizer = TrainingScheduleOptimizer(
            total_samples=1000, validation_split=0.2
        )

        # Create schedule for different complexities
        for complexity in ["lightweight", "medium", "heavy"]:
            schedule = optimizer.create_curriculum_schedule(
                model_complexity=complexity
            )

            # Verify schedule integrity
            assert len(schedule) == 3

            # Verify all stages have required components
            for stage in schedule:
                timing = optimizer.calculate_training_time(
                    epochs=stage["epochs"], batch_size=stage["batch_size"]
                )

                # Timing should be consistent
                assert (
                    stage["timing"]["batches_per_epoch"]
                    == timing["batches_per_epoch"]
                )
                assert (
                    stage["timing"]["total_batches"] == timing["total_batches"]
                )

    def test_schedule_optimizer_edge_cases(self):
        """[SUCCESS] Verify schedule optimizer handles edge cases."""
        # Very small dataset
        small_optimizer = TrainingScheduleOptimizer(
            total_samples=10, validation_split=0.1
        )
        assert small_optimizer.train_samples == 9
        assert small_optimizer.val_samples == 1

        # Large dataset
        large_optimizer = TrainingScheduleOptimizer(
            total_samples=100000, validation_split=0.15
        )
        assert large_optimizer.train_samples == 85000
        assert large_optimizer.val_samples == 15000

        # Extreme validation split
        extreme_optimizer = TrainingScheduleOptimizer(
            total_samples=100, validation_split=0.99
        )
        assert extreme_optimizer.train_samples == 1
        assert extreme_optimizer.val_samples == 99
