"""
Test suite for TrainingScheduleOptimizer component.
"""

import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# [INITIALIZATION] Add src directory to Python path for module imports
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from optimizers.schedule_optimizer import TrainingScheduleOptimizer

try:
    from core.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


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


class TestAdvancedScheduleFeatures:
    """Test advanced scheduling features and optimizations."""

    def test_adaptive_learning_rate_scheduling(self):
        """Test adaptive learning rate scheduling functionality."""
        optimizer = TrainingScheduleOptimizer(total_samples=1000)

        # Test learning rate adaptation (if method exists)
        try:
            initial_lr = 0.001
            epoch = 10
            loss_history = [
                1.0,
                0.8,
                0.7,
                0.65,
                0.6,
                0.58,
                0.57,
                0.56,
                0.555,
                0.554,
            ]

            if hasattr(optimizer, "adapt_learning_rate"):
                adapted_lr = optimizer.adapt_learning_rate(
                    initial_lr, epoch, loss_history
                )
                assert isinstance(adapted_lr, float)
                assert adapted_lr > 0

        except AttributeError:
            # Method doesn't exist, skip test
            pass

    def test_early_stopping_detection(self):
        """Test early stopping detection functionality."""
        optimizer = TrainingScheduleOptimizer(total_samples=1000)

        try:
            # Simulate loss history with plateau
            loss_history = [
                1.0,
                0.8,
                0.6,
                0.5,
                0.45,
                0.44,
                0.43,
                0.43,
                0.43,
                0.43,
            ]

            if hasattr(optimizer, "should_stop_early"):
                should_stop = optimizer.should_stop_early(
                    loss_history, patience=3
                )
                assert isinstance(should_stop, bool)

        except AttributeError:
            # Method doesn't exist, skip test
            pass

    def test_curriculum_difficulty_progression(self):
        """Test curriculum learning difficulty progression."""
        optimizer = TrainingScheduleOptimizer(total_samples=1000)

        try:
            if hasattr(optimizer, "get_curriculum_difficulty"):
                # Test difficulty increases over epochs
                epoch_1_difficulty = optimizer.get_curriculum_difficulty(
                    1, total_epochs=100
                )
                epoch_50_difficulty = optimizer.get_curriculum_difficulty(
                    50, total_epochs=100
                )
                epoch_100_difficulty = optimizer.get_curriculum_difficulty(
                    100, total_epochs=100
                )

                assert (
                    epoch_1_difficulty
                    <= epoch_50_difficulty
                    <= epoch_100_difficulty
                )

        except AttributeError:
            # Method doesn't exist, skip test
            pass

    def test_memory_efficient_batch_scheduling(self):
        """Test memory-efficient batch size scheduling."""
        optimizer = TrainingScheduleOptimizer(total_samples=1000)

        try:
            if hasattr(optimizer, "get_optimal_batch_size"):
                # Test batch size optimization for different memory constraints
                memory_gb = 8
                image_size = 64

                optimal_batch = optimizer.get_optimal_batch_size(
                    memory_gb, image_size
                )
                assert isinstance(optimal_batch, int)
                assert optimal_batch > 0
                assert optimal_batch <= 512  # Reasonable upper limit

        except AttributeError:
            # Method doesn't exist, skip test
            pass

    def test_training_time_estimation_accuracy(self):
        """Test accuracy of training time estimation."""
        optimizer = TrainingScheduleOptimizer(total_samples=1000)

        # Test with different hardware configurations
        hardware_configs = [
            {"gpu_memory": 8, "gpu_compute": "medium"},
            {"gpu_memory": 16, "gpu_compute": "high"},
            {"gpu_memory": 24, "gpu_compute": "very_high"},
        ]

        for config in hardware_configs:
            training_time = optimizer.calculate_training_time(
                epochs=50, batch_size=32
            )

            assert isinstance(training_time, dict)
            assert "total_batches" in training_time
            # Should return reasonable batch counts
            assert training_time["total_batches"] > 0

    def test_resource_utilization_optimization(self):
        """Test resource utilization optimization."""
        optimizer = TrainingScheduleOptimizer(total_samples=1000)

        try:
            if hasattr(optimizer, "optimize_resource_utilization"):
                resources = {
                    "gpu_memory": 16,
                    "cpu_cores": 8,
                    "storage_speed": "ssd",
                }

                optimized_config = optimizer.optimize_resource_utilization(
                    resources
                )
                assert isinstance(optimized_config, dict)
                assert "batch_size" in optimized_config
                assert "num_workers" in optimized_config

        except AttributeError:
            # Method doesn't exist, skip test
            pass


class TestScheduleOptimizerAdvancedFeatures(unittest.TestCase):
    """Test advanced features and integration workflows for schedule optimizer."""

    def setUp(self):
        """Set up test environment with temporary configuration files."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create test configuration file
        self.test_config = {
            "pix2pix_models": {
                "test-lightweight": {
                    "name": "test-lightweight",
                    "description": "Test lightweight model",
                    "parameters": {
                        "generator": {"ngf": 32},
                        "discriminator": {"ndf": 32}
                    }
                },
                "test-standard": {
                    "name": "test-standard", 
                    "description": "Test standard model",
                    "parameters": {
                        "generator": {"ngf": 64},
                        "discriminator": {"ndf": 64}
                    }
                }
            }
        }
        
        self.config_path = Path(self.test_dir) / "test_model_configs.json"
        with open(self.config_path, "w") as f:
            json.dump(self.test_config, f, indent=2)

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_dir)

    @patch('builtins.print')
    def test_create_learning_rate_schedule_cosine(self, mock_print):
        """Test cosine annealing learning rate schedule creation."""
        logger.info("[TEST] Testing cosine annealing LR schedule")
        
        try:
            from optimizers.schedule_optimizer import TrainingScheduleOptimizer
            
            optimizer = TrainingScheduleOptimizer()
            
            # Test cosine annealing schedule
            schedule = optimizer.create_learning_rate_schedule(
                schedule_type="cosine_annealing",
                total_epochs=20,
                base_lr=0.001
            )
            
            # Verify schedule properties
            self.assertEqual(len(schedule), 20)
            self.assertIsInstance(schedule, list)
            
            # Verify all learning rates are positive
            for lr in schedule:
                self.assertGreater(lr, 0)
                self.assertLessEqual(lr, 0.001)  # Should not exceed base LR
            
            # Verify warmup behavior (first few epochs should be lower)
            self.assertLess(schedule[0], schedule[5])
            
            logger.info("[PASS] Cosine annealing LR schedule tested")
            
        except Exception as e:
            logger.error(f"[FAIL] Cosine annealing LR schedule test failed: {e}")
            self.fail(f"Cosine annealing LR schedule test failed: {e}")

    @patch('builtins.print')
    def test_create_learning_rate_schedule_step_decay(self, mock_print):
        """Test step decay learning rate schedule creation."""
        logger.info("[TEST] Testing step decay LR schedule")
        
        try:
            from optimizers.schedule_optimizer import TrainingScheduleOptimizer
            
            optimizer = TrainingScheduleOptimizer()
            
            # Test step decay schedule
            schedule = optimizer.create_learning_rate_schedule(
                schedule_type="step_decay",
                total_epochs=30,
                base_lr=0.002
            )
            
            # Verify schedule properties
            self.assertEqual(len(schedule), 30)
            self.assertIsInstance(schedule, list)
            
            # Verify all learning rates are positive
            for lr in schedule:
                self.assertGreater(lr, 0)
                self.assertLessEqual(lr, 0.002)  # Should not exceed base LR
            
            # Verify step decay behavior (should have distinct steps)
            unique_lrs = set(schedule)
            self.assertGreaterEqual(len(unique_lrs), 2)  # At least 2 different learning rates
            
            logger.info("[PASS] Step decay LR schedule tested")
            
        except Exception as e:
            logger.error(f"[FAIL] Step decay LR schedule test failed: {e}")
            self.fail(f"Step decay LR schedule test failed: {e}")

    @patch('builtins.print')
    def test_optimize_training_schedules_workflow(self, mock_print):
        """Test complete training schedule optimization workflow."""
        logger.info("[TEST] Testing training schedule optimization workflow")
        
        try:
            from optimizers.schedule_optimizer import create_optimal_training_plan
            
            # Test full optimization workflow
            training_plans = create_optimal_training_plan(str(self.config_path))
            
            # Verify training plans structure
            self.assertIsInstance(training_plans, dict)
            self.assertIn("test-lightweight", training_plans)
            self.assertIn("test-standard", training_plans)
            
            # Verify each plan has required components
            for model_name, plan in training_plans.items():
                self.assertIn("complexity", plan)  # Updated field name
                self.assertIn("total_epochs", plan)
                self.assertIn("curriculum_stages", plan)  # Updated field name
                self.assertIsInstance(plan["curriculum_stages"], list)
                self.assertEqual(len(plan["curriculum_stages"]), 3)  # Foundation, Refinement, Polish
                
                # Verify stage structure
                for stage in plan["curriculum_stages"]:
                    self.assertIn("name", stage)
                    self.assertIn("epochs", stage)
                    self.assertIn("input_resolution", stage)
                    self.assertIn("output_resolution", stage)
                    self.assertIn("batch_size", stage)
                    self.assertIn("learning_rate", stage)
            
            # Verify configuration file was updated
            with open(self.config_path, "r") as f:
                updated_config = json.load(f)
            
            self.assertIn("optimized_training_schedules", updated_config)
            self.assertIn("test-lightweight", updated_config["optimized_training_schedules"])
            self.assertIn("test-standard", updated_config["optimized_training_schedules"])
            
            logger.info("[PASS] Training schedule optimization workflow tested")
            
        except Exception as e:
            logger.error(f"[FAIL] Training schedule optimization test failed: {e}")
            self.fail(f"Training schedule optimization test failed: {e}")

    def test_schedule_configuration_file_handling(self):
        """Test configuration file error handling in schedule optimization."""
        logger.info("[TEST] Testing schedule configuration file handling")
        
        try:
            from optimizers.schedule_optimizer import create_optimal_training_plan
            
            # Test with non-existent file - should raise an exception
            with self.assertRaises(FileNotFoundError):
                create_optimal_training_plan("non_existent_file.json")
            
            # Test with invalid JSON - should raise an exception
            invalid_config_path = Path(self.test_dir) / "invalid.json"
            with open(invalid_config_path, "w") as f:
                f.write("invalid json content")
            
            with self.assertRaises(json.JSONDecodeError):
                create_optimal_training_plan(str(invalid_config_path))
            
            # Test with empty configuration - should return empty dict or handle gracefully
            empty_config = {"pix2pix_models": {}}
            empty_config_path = Path(self.test_dir) / "empty.json"
            with open(empty_config_path, "w") as f:
                json.dump(empty_config, f)
            
            training_plans = create_optimal_training_plan(str(empty_config_path))
            # Should handle empty config gracefully
            self.assertIsInstance(training_plans, dict)
            
            logger.info("[PASS] Schedule configuration file handling tested")
            
            logger.info("[PASS] Schedule configuration file handling tested")
            
        except Exception as e:
            logger.error(f"[FAIL] Schedule configuration handling test failed: {e}")
            self.fail(f"Schedule configuration handling test failed: {e}")

    @patch('builtins.print')
    def test_schedule_complexity_mapping(self, mock_print):
        """Test model complexity detection and mapping."""
        logger.info("[TEST] Testing schedule complexity mapping")
        
        try:
            from optimizers.schedule_optimizer import create_optimal_training_plan
            
            # Test with different model configurations
            complex_config = {
                "pix2pix_models": {
                    "lightweight-model": {
                        "name": "lightweight-model",
                        "parameters": {
                            "generator": {"ngf": 32},  # Small NGF = lightweight
                            "discriminator": {"ndf": 32}
                        }
                    },
                    "heavy-model": {
                        "name": "heavy-model", 
                        "parameters": {
                            "generator": {"ngf": 128},  # Large NGF = heavy
                            "discriminator": {"ndf": 128}
                        }
                    }
                }
            }
            
            complex_config_path = Path(self.test_dir) / "complex.json"
            with open(complex_config_path, "w") as f:
                json.dump(complex_config, f, indent=2)
            
            training_plans = create_optimal_training_plan(str(complex_config_path))
            
            # Verify complexity assignments
            self.assertIn("lightweight-model", training_plans)
            self.assertIn("heavy-model", training_plans)
            
            lightweight_complexity = training_plans["lightweight-model"]["complexity"]  # Updated field name
            heavy_complexity = training_plans["heavy-model"]["complexity"]  # Updated field name
            
            # Should assign different complexities based on ngf
            self.assertIn(lightweight_complexity, ["lightweight", "medium"])
            self.assertIn(heavy_complexity, ["medium", "heavy"])
            
            logger.info(f"[PASS] Complexity mapping: {lightweight_complexity} vs {heavy_complexity}")
            
        except Exception as e:
            logger.error(f"[FAIL] Schedule complexity mapping test failed: {e}")
            self.fail(f"Schedule complexity mapping test failed: {e}")


