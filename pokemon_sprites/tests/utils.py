"""
Test utilities and common functions for the Pokemon sprite generation test suite.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


class TestDataFactory:
    """Factory for creating test data."""

    @staticmethod
    def create_test_image(
        size: tuple = (64, 64), color: str = "red"
    ) -> Image.Image:
        """Create a test image."""
        return Image.new("RGB", size, color=color)

    @staticmethod
    def create_test_tensor(shape: tuple, device: str = "cpu") -> torch.Tensor:
        """Create a test tensor."""
        return torch.randn(shape, device=device)

    @staticmethod
    def create_test_pokemon_pair(pokemon_id: str, base_dir: Path) -> tuple:
        """Create a test Pokemon sprite-artwork pair."""
        sprites_dir = base_dir / "sprites"
        artwork_dir = base_dir / "artwork"

        sprites_dir.mkdir(exist_ok=True)
        artwork_dir.mkdir(exist_ok=True)

        sprite_path = sprites_dir / f"pokemon_{pokemon_id}.png"
        artwork_path = artwork_dir / f"pokemon_{pokemon_id}_artwork.png"

        sprite_img = TestDataFactory.create_test_image((64, 64), "red")
        artwork_img = TestDataFactory.create_test_image((256, 256), "blue")

        sprite_img.save(sprite_path)
        artwork_img.save(artwork_path)

        return sprite_path, artwork_path


class TestEnvironment:
    """Manages test environment setup and cleanup."""

    def __init__(self):
        self.temp_dirs = []
        self.original_env = {}

    def create_temp_dir(self) -> Path:
        """Create a temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_dirs.append(temp_dir)
        return temp_dir

    def set_env_var(self, key: str, value: str):
        """Set environment variable and remember original."""
        if key not in self.original_env:
            self.original_env[key] = os.environ.get(key)
        os.environ[key] = value

    def cleanup(self):
        """Clean up test environment."""
        # Restore environment variables
        for key, original_value in self.original_env.items():
            if original_value is None:
                if key in os.environ:
                    del os.environ[key]
            else:
                os.environ[key] = original_value

        # Remove temporary directories
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

        self.temp_dirs.clear()
        self.original_env.clear()


class MockConfig:
    """Mock configuration for testing."""

    def __init__(self, **kwargs):
        self.batch_size = kwargs.get("batch_size", 4)
        self.num_epochs = kwargs.get("num_epochs", 10)
        self.learning_rate = kwargs.get("learning_rate", 0.0002)
        self.device = kwargs.get("device", "cpu")
        self.image_size = kwargs.get("image_size", (64, 64))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "device": self.device,
            "image_size": self.image_size,
        }


def skip_if_no_gpu(test_func):
    """Decorator to skip tests if no GPU is available."""
    import unittest

    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("GPU not available")
        return test_func(*args, **kwargs)

    return wrapper


def require_modules(*module_names):
    """Decorator to skip tests if required modules are not available."""
    import unittest

    def decorator(test_func):
        def wrapper(*args, **kwargs):
            for module_name in module_names:
                try:
                    __import__(module_name)
                except ImportError:
                    raise unittest.SkipTest(
                        f"Required module '{module_name}' not available"
                    )
            return test_func(*args, **kwargs)

        return wrapper

    return decorator


class PerformanceAssertion:
    """Helper for performance-related assertions."""

    @staticmethod
    def assert_execution_time(actual_time: float, max_time: float, test_case):
        """Assert that execution time is within acceptable limits."""
        test_case.assertLessEqual(
            actual_time,
            max_time,
            f"Execution time {actual_time:.3f}s exceeded limit {max_time:.3f}s",
        )

    @staticmethod
    def assert_memory_usage(
        actual_memory: float, max_memory: float, test_case
    ):
        """Assert that memory usage is within acceptable limits."""
        test_case.assertLessEqual(
            actual_memory,
            max_memory,
            f"Memory usage {actual_memory:.2f}MB exceeded limit {max_memory:.2f}MB",
        )

    @staticmethod
    def assert_throughput(
        actual_throughput: float, min_throughput: float, test_case
    ):
        """Assert that throughput meets minimum requirements."""
        test_case.assertGreaterEqual(
            actual_throughput,
            min_throughput,
            f"Throughput {actual_throughput:.2f} below minimum {min_throughput:.2f}",
        )


# Test constants
TEST_CONSTANTS = {
    "DEFAULT_IMAGE_SIZE": (64, 64),
    "DEFAULT_BATCH_SIZE": 4,
    "DEFAULT_TIMEOUT": 30.0,
    "MAX_MEMORY_MB": 1000,
    "PERFORMANCE_TOLERANCE": 0.1,
}
