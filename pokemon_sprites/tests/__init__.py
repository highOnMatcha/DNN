"""
Pokemon Sprite Generation Pipeline - Test Suite

This package contains comprehensive tests for ensuring code robustness
and reliability across all components of the pipeline.

Test Structure:
- unit/: Unit tests for individual functions and classes
- integration/: Integration tests for component interactions
- performance/: Performance and benchmark tests
"""

__version__ = "1.0.0"
__author__ = "Pokemon Sprite Generation Team"

# Test configuration
TEST_CONFIG = {
    "timeout": 300,  # 5 minutes default timeout
    "max_memory_mb": 2048,  # 2GB memory limit
    "performance_thresholds": {
        "inference_time_ms": 1000,
        "training_step_time_s": 10,
        "memory_efficiency": 0.7,
    },
}

# Import test utilities
from .test_runner import PipelineTestRunner

__all__ = ["PipelineTestRunner", "TEST_CONFIG"]
