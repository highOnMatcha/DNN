# Pokemon Sprite Generation Pipeline - Test Suite

This directory contains a comprehensive test suite designed to ensure the robustness, reliability, and performance of the Pokemon sprite generation pipeline. The test suite follows professional testing practices and provides maximum code coverage.

## 📁 Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── requirements-test.txt       # Test-specific dependencies
├── test_runner.py             # Comprehensive test runner
├── utils.py                   # Test utilities and helpers
├── unit/                      # Unit tests
│   ├── test_data_loaders.py   # Data loading functionality tests
│   ├── test_models.py         # Model architecture tests
│   ├── test_augmentation.py   # Data augmentation tests
│   └── test_config_settings.py # Configuration management tests
├── integration/               # Integration tests
│   └── test_pipeline_integration.py # End-to-end pipeline tests
└── performance/               # Performance tests
    └── test_performance_benchmarks.py # Performance benchmarks
```

## 🚀 Quick Start

### Running All Tests

```bash
# Run the complete test suite
python tests/test_runner.py

# Run with verbose output
python tests/test_runner.py --verbose
```

### Running Specific Test Suites

```bash
# Run only unit tests
python tests/test_runner.py --suite unit

# Run only integration tests
python tests/test_runner.py --suite integration

# Run only performance tests
python tests/test_runner.py --suite performance
```

### Using pytest (Alternative)

```bash
# Install test dependencies
pip install -r tests/requirements-test.txt

# Run all tests with coverage
pytest

# Run specific test category
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/performance/ -v
```

## 📊 Test Categories

### Unit Tests
Unit tests verify individual functions and classes in isolation:

- **Data Loaders** (`test_data_loaders.py`)
  - Pokemon data downloading with caching
  - Valid image pair detection
  - Image processing and resizing
  - Dataset creation and splitting
  - Statistics calculation

- **Models** (`test_models.py`)
  - UNet generator architecture
  - PatchGAN discriminator
  - Pix2Pix generator
  - Residual blocks and attention mechanisms
  - Model parameter efficiency

- **Data Augmentation** (`test_augmentation.py`)
  - Paired random transformations
  - Color jittering
  - Geometric transformations
  - Pipeline composition
  - Memory efficiency

- **Configuration** (`test_config_settings.py`)
  - Directory utilities
  - Configuration dataclasses
  - JSON serialization/deserialization
  - Environment variable handling

### Integration Tests
Integration tests verify component interactions:

- **Pipeline Integration** (`test_pipeline_integration.py`)
  - Data loading to model pipeline
  - Augmentation and model integration
  - Configuration-driven setup
  - Batch processing
  - Training loop integration
  - Directory structure management

### Performance Tests
Performance tests measure speed, memory usage, and efficiency:

- **Performance Benchmarks** (`test_performance_benchmarks.py`)
  - Data loading performance
  - Augmentation speed
  - Model inference timing
  - Memory usage monitoring
  - Concurrent operations
  - Sustained operation stability

## 🎯 Test Features

### Professional Output Format
All tests use colored terminal output with clear [SUCCESS]/[FAIL] indicators:

```
[SUCCESS] test_unet_forward_pass: Forward pass output shape: (1, 3, 256, 256)
[FAIL] test_invalid_config: Configuration validation failed as expected
[INFO] Running performance benchmark for UNet inference
```

### Comprehensive Coverage
- **Function Coverage**: Tests every public function and method
- **Edge Cases**: Handles invalid inputs, missing files, network errors
- **Error Handling**: Validates proper exception handling
- **Performance Metrics**: Measures execution time and memory usage

### GitHub CI Integration
Automated testing on push/pull requests with:
- Multi-Python version testing (3.8, 3.9, 3.10, 3.11)
- Code coverage reporting
- Security vulnerability scanning
- Performance regression detection

## 📈 Performance Thresholds

The test suite enforces performance standards:

| Metric | Threshold | Test Type |
|--------|-----------|-----------|
| Model Inference | < 1000ms | Performance |
| Training Step | < 10s | Performance |
| Data Loading | < 0.1s per file | Performance |
| Memory Efficiency | > 70% cleanup | Performance |
| Augmentation | < 100ms per pair | Performance |

## 🛠 Test Utilities

### TestDataFactory
Helper class for creating test data:
```python
from tests.utils import TestDataFactory

# Create test images
img = TestDataFactory.create_test_image((64, 64), 'red')

# Create test tensors
tensor = TestDataFactory.create_test_tensor((1, 3, 64, 64))

# Create Pokemon pairs
sprite_path, artwork_path = TestDataFactory.create_test_pokemon_pair("0001", base_dir)
```

### TestEnvironment
Manages test environment setup and cleanup:
```python
from tests.utils import TestEnvironment

env = TestEnvironment()
temp_dir = env.create_temp_dir()
env.set_env_var('POKEMON_DATA_ROOT', str(temp_dir))
# ... run tests ...
env.cleanup()  # Automatic cleanup
```

### Performance Assertions
Specialized assertions for performance testing:
```python
from tests.utils import PerformanceAssertion

PerformanceAssertion.assert_execution_time(actual_time, 1.0, self)
PerformanceAssertion.assert_memory_usage(actual_memory, 500, self)
```

## 🔧 Configuration

### pytest.ini
Configured for comprehensive testing:
- Coverage reporting (HTML, XML, terminal)
- Timeout handling (5 minutes default)
- Strict marker enforcement
- Categorized test markers

### CI/CD Integration
GitHub Actions workflow includes:
- Parallel test execution
- Coverage reporting to Codecov
- Security vulnerability scanning
- Multi-environment testing

## 📋 Test Guidelines

### Writing New Tests
1. **Professional Naming**: Use descriptive test names with `test_` prefix
2. **Color Coding**: Use `print_test_result()` for consistent output
3. **Documentation**: Include docstrings explaining test purpose
4. **Cleanup**: Always clean up resources in `tearDown()`
5. **Assertions**: Use specific assertions with descriptive messages

### Test Categories
Mark tests appropriately:
```python
@pytest.mark.unit
def test_function_logic(self):
    """Test individual function logic."""
    pass

@pytest.mark.integration
def test_component_interaction(self):
    """Test interaction between components."""
    pass

@pytest.mark.performance
def test_execution_speed(self):
    """Test execution speed and memory usage."""
    pass
```

### Error Handling
Test both success and failure cases:
```python
def test_invalid_input_handling(self):
    """Test that invalid inputs are handled correctly."""
    with self.assertRaises(ValueError):
        function_under_test(invalid_input)
```

## 📊 Coverage Reports

Coverage reports are generated in multiple formats:
- **Terminal**: Real-time coverage feedback
- **HTML**: Detailed line-by-line coverage (`htmlcov/index.html`)
- **XML**: Machine-readable format for CI integration

Target coverage: **≥ 80%**

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure src path is correct
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

2. **GPU Tests on CPU-Only Systems**
   ```python
   # Tests automatically skip if no GPU available
   @skip_if_no_gpu
   def test_gpu_functionality(self):
       pass
   ```

3. **Memory Issues**
   ```bash
   # Run performance tests individually
   python tests/test_runner.py --suite performance
   ```

4. **Timeout Issues**
   ```bash
   # Increase timeout for slow systems
   pytest --timeout=600  # 10 minutes
   ```

### Debug Mode
Enable verbose logging for debugging:
```bash
python tests/test_runner.py --verbose --failfast
```

## 🎯 Contributing

When adding new functionality:
1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain coverage above 80%
4. Add performance tests for critical paths
5. Update documentation

## 📞 Support

For test-related issues:
1. Check existing test documentation
2. Review error messages and logs
3. Verify environment setup
4. Check for dependency conflicts

The test suite is designed to be robust, informative, and maintainable, ensuring the highest quality standards for the Pokemon sprite generation pipeline.
