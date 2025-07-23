# Training Pipeline Documentation

## Overview

This directory contains a comprehensive single-model training pipeline (`train.py`) with integrated WandB logging for dialog model training using the Alpaca-GPT4 dataset.

## Features

### 🚀 **Single Model Training Pipeline**
- Trains one model at a time with comprehensive logging
- Supports both pre-trained and custom (from-scratch) models
- Configurable training parameters via JSON configuration files
- Built-in dataset loading and preprocessing
- Automatic model saving and checkpointing

### 📊 **Comprehensive WandB Integration**
- **Model Architecture Logging**: Parameters, layers, dimensions
- **Training Metrics**: Loss curves, learning rates, timing
- **System Monitoring**: GPU memory usage, training time
- **Gradient/Weight Tracking**: Histograms and distributions via `wandb.watch()`
- **Generation Testing**: Response quality metrics and comparison tables
- **Experiment Tracking**: Full experiment metadata and results

### 🎯 **Configuration Management**
- **Test Config**: 1 epoch, 2 batch size, 100 samples (quick testing)
- **Development Config**: 2 epochs, 4 batch size, 1000 samples (development)
- **Production Config**: 3 epochs, 8 batch size, full dataset (production)

## Quick Start

### 1. Setup Environment
```bash
# Activate virtual environment
source venv/bin/activate

# Ensure WandB API key is set
export WANDB_API_KEY=your_wandb_key
# OR add to .env file: WANDB_API_KEY=your_wandb_key
```

### 2. List Available Models
```bash
python src/train.py --list-models
```

### 3. Train a Model
```bash
# Quick test with tiny model (recommended first run)
python src/train.py --model custom-tiny --config test --samples 50

# Development training
python src/train.py --model custom-small --config development

# Production training
python src/train.py --model gpt2-small --config production
```

## Available Models

### Pre-trained Models (Fine-tuning)
- **gpt2-small**: Base GPT-2 (117M params)
- **gpt2-medium**: Medium GPT-2 (345M params) 
- **dialogpt-small**: DialoGPT Small (117M params)
- **dialogpt-medium**: DialoGPT Medium (345M params)

### Custom Models (From Scratch)
- **custom-nano**: Ultra-tiny (5M params, 3 layers, 128 dim)
- **custom-tiny**: Tiny model (10M params, 4 layers, 256 dim)
- **custom-small**: Small model (50M params, 8 layers, 512 dim)
- **custom-medium**: Medium model (100M params, 12 layers, 768 dim)
- **custom-large**: Large model (200M params, 16 layers, 1024 dim)

## Training Configuration Types

### Test Configuration
```json
{
  "epochs": 1,
  "batch_size": 2, 
  "max_samples": 100,
  "description": "Quick testing configuration"
}
```

### Development Configuration  
```json
{
  "epochs": 2,
  "batch_size": 4,
  "max_samples": 1000,
  "description": "Development training configuration"
}
```

### Production Configuration
```json
{
  "epochs": 3,
  "batch_size": 8,
  "max_samples": null,
  "description": "Full production training"
}
```

## Command Line Options

```bash
python src/train.py [OPTIONS]

Options:
  --model, -m MODEL              Model to train (default: custom-tiny)
  --config, -c CONFIG            Configuration: test|development|production
  --samples, -s SAMPLES          Max samples (overrides config)
  --project, -p PROJECT          WandB project name
  --list-models                  List available models and exit
  --no-cuda                      Disable CUDA
  --help, -h                     Show help message
```

## WandB Integration Details

### What Gets Logged

#### Model Information
- Total and trainable parameters
- Model architecture details
- Configuration parameters

#### Training Metrics
- Loss curves (train/validation)
- Learning rate schedule
- Training duration and timing
- GPU memory usage
- Gradient and weight histograms

#### Generation Testing
- Response generation metrics
- Generation time and quality
- Comparison tables across test prompts
- Success rates and statistics

#### Experiment Metadata
- Training configuration
- Dataset information
- System specifications
- Timestamps and versions

### WandB Project Structure
- **Project**: Configurable (default: "dialog-model-training")
- **Run Names**: `{model}-{config}-{timestamp}`
- **Tags**: Model name, config type, "single-model", "pipeline"

## Example Workflows

### 1. Quick Model Testing
```bash
# Test if everything works (30 seconds)
python src/train.py --model custom-nano --config test --samples 20

# Quick evaluation of tiny model (2 minutes)
python src/train.py --model custom-tiny --config test --samples 100
```

### 2. Development Workflow
```bash
# Develop with small model (10-15 minutes)
python src/train.py --model custom-small --config development

# Test pre-trained model fine-tuning (5-10 minutes)
python src/train.py --model gpt2-small --config development
```

### 3. Production Training
```bash
# Full training with medium model (several hours)
python src/train.py --model custom-medium --config production

# Production fine-tuning (1-2 hours)
python src/train.py --model gpt2-medium --config production
```

## Output Structure

```
models/
├── custom_tiny/           # Model checkpoints and final model
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── training_args.bin
└── wandb/                 # WandB logs and artifacts
    └── run-{timestamp}-{id}/
        ├── logs/
        └── files/
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the `llm/` directory
2. **CUDA Issues**: Use `--no-cuda` flag if GPU memory is insufficient
3. **WandB Authentication**: Set `WANDB_API_KEY` environment variable
4. **Memory Issues**: Reduce batch size or use smaller model

### Performance Tips

1. **Start Small**: Begin with `custom-nano` or `custom-tiny`
2. **Use GPU**: Significantly faster training on CUDA devices
3. **Monitor Memory**: Watch WandB system metrics for optimization
4. **Batch Size**: Adjust based on available memory

## Integration with Notebooks

The training pipeline can be imported and used in Jupyter notebooks:

```python
from src.train import train_single_model

# Train model programmatically
trainer, metrics = train_single_model(
    model_name="custom-tiny",
    config_type="test", 
    max_samples=50,
    project_name="notebook-experiment"
)
```

## Next Steps

1. **Experiment Tracking**: Use WandB dashboard to compare runs
2. **Hyperparameter Tuning**: Modify configurations in `model_configs.json`
3. **Custom Models**: Add new model architectures to the configuration
4. **Advanced Training**: Extend pipeline with additional features
