# Training Pipeline Documentation

## Overview

This directory contains a comprehensive single-model training pipeline (`train.py`) with integrated WandB logging for dialog model training using the Alpaca-GPT4 dataset.

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
