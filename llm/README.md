# Dialog Model Training Project

PyTorch framework for training and comparing dialog models. Research project for experimenting with different ML architectures on conversational AI tasks.

## Features

- Train pre-trained models (GPT-2, DialoGPT) or build custom architectures from scratch
- Compare multiple model variants in controlled experiments
- WandB integration for experiment tracking and visualization
- Interactive Jupyter notebooks for step-by-step development
- Direct approach focused on research and learning

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Notebooks (Recommended)
Work through the interactive notebooks step-by-step:

1. **Data Exploration**: `notebooks/01_data_exploration.ipynb`
   - Load and visualize dialog data
   - Quality analysis and statistics
   - WandB logging setup

2. **Model Training**: `notebooks/02_model_training.ipynb`
   - Train multiple model variants
   - Compare architectures and performance
   - Interactive dialog testing

### Command Line Scripts
Alternatively use the CLI scripts:

```bash
# List available models
python scripts/train_model.py --list-models

# Train a model
python scripts/train_model.py --model gpt2-small

# Generate text
python scripts/generate_text.py --model gpt2-small --instruction "Explain AI"
```

## Project Structure

```
llm/
├── src/                         # Source code
│   ├── core/                    # Model training logic
│   ├── data/                    # Data loading utilities
│   ├── config/                  # Model configurations
│   └── utils/                   # Helper functions
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
├── scripts/                     # CLI utilities (optional)
├── data/                        # Dataset storage
└── outputs/                     # Training results
    ├── models/                  # Saved models
    ├── logs/                    # Training logs
    └── generated/               # Generated samples
```

## Available Models

### Pre-trained (Fine-tuning)
- **gpt2-small**: 117M parameters
- **gpt2-medium**: 345M parameters  
- **dialogpt-small**: For dialogue tasks
- **dialogpt-medium**: Larger dialogue model
- **alpaca-dialogpt**: Trained on Alpaca data

### Custom (From Scratch)
- **custom-small**: 50M params, 8 layers, 512 dim
- **custom-medium**: 100M params, 12 layers, 768 dim
- **custom-large**: 200M params, 16 layers, 1024 dim
- **custom-tiny**: 10M params, 4 layers, 256 dim

## Configuration

Model settings are in `src/config/model_configs.json`. Training configs:
- `get_test_config()`: 1 epoch, 100 samples  
- `get_development_config()`: 2 epochs, 1000 samples
- `get_production_config()`: 3 epochs, full dataset

## Research Usage

```python
# Load in notebook
from src.core.trainer import DialogTrainer
from src.config.settings import get_model_config

model_config = get_model_config("custom-small")
trainer = DialogTrainer(model_config=model_config)
```

## Changelog

## Experiment Tracking

All training runs are logged to WandB:
- Training metrics and loss curves
- Model comparisons and performance
- Generated text samples
- Hardware utilization

Set up WandB account and run notebooks to start tracking experiments.

## Results

Check the notebooks for:
- Data analysis insights
- Model comparison results  
- Dialog generation examples
- Training performance metrics
