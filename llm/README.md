# LLM Training Pipeline

Training pipeline for instruction-response language models using the Alpaca-GPT4 dataset. Supports data analysis, model training, and text generation with both in-memory and streaming data processing.

## Quick Start

```bash
# Train a model
python src/train.py --model custom-tiny --config test

# Generate text
python src/generate.py --model custom-tiny "Explain AI"

# Interactive mode
python src/generate.py --interactive --model custom-tiny

# List available models
python src/generate.py --list-models
```

## Setup

```bash
cd llm
pip install -r requirements.txt
```

Optional WandB integration for experiment tracking:
```bash
cp .env.example .env
# Add API key from https://wandb.ai/authorize
```

## Usage

### Training
```bash
# Basic training
python src/train.py --model custom-tiny --config test

# With custom sample count
python src/train.py --model gpt2-small --config development --samples 500

# Resume from checkpoint
python src/train.py --resume ./models/custom_tiny/checkpoint-1000

# Streaming mode for large datasets
python src/train.py --model custom-medium --config production --streaming
```

### Generation
```bash
# Single prompt
python src/generate.py --model custom-tiny "Write a haiku about coding"

# Batch generation
python src/generate.py --model custom-tiny --file prompts.txt

# Interactive session
python src/generate.py --interactive --model custom-tiny

# With custom parameters
python src/generate.py --model custom-tiny "Tell me a story" --max-length 200 --temperature 0.8
```

## Architecture

```
src/
├── train.py              # Training entry point
├── generate.py           # Text generation interface
├── core/
│   ├── models.py         # Model architectures and loading
│   └── trainer.py        # Training logic and experiment management
├── data/
│   ├── loaders.py        # Dataset loading and preprocessing
│   └── streaming.py      # Large dataset streaming utilities
└── config/
    ├── model_configs.json # Model architecture definitions
    └── settings.py        # Training configurations
```

The pipeline automatically handles model loading, tokenization, dataset preparation, and checkpoint management. Both local CSV files and PostgreSQL databases are supported as data sources.
