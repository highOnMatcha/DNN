# LLM Training and Generation

Instruction-response models using the Alpaca-GPT4 dataset. Training and evaluation of different architectures for text generation tasks.

## Quick Start

From the LLM directory:

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

Install dependencies:
```bash
cd llm
pip install -r requirements.txt
```

Optional WandB tracking:
```bash
cp .env.example .env
# Add your WandB API key from https://wandb.ai/authorize
```

## Usage

## Usage

### Training Models
Use the comprehensive training script:
```bash
# Basic training
python src/train.py --model custom-tiny --config test

# Development training with custom samples
python src/train.py --model gpt2-small --config development --samples 500

# Resume from checkpoint
python src/train.py --model custom-tiny --config test --resume ./models/custom_tiny/checkpoint-1000

# Production training
python src/train.py --model custom-small --config production

# List available models
python src/train.py --list-models
```

### Text Generation
Use the dedicated generation script:
```bash
# Single generation
python src/generate.py --model custom-tiny "Explain machine learning"

# Longer responses
python src/generate.py --model gpt2-small "Write a poem" --max-length 200

# Interactive session
python src/generate.py --interactive --model custom-tiny

# List available models
python src/generate.py --list-models
```

### Notebooks
Notebooks provide the main development interface:
- `01_data_exploration.ipynb` for dataset analysis and quality assessment
- Data exploration covers length distributions and optimal model configurations
- Encoder-decoder architectures perform better than decoder-only for the 11x expansion ratio in this dataset

## Project Structure

```
llm/
├── src/                         # Core implementation
│   ├── core/                    # Training and model logic
│   ├── data/                    # Data loading and preprocessing
│   ├── config/                  # Model and training configurations
│   └── train.py                 # Main training script
├── notebooks/                   # Development and analysis
│   └── 01_data_exploration.ipynb    # Dataset analysis
├── models/                      # Trained model outputs
│   ├── custom_tiny/             # Small model for testing
│   ├── custom_medium/           # Medium-sized model
│   └── custom_large/            # Large model configurations
└── data/                        # Training datasets
```

## Configuration

Model configs in `src/config/model_configs.json`. Data exploration determines optimal parameters including max_position_embeddings (2048) and batch sizes based on dataset characteristics.

## Experiment Tracking

Uses WandB for comprehensive experiment logging including:
- Training metrics (loss, learning rate, etc.)
- Generation testing results
- Model configuration tracking
- Automatic checkpoint management

Set `WANDB_API_KEY` in your environment and the system handles logging automatically. Training saves checkpoints every few steps (configurable per training config). Pause training with `Ctrl+C` and resume later using the `--resume` flag.