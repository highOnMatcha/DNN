# NLP Research Project

Instruction-response models using the Alpaca-GPT4 dataset. Training and evaluation of different architectures for text generation tasks.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Notebooks provide the main interface. Start with `01_data_exploration.ipynb` for dataset analysis, then `02_model_training.ipynb` for training experiments.

Data exploration covers quality analysis, length distributions, and optimal model configurations. Encoder-decoder architectures perform better than decoder-only for the 11x expansion ratio in this dataset.

Command line training available:
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

Text generation testing:
```bash
python scripts/generate_text.py --model gpt2-small --instruction "Explain AI"
```

## Project structure

```
llm/
├── src/                         # Core code
│   ├── core/                    # Training logic
│   ├── data/                    # Data loaders
│   └── config/                  # Model configurations
├── notebooks/                   # Development environment
│   ├── 01_data_exploration.ipynb    # Dataset analysis
│   └── 02_model_training.ipynb      # Model experiments
└── scripts/                     # CLI utilities
```

## Configuration

Model configs in `src/config/model_configs.json`. Data exploration determines optimal parameters including max_position_embeddings (2048) and batch sizes based on dataset characteristics.

## Tracking

Uses WandB for experiment logging. Set `WANDB_API_KEY` in your environment and the system handles logging automatically.

Training automatically saves checkpoints every few steps (configurable per training config). You can pause training with `Ctrl+C` and resume later using the `--resume` flag.

Set up environment variables:
```bash
cp .env.example .env
# Add your WandB API key from https://wandb.ai/authorize
```