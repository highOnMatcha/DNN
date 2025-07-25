# LLM Training Pipeline

Training script for instruction-response models on Alpaca-GPT4 dataset.

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

WandB (optional):
```bash
cp .env.example .env
# Add API key from https://wandb.ai/authorize
```

## Usage

## Usage

### Training
```bash
# Basic
python src/train.py --model custom-tiny --config test

# With custom sample count
python src/train.py --model gpt2-small --config development --samples 500

# Resume from checkpoint
python src/train.py --resume ./models/custom_tiny/checkpoint-1000

# List models
python src/train.py --list-models
```

### Generation
```bash
# Single prompt
python src/generate.py --model custom-tiny "Explain machine learning"

# Interactive
python src/generate.py --interactive --model custom-tiny

# Custom length
python src/generate.py --model gpt2-small "Write a poem" --max-length 200
```

### Notebooks
- `01_data_exploration.ipynb` - Dataset analysis and quality assessment
- Shows length distributions and optimal model configurations
- Encoder-decoder works better than decoder-only for 11x expansion ratio

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

Model configs in `src/config/model_configs.json`. Max position embeddings set to 2048 based on dataset analysis.

### Learning Rate & Early Stopping

- **Test**: `constant_with_warmup`, no early stopping (fast experimentation)
- **Development**: `cosine` scheduler, 5% warmup, patience=3
- **Production**: `cosine` scheduler, 10% warmup, patience=5

Settings in `model_configs.json`. See `LEARNING_RATE_AND_PATIENCE.md`.

**Schedulers**: `cosine`, `constant_with_warmup`, `linear`, `cosine_with_restarts`, `polynomial`, `constant`

**Early Stopping**: Monitors eval loss, saves best checkpoint, configurable patience.

## Experiment Tracking

WandB logging:
- Training metrics (loss, learning rate)
- Generation testing results
- Model configuration
- Checkpoint management

Set `WANDB_API_KEY` environment variable. Checkpoints saved automatically. Resume with `--resume` flag.