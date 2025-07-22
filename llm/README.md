# NLP Research Project

Working on instruction-response models using the Alpaca-GPT4 dataset. Experimenting with different architectures and training approaches for text generation tasks.

## Setup

Install dependencies and you're good to go:
```bash
pip install -r requirements.txt
```

## Usage

The main work happens in the notebooks. Start with `01_data_exploration.ipynb` to understand the dataset characteristics, then move to `02_model_training.ipynb` for experiments.

The data exploration notebook covers dataset quality analysis, length distributions, and figures out optimal model configurations based on the data patterns. Found that encoder-decoder architectures work better than decoder-only for this 11x expansion ratio (instruction → response).

For quick testing there are also command line scripts:
```bash
python scripts/train_model.py --model gpt2-small
python scripts/generate_text.py --model gpt2-small --instruction "Explain AI"
```

## Project structure

```
llm/
├── src/                         # Core code
│   ├── core/                    # Training logic
│   ├── data/                    # Data loaders (supports both DB and file loading)
│   └── config/                  # Model configurations
├── notebooks/                   # Main development environment
│   ├── 01_data_exploration.ipynb    # Dataset analysis and config planning
│   └── 02_model_training.ipynb      # Model experiments
└── scripts/                     # CLI utilities
```

## Configuration

Model configs are in `src/config/model_configs.json`. The data exploration notebook determines optimal parameters like max_position_embeddings (2048 for this dataset) and batch sizes based on actual data characteristics.

## Tracking

Uses WandB for experiment logging. Set `WANDB_API_KEY` in your `.env` file and the notebooks handle the rest.

Copy the .env.example to `.env` and fill in your WandB API key, after logging in to https://wandb.ai/authorize:
```bash
cp .env.example .env
```