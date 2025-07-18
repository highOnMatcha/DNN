# Deep Neural Network tasks for practice/fun

## Setup environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Database Configuration (Optional)

This project can optionally use PostgreSQL for data storage. The database functionality is built into the `dataset_utils.py` module but is not required for training.

### Setup (only if you want to use database features):

### 1. Copy the example environment script
```bash
cp set_db_env.example.sh set_db_env.sh
```

### 2. Edit the database credentials
```bash
export DB_HOST="your_database_host"
export DB_PORT="5432"
export DB_NAME="your_database_name"
export DB_USER="your_username"
export DB_PASSWORD="your_password"
```

### 3. Use database features in your code
```python
from dataset_utils import get_database_manager
db_manager = get_database_manager()
```

## Language Model for Movie dialogue/script 

### Alpaca-GPT4 Dataset Training with PyTorch
The project includes a complete PyTorch-based training pipeline for fine-tuning language models:

**Files:**
- `llm/dataset_utils.py` - Pure utility classes for dataset management
- `llm/train.py` - PyTorch training script with Transformers integration

**Usage:**

```bash
# Analyze dataset only (no training)
cd llm
python train.py --analyze-only

# Train a language model (full PyTorch pipeline)
cd llm
python train.py

# Use CPU only (if no GPU available)
cd llm
python train.py --no-cuda
```

**Dataset Management:**
- Automatically downloads dataset if not found locally
- Saves to `llm/data/alpaca-gpt4.csv` (approx. 85 MB)
- Loads from local file on subsequent runs
- Option to load in memory only for large datasets

**Training Features:**
- PyTorch + Transformers integration
- Automatic model downloading from HuggingFace
- Custom dataset class for Alpaca format
- Configurable training parameters
- GPU/CPU support
- Model evaluation and text generation
- Checkpoint saving and loading

**Model Details:**
- Default model: microsoft/DialoGPT-small (124M parameters)
- Training: Causal language modeling on instruction-response pairs
- Dataset: 52,002 instruction-response pairs from Alpaca-GPT4
- Format: Properly formatted training text with instruction/response structure

**Dataset Information:**
- Contains 52,002 instruction-response pairs
- Stored in `llm/data/alpaca-gpt4.csv` for better organization
- Automatically downloads from HuggingFace if not present locally

**Dataset Fields:**
- `instruction`: The task or question
- `input`: Additional context (often empty)
- `output`: The response/answer
- `text`: Formatted training text combining instruction and output

**Hardware Requirements:**
- CPU: Works on any modern CPU (slower training)
- GPU: CUDA-compatible GPU recommended for faster training
- RAM: 8GB+ recommended (depending on model size and batch size)

## Time-Series prediction for Trading

## Segmentation CV Neural network