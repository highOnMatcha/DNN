# LSTM Stock Price Prediction Pipeline

Deep learning pipeline for predicting stock prices using LSTM neural networks. Provides data collection, feature engineering, model training, and prediction for major tech stocks.

## Quick Start

```bash
# Download stock data
python src/data_collection.py --symbols AAPL --symbols GOOGL --symbols META --symbols TSLA --days 1000

# Train model
python src/train.py --model lstm-small --symbol AAPL --config development

# Make predictions
python src/predict.py --model lstm-small --symbol AAPL --days 30 --plot

# Interactive mode
python src/predict.py --interactive --model lstm-small
```

## Features

### Data Collection & Processing
- Multi-source data collection: Yahoo Finance, Alpha Vantage, custom CSV
- Technical indicators: RSI, MACD, Bollinger Bands, ATR
- Feature engineering: Price, time, lag features, rolling statistics
- Data validation and quality checks
- Multiple formats: CSV, Parquet, HDF5

### LSTM Model Architectures
- BasicLSTM: Standard LSTM with configurable layers
- AttentionLSTM: LSTM with attention mechanism
- CNN-LSTM: Hybrid CNN and LSTM model
- MultiHeadAttentionLSTM: Advanced multi-head attention
- Bidirectional support for all models

### Experiment Tracking & Monitoring
- WandB integration for experiment tracking
- Training progress and metrics logging
- Model checkpointing and early stopping
- System information monitoring

### Configuration System
- Model configs: Pre-defined architectures (tiny, small, medium, large)
- Training configs: Different strategies (test, development, production)
- Data configs: Various feature sets (basic, enhanced, advanced)
- JSON-based configuration

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, but recommended)

### Setup
```bash
# Clone or navigate to the time_series directory
cd time_series

# Install dependencies
pip install -r requirements.txt

# Setup environment
python setup.py

# Copy environment template and add API keys (optional)
cp .env.example .env
# Edit .env to add your WandB API key for experiment tracking
```

## Usage Guide

### 1. Data Collection

```bash
# Download data for specific symbols
python src/data_collection.py --symbols AAPL --symbols GOOGL --symbols META --symbols TSLA --days 1000

# Download with custom date range
python src/data_collection.py --symbols AAPL --start-date 2020-01-01 --end-date 2024-12-31

# Download all available symbols
python src/data_collection.py --all --days 730

# Update existing data
python src/data_collection.py --symbols AAPL --update

# List available symbols
python src/data_collection.py --list-symbols
```

### 2. Model Training

```bash
# Basic training
python src/train.py --model lstm-small --symbol AAPL --config development

# Train with enhanced features
python src/train.py --model lstm-medium --symbol GOOGL --config development --data-config enhanced

# Multi-symbol training
python src/train.py --model lstm-large --symbols AAPL --symbols GOOGL --symbols META --symbols TSLA --config production

# Resume from checkpoint
python src/train.py --resume ./models/lstm_small_AAPL/checkpoint-100

# List available options
python src/train.py --list-models
python src/train.py --list-configs
```

### 3. Prediction & Evaluation

```bash
# Simple prediction
python src/predict.py --model lstm-small --symbol AAPL --days 30

# Prediction with visualization
python src/predict.py --model lstm-small --symbol AAPL --days 30 --plot

# Evaluate model performance
python src/predict.py --model lstm-small --symbol AAPL --evaluate

# Create animated prediction visualization
python src/predict.py --model lstm-large --symbol AAPL --evaluate --animate --start-index 50 --fps 12 --config production --model-dir .

# Batch prediction for multiple symbols
python src/predict.py --model lstm-large --symbols AAPL --symbols GOOGL --symbols META --symbols TSLA --days 14 --config production --model-dir .

# Interactive mode
python src/predict.py --interactive --model lstm-small
```

### 4. Animated Visualizations

The pipeline now supports animated visualizations that show how predictions compare to actual prices over time:

```bash
# Basic animation (starts from beginning, 10 FPS)
python src/predict.py --model lstm-large --symbol AAPL --evaluate --animate --config production --model-dir .

# Custom animation settings
python src/predict.py --model lstm-large --symbol AAPL --evaluate --animate --start-index 100 --fps 15 --config production --model-dir .

# Animation in interactive mode (option 4)
python src/predict.py --interactive --model lstm-large --config production --model-dir .
```

### 5. Jupyter Notebooks

```bash
# Start Jupyter and open the exploration notebook
cd notebooks
jupyter notebook 01_stock_data_exploration.ipynb
```

## Architecture Overview

### Model Architectures

| Model | Layers | Hidden Size | Parameters | Use Case |
|-------|--------|-------------|------------|----------|
| `lstm-tiny` | 1 | 32 | ~5K | Quick testing, prototyping |
| `lstm-small` | 2 | 64 | ~35K | Basic prediction, experimentation |
| `lstm-medium` | 3 | 128 | ~200K | Balanced performance |
| `lstm-large` | 4 | 256 | ~1M | Production deployment |
| `lstm-attention` | 2+Attn | 128 | ~250K | Complex pattern recognition |

### Data Flow Pipeline

```
Raw Stock Data → Feature Engineering → Technical Indicators → Normalization → Sequence Generation → LSTM Training → Prediction
```

1. **Data Collection**: Download OHLCV data from financial APIs
2. **Feature Engineering**: Add technical indicators, time features, lag features
3. **Preprocessing**: Normalize data, create sequences, split datasets
4. **Model Training**: Train LSTM with configurable architecture
5. **Evaluation**: Assess performance with multiple metrics
6. **Prediction**: Generate future price predictions

## Supported Stock Symbols

- **AAPL**: Apple Inc.
- **GOOGL**: Alphabet Inc. (Google)
- **META**: Meta Platforms Inc. (Facebook)
- **TSLA**: Tesla Inc.
- **NVDA**: NVIDIA Corporation
- **MSFT**: Microsoft Corporation
- **AMZN**: Amazon.com Inc.
- **NFLX**: Netflix Inc.
- And more...

## Configuration Options

### Model Configurations
```python
# Example: Custom model configuration
model_config = ModelConfig(
    name="custom-lstm",
    model_type="lstm",
    input_size=10,
    hidden_size=128,
    num_layers=3,
    dropout=0.2,
    bidirectional=True
)
```

### Training Configurations
- **test**: 5 epochs, quick validation
- **development**: 50 epochs, moderate training
- **production**: 200 epochs, full training with advanced scheduling

### Data Configurations
- **basic**: OHLCV features only
- **enhanced**: OHLCV + 6 technical indicators
- **advanced**: OHLCV + 18 technical indicators + time features

## 📈 Performance Metrics

The pipeline evaluates models using multiple metrics:

- **Mean Absolute Error (MAE)**: Average absolute prediction error
- **Root Mean Square Error (RMSE)**: Penalizes larger errors more heavily
- **Mean Absolute Percentage Error (MAPE)**: Percentage-based error metric
- **Directional Accuracy**: Percentage of correct price direction predictions
- **Sharpe Ratio**: Risk-adjusted return for trading strategies


### Experiment Tracking
```bash
# Enable WandB tracking
export WANDB_API_KEY=your_api_key
python src/train.py --model lstm-medium --symbol AAPL --wandb
```

### Custom Model Development
```python
from src.models.lstm_models import BasicLSTM
from src.config.settings import ModelConfig

# Create custom model
config = ModelConfig(
    name="my-model",
    model_type="lstm",
    input_size=15,
    hidden_size=256,
    num_layers=4,
    dropout=0.3
)

model = create_model(config)
```

### Feature Engineering
```python
from src.data.preprocessing import FeatureEngineer

engineer = FeatureEngineer(data_config)
processed_data = engineer.process_data(raw_data)
```

## 📁 Project Structure

```
time_series/
├── src/                        # Source code
│   ├── data_collection.py      # Data download script
│   ├── train.py               # Training pipeline
│   ├── predict.py             # Prediction pipeline
│   ├── config/                # Configuration management
│   │   └── settings.py        # Model, training, and data configs
│   ├── models/                # LSTM architectures
│   │   └── lstm_models.py     # Model implementations
│   ├── data/                  # Data processing
│   │   ├── preprocessing.py   # Feature engineering
│   │   └── loaders.py         # Data loading utilities
│   └── utils/                 # Utility functions
│       └── logging.py         # Logging configuration
├── data/                      # Data storage
│   ├── raw/                   # Downloaded stock data
│   ├── processed/             # Preprocessed data
│   └── features/              # Feature engineered data
├── models/                    # Trained model checkpoints
├── results/                   # Prediction results and plots
├── notebooks/                 # Jupyter notebooks
│   └── 01_stock_data_exploration.ipynb
├── logs/                      # Training and execution logs
├── requirements.txt           # Python dependencies
├── setup.py                   # Setup and demo script
├── .env.example              # Environment template
└── README.md                 # This file
```

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python src/train.py --model lstm-small --symbol AAPL --config development
   # Edit training config to use smaller batch_size
   ```

2. **Missing Data**
   ```bash
   # Re-download data
   python src/data_collection.py --symbols AAPL --update
   ```

3. **Import Errors**
   ```bash
   # Ensure you're in the time_series directory
   cd time_series
   python src/train.py --help
   ```

### Performance Optimization

1. **Use GPU acceleration** (install CUDA-compatible PyTorch)
2. **Increase batch size** for faster training on powerful hardware
3. **Use data parallelism** for multi-GPU setups
4. **Enable mixed precision** training for memory efficiency

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow the existing code style and add comprehensive logging
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

### Code Style Guidelines
- Use type hints for function parameters and return values
- Add comprehensive docstrings for all functions and classes
- Follow PEP 8 style guidelines
- Include error handling and logging

## 📄 License

This project is part of the DNN_tmp research workspace and is intended for educational and research purposes.

## 🙏 Acknowledgments

- **Yahoo Finance** for providing free stock data API
- **TA-Lib** for technical analysis indicators
- **PyTorch** for the deep learning framework
- **Weights & Biases** for experiment tracking
- **Plotly** for interactive visualizations

## 📞 Support

For questions, issues, or contributions:

1. Check the [troubleshooting section](#troubleshooting)
2. Review existing issues in the repository
3. Create a new issue with detailed information
4. Join discussions in the project repository

---

**Happy Predicting! 📈🚀**

*Remember: Past performance does not guarantee future results. This tool is for educational and research purposes only and should not be used as the sole basis for investment decisions.*
