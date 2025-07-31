# Pokemon Sprite Generation Training Pipeline

A comprehensive machine learning pipeline for training image-to-image translation models to convert Ken Sugimori official Pokemon artwork to Black/White sprite style.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare dataset (run notebook)
jupyter notebook notebooks/01_data_analysis.ipynb

# 3. Maximum performance training with all features
python src/train.py --model pix2pix-large --config production
```

**What this does:**
- ✅ Trains on all 898 Pokemon pairs (200 epochs)
- ✅ Generates sprites for ALL 127+ missing Pokemon
- ✅ Logs everything to WandB with visual samples
- ✅ Saves best model checkpoints automatically

## Features

- **Multiple Architectures**: Support for U-Net, Pix2Pix, CycleGAN, and DDPM models
- **Comprehensive Logging**: Structured logging with JSON output and WandB integration  
- **Flexible Configuration**: JSON-based model and training configurations
- **Professional Pipeline**: Following best practices from production ML systems
- **Experiment Tracking**: Full WandB integration with metrics, samples, and model artifacts
- **Modular Design**: Clean separation of concerns with reusable components
- **Automatic Sprite Generation**: Post-training generation of sprites for Pokemon missing Black/White sprites
- **Missing Pokemon Detection**: Automatically identifies and generates sprites for 127+ Pokemon without official BW sprites

## Project Structure

```
pokemon_sprites/
├── data/                          # Dataset storage
│   ├── pokemon_complete/          # Raw dataset
│   ├── training_data/             # Processed training data
│   └── generated_sprites/         # AI-generated sprites for missing Pokemon
├── logs/                          # Training logs
├── models/                        # Saved model checkpoints
├── notebooks/                     # Data analysis and exploration
│   └── 01_data_analysis.ipynb     # Dataset preparation notebook
├── src/                           # Source code
│   ├── config/                    # Configuration files
│   │   ├── model_configs.json     # Model architecture definitions
│   │   └── settings.py            # Configuration management
│   ├── core/                      # Core training components
│   │   ├── logging_config.py      # Logging setup
│   │   ├── models.py              # Model architectures
│   │   └── trainer.py             # Training orchestration
│   ├── data/                      # Data handling
│   │   └── loaders.py             # Dataset utilities
│   ├── generate.py                # Inference script
│   └── train.py                   # Main training script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Available Models

### U-Net Models
- **unet-small**: 3M parameters, lightweight and fast
- **unet-medium**: 12M parameters, good quality for most use cases  
- **unet-attention**: 15M parameters, enhanced with attention mechanism

### Pix2Pix Models
- **pix2pix-small**: 8M parameters, fast training for experimentation
- **pix2pix-medium**: 25M parameters, good balance of quality and speed
- **pix2pix-large**: 54M parameters, high quality results

### CycleGAN Models  
- **cyclegan-small**: 16M parameters, unpaired training capability
- **cyclegan-medium**: 50M parameters, better quality unpaired translation

### Diffusion Models
- **ddpm-small**: 10M parameters, denoising diffusion for high quality
- **ddpm-medium**: 25M parameters, better quality diffusion model

## Training Configurations

- **test**: 5 epochs, small batch size, quick validation
- **development**: 50 epochs, medium batch size, for experimentation  
- **production**: 200 epochs, large batch size, full training

## Getting Started

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment variables (optional)
cp .env.example .env
# Edit .env with your WANDB_API_KEY
```

### 2. Prepare Dataset

Run the data preparation notebook to download and process the Pokemon dataset:

```bash
jupyter notebook notebooks/01_data_analysis.ipynb
```

This will:
- Download Pokemon artwork and Black/White sprites
- Create training/validation splits
- Generate dataset statistics and visualizations

### 3. Train a Model

```bash
# Quick test training (5 epochs, 10 generated sprites)
python src/train.py --model pix2pix-small --config test

# Development training (50 epochs, 50 generated sprites)
python src/train.py --model pix2pix-medium --config development

# 🚀 MAXIMUM PERFORMANCE: Full production training with all data and all missing sprites
python src/train.py --model pix2pix-large --config production

# Custom training with specific sample limits
python src/train.py --model pix2pix-medium --config development --max-samples 1000

# Disable WandB logging
python src/train.py --model unet-medium --config development --no-wandb

# Control sprite generation
python src/train.py --model pix2pix-small --config test --max-generate 20  # Generate max 20 sprites
python src/train.py --model pix2pix-medium --config development --skip-generation  # No sprite generation
```

### 4. Post-Training Sprite Generation

After training, the pipeline automatically:
- 🔍 **Detects Missing Sprites**: Finds Pokemon artwork without corresponding Black/White sprites (127+ Pokemon)
- 🎨 **Generates New Sprites**: Uses the trained model to create sprites for missing Pokemon
- 📊 **Logs to WandB**: Uploads generated sprites with captions for easy viewing
- 💾 **Saves Locally**: Stores generated sprites in `data/generated_sprites/`

**Generation Limits by Configuration:**
- **Test**: 10 missing sprites generated
- **Development**: 50 missing sprites generated  
- **Production**: ALL missing sprites generated (127+)
- **Custom**: Use `--max-generate N` to specify exact number

### 5. Manual Sprite Generation

```bash
# Generate from single artwork
python src/generate.py --model models/pix2pix_medium/checkpoints/best_model.pth \\
                      --input artwork/pikachu.png \\
                      --output results/

# Generate from directory with comparisons
python src/generate.py --model models/pix2pix_medium/checkpoints/best_model.pth \\
                      --input artwork_folder/ \\
                      --output results/ \\
                      --comparison

# Use specific device
python src/generate.py --model models/pix2pix_medium/checkpoints/best_model.pth \\
                      --input artwork.png \\
                      --output results/ \\
                      --device cuda
```

## Command Line Options

### Training Script Options

```bash
python src/train.py [OPTIONS]

Required:
  --model MODEL              Model configuration name (pix2pix-small, unet-medium, etc.)

Optional:
  --config {test,development,production}
                             Training configuration type (default: development)
  --max-samples N            Maximum training samples for testing (default: all)
  --no-wandb                 Disable WandB logging
  --log-level {DEBUG,INFO,WARNING,ERROR}
                             Logging verbosity (default: INFO)
  --skip-generation          Skip post-training sprite generation
  --max-generate N           Maximum missing sprites to generate (overrides config defaults)
```

### Examples

```bash
# 🏆 RECOMMENDED: Maximum performance training
python src/train.py --model pix2pix-large --config production

# Quick test with custom sprite generation
python src/train.py --model pix2pix-small --config test --max-generate 5

# Development training without sprite generation
python src/train.py --model unet-medium --config development --skip-generation

# Large-scale training with limited samples
python src/train.py --model pix2pix-medium --config production --max-samples 10000
```

## Configuration

### Model Configuration

Models are defined in `src/config/model_configs.json`. Example:

```json
{
  "pix2pix_models": {
    "pix2pix-custom": {
      "name": "pix2pix-custom",
      "description": "Custom Pix2Pix configuration",
      "output_dir": "./models/pix2pix_custom",
      "architecture": "pix2pix",
      "parameters": {
        "generator": {
          "input_channels": 3,
          "output_channels": 3,
          "ngf": 64,
          "n_blocks": 6
        },
        "discriminator": {
          "input_channels": 6,
          "ndf": 64,
          "n_layers": 3
        }
      }
    }
  }
}
```

### Training Configuration

Training parameters can be customized in the same JSON file:

```json
{
  "training_configs": {
    "custom": {
      "epochs": 100,
      "batch_size": 32,
      "learning_rate": 0.0002,
      "eval_frequency": 5,
      "save_frequency": 10
    }
  }
}
```

## Monitoring and Logging

### Local Logs

Training logs are saved in the `logs/` directory with multiple formats:
- `experiment_id.log`: Main training log
- `experiment_id_errors.log`: Error-only log  
- `experiment_id_structured.jsonl`: Structured JSON logs

### WandB Integration

If configured with a WandB API key, the pipeline automatically logs:
- Training and validation metrics
- Generated sample images during training
- **Generated missing sprites** with Pokemon IDs
- Model architecture and hyperparameters
- System information and performance metrics

### Generated Sprites in WandB

The post-training sprite generation feature logs:
- Up to 8 generated sprites per training run
- Clear captions identifying Pokemon IDs
- Easy comparison with original artwork
- Organized under `generated_missing_sprites` section

## Advanced Usage

### Custom Models

Add new model architectures in `src/core/models.py` and update the configuration:

```python
def create_custom_model(config):
    # Your custom architecture implementation
    pass
```

### Custom Loss Functions

Extend the trainer in `src/core/trainer.py`:

```python
def _setup_custom_loss_functions(self):
    # Your custom loss functions
    pass
```

### Distributed Training

The pipeline supports multi-GPU training automatically when multiple GPUs are available.

## Dataset Information

- **Artwork**: Ken Sugimori official artwork (1025 files, ~127MB)
- **Sprites**: Pokemon Black/White sprites (898 files, ~0.7MB)  
- **Matched Pairs**: 898 artwork-sprite pairs for supervised training
- **Missing Sprites**: 127 Pokemon without official Black/White sprites
- **Resolution**: Artwork (475x475), Sprites (96x96)
- **Training Split**: 80% train, 20% validation

### Missing Pokemon Coverage

The pipeline automatically generates sprites for Pokemon that lack official Black/White sprites, including:
- Generation 6+ Pokemon (post-Black/White release)
- Alternative forms and regional variants
- Special event Pokemon
- Mega evolutions and other form variations

## Results

The pipeline produces:
- Trained model checkpoints with best validation performance
- Generated sprite samples during training for monitoring
- **AI-generated sprites for all missing Pokemon** (127+ new sprites)
- Comprehensive training metrics and logs
- Comparison visualizations (artwork vs generated sprite)
- WandB experiment tracking with visual samples

## Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive logging for new features
3. Update configuration files for new models/parameters
4. Include proper error handling and validation
5. Test with the test configuration before full training

## License

This project is for educational and research purposes. Pokemon artwork and sprites are property of Nintendo/Game Freak/Creatures Inc.
