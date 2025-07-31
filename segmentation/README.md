# Semantic Segmentation Project

A comprehensive deep learning project for semantic segmentation using U-Net architecture with PyTorch, featuring WandB integration for experiment tracking and comprehensive evaluation metrics.

## Project Structure

```
segmentation/
├── src/
│   ├── config/
│   │   ├── settings.py           # Configuration management
│   │   └── model_configs.json    # Model configurations
│   ├── core/
│   │   ├── logging_config.py     # Logging setup
│   │   └── trainer.py            # Training pipeline
│   ├── data/
│   │   └── loaders.py            # Dataset loading and preprocessing
│   ├── models/
│   │   └── unet.py               # U-Net model implementation
│   ├── utils/
│   │   └── metrics.py            # Evaluation metrics
│   ├── train.py                  # Main training script
│   └── predict.py                # Inference script
├── models/                       # Saved model checkpoints
├── data/                         # Dataset storage
├── results/                      # Output visualizations and results
├── logs/                         # Training logs
├── notebooks/                    # Jupyter notebooks for experimentation
└── requirements.txt              # Python dependencies
```

## Features

- **Custom U-Net Implementation**: Built from scratch with skip connections
- **Pre-trained Backbones**: Support for ResNet, EfficientNet, and other encoders
- **Comprehensive Metrics**: IoU, Dice coefficient, pixel accuracy, and per-class metrics
- **Data Augmentation**: Advanced augmentation pipeline using Albumentations
- **Experiment Tracking**: WandB integration for monitoring and visualization
- **Mixed Precision Training**: Automatic mixed precision for faster training
- **Flexible Configuration**: JSON-based configuration management
- **Visualization Tools**: Prediction visualization and confusion matrix plotting

## Dataset

The project is designed to work with the **PASCAL VOC 2012** segmentation dataset, which contains:
- 21 classes (20 object classes + background)
- 1,464 training images
- 1,449 validation images
- Classes: background, aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor

## Quick Reference

### Common Training Commands

```bash
# Quick test (5 epochs, small images)
python src/train.py --config test

# Balanced training (ResNet50 backbone)
python src/train.py --config default

# Best accuracy (EfficientNet-B4 backbone)
python src/train.py --config production

# Development/debugging
python src/train.py --config development

# Resume training
python src/train.py --config production --resume models/checkpoint.pt

# Evaluation only
python src/train.py --evaluate-only --resume models/best_model.pt
```

### Model Selection Quick Guide

| Need | Configuration | Command |
|------|---------------|---------|
| **Fast testing** | `test` | `python src/train.py --config test` |
| **Best accuracy** | `production` | `python src/train.py --config production` |
| **Balanced performance** | `default` | `python src/train.py --config default` |
| **Development/debugging** | `development` | `python src/train.py --config development` |

### Installation

1. Clone the repository:
```bash
cd /home/khalil/DNN/segmentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up WandB (optional):
```bash
wandb login
```

4. Create environment file for WandB:
```bash
cp set_db_env.example.sh set_db_env.sh
# Edit set_db_env.sh with your WandB credentials
source set_db_env.sh
```

## Quick Start

### Available Models and Configurations

The project supports multiple model architectures and training configurations:

#### Model Architectures
- **U-Net with ResNet50**: Default balanced model (`unet_resnet50`)
- **U-Net with EfficientNet-B4**: High accuracy model (`unet_efficientnet_b4`) 
- **DeepLabV3 with ResNet50**: State-of-the-art segmentation (`deeplabv3_resnet50`)
- **Custom U-Net**: From-scratch implementation (`unet_from_scratch`)

#### Training Configurations
- **`test`**: Fast testing (2 epochs, 128x128 images, batch_size=2)
- **`development`**: Medium training (20 epochs, 256x256 images, batch_size=4)
- **`production`**: Full training (100 epochs, 512x512 images, batch_size=16, EfficientNet-B4)
- **`default`**: Balanced settings (100 epochs, 256x256 images, batch_size=8, ResNet50)

### Training

1. **Basic training with default U-Net ResNet50 model:**
```bash
cd src
python train.py --config default
```

2. **Training specific models with different configurations:**
```bash
# Quick test with small U-Net model
python train.py --config test

# Development training with medium settings
python train.py --config development

# Production training with EfficientNet-B4 backbone (best accuracy)
python train.py --config production
```

3. **Custom model training by modifying configuration:**
```bash
# Edit src/config/settings.py to change model architecture:
# For DeepLabV3: architecture="deeplabv3"
# For different encoder: encoder_name="efficientnet-b0"
python train.py --config default

4. **Resume training from checkpoint:**
```bash
python train.py --config development --resume ../models/unet_dev/checkpoint-10.pt
```

5. **Training without WandB logging:**
```bash
python train.py --config default --no-wandb
```

### Model Selection Guide

Choose your model and configuration based on your requirements:

| Use Case | Configuration | Model | Training Time | Accuracy | Memory Usage |
|----------|---------------|-------|---------------|----------|--------------|
| Quick Testing | `test` | UNet + ResNet50 | ~5 min | Basic | Low |
| Development | `development` | UNet + ResNet50 | ~2 hours | Good | Medium |
| Best Accuracy | `production` | UNet + EfficientNet-B4 | ~8 hours | Highest | High |
| Balanced | `default` | UNet + ResNet50 | ~4 hours | Good | Medium |

### Customizing Model Architecture

To train with a specific model architecture, modify `src/config/settings.py`:

```python
# Example: Training DeepLabV3 with EfficientNet-B0
DEFAULT_CONFIG = SegmentationConfig(
    name="deeplabv3_efficientnet_b0",
    architecture="deeplabv3",           # Choose: unet, deeplabv3, unet_custom
    encoder_name="efficientnet-b0",     # Choose: resnet50, efficientnet-b4, etc.
    encoder_weights="imagenet",         # Use ImageNet pre-trained weights
    batch_size=8,
    num_epochs=50,
    # ... other parameters
)
```

### Monitoring Your Training

When you start training, the script will clearly display which model is being used:

```
================================================================================
SEGMENTATION TRAINING PIPELINE
================================================================================
Configuration: production
Model Name: unet_production
Architecture: unet
Encoder: efficientnet-b4
Pre-trained Weights: imagenet
Number of Classes: 21
Input Size: (512, 512)
Batch Size: 16
Epochs: 100
Learning Rate: 5e-05
Optimizer: adamw
WandB Logging: Enabled
Data Augmentation: Enabled
Mixed Precision: Enabled
================================================================================
```

This information helps you verify:
- **Model Name**: Unique identifier for this training run
- **Architecture**: The segmentation model being used (unet, deeplabv3, etc.)
- **Encoder**: The backbone network (resnet50, efficientnet-b4, etc.)
- **Pre-trained Weights**: Whether using ImageNet pre-trained weights
- **Training Settings**: Batch size, learning rate, number of epochs

#### Supported Encoder Backbones

Choose the encoder backbone based on your accuracy vs speed requirements:

**Fast Training (ResNet family)**
- `resnet18`: Fastest, lowest accuracy
- `resnet34`: Fast, basic accuracy  
- `resnet50`: **Recommended balance**
- `resnet101`: Slower, higher accuracy
- `resnet152`: Slowest, highest accuracy

**Best Accuracy (EfficientNet family)**
- `efficientnet-b0`: Efficient baseline
- `efficientnet-b1`: Better accuracy
- `efficientnet-b2`: Good balance
- `efficientnet-b3`: High accuracy
- `efficientnet-b4`: **Recommended for production**
- `efficientnet-b5-b7`: Highest accuracy, very slow

**Other Options**
- **DenseNet**: `densenet121`, `densenet169`, `densenet201`
- **VGG**: `vgg11`, `vgg13`, `vgg16`, `vgg19`
- **Many more**: Available through the timm library

### Inference

1. **Predict on a single image:**
```bash
python predict.py --model ../models/unet_production/best_model.pt --input /path/to/image.jpg --visualize
```

2. **Batch prediction on multiple images:**
```bash
python predict.py --model ../models/unet_production/best_model.pt --input /path/to/images/ --output ../results --save-masks
```

3. **Evaluation only (no new training):**
```bash
python train.py --config production --resume ../models/unet_production/best_model.pt --evaluate-only
```

## Configuration

The project uses a flexible configuration system with pre-defined model and training configurations.

### Available Training Configurations

| Configuration | Model Name | Architecture | Encoder | Image Size | Batch Size | Epochs | Use Case |
|---------------|------------|--------------|---------|------------|------------|--------|----------|
| `test` | unet_test | U-Net | ResNet50 | 128×128 | 2 | 5 | Quick testing |
| `development` | unet_dev | U-Net | ResNet50 | 256×256 | 4 | 20 | Development |
| `production` | unet_production | U-Net | EfficientNet-B4 | 512×512 | 16 | 100 | Best accuracy |
| `default` | unet_resnet50 | U-Net | ResNet50 | 256×256 | 8 | 100 | Balanced |

### Model Architecture Options

The following model architectures are supported:

1. **U-Net** (`architecture="unet"`): 
   - Classic encoder-decoder with skip connections
   - Excellent for medical and general segmentation
   - Supports any encoder backbone

2. **DeepLabV3** (`architecture="deeplabv3"`):
   - State-of-the-art semantic segmentation
   - Atrous convolutions for multi-scale features
   - Higher accuracy, more computational cost

3. **Custom U-Net** (`architecture="unet_custom"`):
   - Built from scratch implementation
   - Educational and research purposes
   - No pre-trained encoders

### Training a Specific Model

To train a specific model architecture:

```bash
# Method 1: Use existing configurations
python src/train.py --config production    # UNet + EfficientNet-B4
python src/train.py --config default       # UNet + ResNet50

# Method 2: Modify configuration in src/config/settings.py
# Edit the DEFAULT_CONFIG or create a new configuration
```

### Creating Custom Configurations

Add custom configurations to `src/config/settings.py`:

```python
CUSTOM_CONFIG = SegmentationConfig(
    name="deeplabv3_resnet101_large",
    architecture="deeplabv3",
    encoder_name="resnet101",
    encoder_weights="imagenet",
    batch_size=4,
    num_epochs=80,
    image_size=(384, 384),
    learning_rate=5e-5,
    use_wandb=True,
    mixed_precision=True
)
```

Then add it to the configs dictionary in `get_config()` function.

Key configuration parameters:
- `architecture`: Model architecture (unet, deeplabv3, etc.)
- `encoder_name`: Backbone encoder (resnet50, efficientnet-b4, etc.)
- `batch_size`: Training batch size
- `learning_rate`: Initial learning rate
- `num_epochs`: Number of training epochs
- `image_size`: Input image size (height, width)
- `use_augmentation`: Enable/disable data augmentation

## Model Architectures

### Available Models

The project supports multiple state-of-the-art segmentation architectures:

| Architecture | Description | Best Use Case | Accuracy | Speed |
|-------------|-------------|---------------|----------|-------|
| **U-Net** | Classic encoder-decoder with skip connections | General segmentation, medical imaging | High | Fast |
| **DeepLabV3** | Atrous convolutions with ASPP module | Complex scenes, fine details | Highest | Slower |
| **Custom U-Net** | From-scratch implementation | Research, learning | Medium | Fast |

### Model Configurations in Detail

1. **U-Net (Recommended)**:
   ```python
   architecture="unet"
   encoder_name="resnet50"          # Or efficientnet-b4 for best accuracy
   encoder_weights="imagenet"       # Pre-trained weights
   ```
   - **Pros**: Fast training, good accuracy, memory efficient
   - **Cons**: May struggle with very complex scenes
   - **Best for**: Most segmentation tasks, beginners

2. **DeepLabV3 (State-of-the-art)**:
   ```python
   architecture="deeplabv3"
   encoder_name="resnet50"          # Or resnet101 for better accuracy
   encoder_weights="imagenet"       # Pre-trained weights
   ```
   - **Pros**: Highest accuracy, handles multiple scales well
   - **Cons**: Slower training, more memory usage
   - **Best for**: Production systems, complex segmentation

3. **Custom U-Net (Educational)**:
   ```python
   architecture="unet_custom"
   encoder_name="custom"            # No pre-trained encoder
   encoder_weights=None             # Train from scratch
   ```
   - **Pros**: Full control, educational value
   - **Cons**: Longer training time, requires more data
   - **Best for**: Research, understanding architecture

### Supported Encoders

- ResNet family (resnet18, resnet34, resnet50, resnet101)
- EfficientNet family (efficientnet-b0 to efficientnet-b7)
- DenseNet family
- And many more from timm library

## Metrics and Evaluation

The project includes comprehensive evaluation metrics:

- **Intersection over Union (IoU)**: Standard segmentation metric
- **Dice Coefficient**: Harmonic mean of precision and recall
- **Pixel Accuracy**: Overall pixel classification accuracy
- **Per-class Metrics**: Individual class performance
- **Confusion Matrix**: Detailed classification analysis

## Loss Functions

Multiple loss functions are supported:
- **Cross-Entropy Loss**: Standard classification loss
- **Dice Loss**: Optimizes Dice coefficient directly
- **Focal Loss**: Addresses class imbalance
- **Combined Loss**: Weighted combination of CE and Dice loss

## Visualization

The project provides rich visualization capabilities:
- Training curves and metrics plots
- Prediction overlays on original images
- Confusion matrix heatmaps
- Per-class performance charts
- Real-time training monitoring via WandB

## Logging and Monitoring

### Local Logging
- Structured logging with multiple levels
- Automatic log rotation
- JSON format support for analysis
- Training progress tracking

### WandB Integration
- Real-time metric tracking
- Model architecture visualization
- Hyperparameter logging
- Experiment comparison
- Artifact management

## Performance Optimization

- **Mixed Precision Training**: Faster training with reduced memory usage
- **Data Loading**: Multi-worker data loading with pin memory
- **GPU Utilization**: Automatic device selection and memory optimization
- **Checkpointing**: Regular model checkpointing for recovery

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in configuration
   - Use gradient accumulation
   - Enable mixed precision training

2. **Dataset Download Issues**:
   - Ensure internet connection
   - Check disk space
   - Verify dataset URLs

3. **WandB Connection Issues**:
   - Check API key setup
   - Use `--no-wandb` flag to disable
   - Verify network connectivity

### Performance Tips

1. **Faster Training**:
   - Use larger batch sizes if memory allows
   - Enable mixed precision training
   - Use multiple GPU workers for data loading

2. **Better Accuracy**:
   - Increase image resolution
   - Use stronger data augmentation
   - Try different loss function combinations
   - Experiment with learning rate schedules

## Development

### Adding New Models

1. Implement model in `src/models/`
2. Add configuration in `src/config/model_configs.json`
3. Update factory function in `src/models/unet.py`

### Adding New Datasets

1. Create dataset class in `src/data/loaders.py`
2. Add dataset configuration in `src/config/settings.py`
3. Update dataset factory function

### Custom Metrics

1. Implement metric in `src/utils/metrics.py`
2. Add to trainer evaluation loop
3. Include in logging and visualization

## Examples

### Training Examples

```bash
# Quick test run with small model
python src/train.py --config test --log-level DEBUG

# Production training with EfficientNet-B4 (best accuracy)
python src/train.py --config production

# Resume interrupted training
python src/train.py --config production --resume models/unet_production/checkpoint-50.pt

# Train DeepLabV3 model (modify config first)
# Edit src/config/settings.py: architecture="deeplabv3"
python src/train.py --config default

# Training without WandB logging
python src/train.py --config development --no-wandb
```

### Inference Examples

```bash
# Single image with visualization
python src/predict.py --model models/best_model.pt --input image.jpg --visualize

# Batch processing
python src/predict.py --model models/best_model.pt --input images/ --output results/ --save-masks
```

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{segmentation_bootcamp_2025,
  title={Semantic Segmentation Bootcamp Project},
  author={Deep Neural Networks Course},
  year={2025},
  howpublished={\url{https://github.com/your-repo/segmentation-bootcamp}}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- U-Net architecture based on the original paper by Ronneberger et al.
- PASCAL VOC dataset from the Visual Object Classes Challenge
- PyTorch and segmentation-models-pytorch communities
- Weights & Biases for experiment tracking platform
