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

## Installation

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

### Training

1. **Basic training with default configuration:**
```bash
cd src
python train.py --config default
```

2. **Training with different configurations:**
```bash
# Test configuration (small model, few epochs)
python train.py --config test

# Development configuration (medium settings)
python train.py --config development

# Production configuration (full training)
python train.py --config production
```

3. **Resume training from checkpoint:**
```bash
python train.py --config development --resume ../models/unet_dev/checkpoint-10.pt
```

4. **Training without WandB logging:**
```bash
python train.py --config default --no-wandb
```

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

The project uses a flexible configuration system. Available configurations:

- **`test`**: Quick testing with small model and few epochs
- **`development`**: Medium-scale training for development
- **`production`**: Full-scale training with best settings
- **`default`**: Balanced configuration for general use

### Custom Configuration

You can modify `src/config/settings.py` to create custom configurations or edit `src/config/model_configs.json` for model-specific settings.

Key configuration parameters:
- `architecture`: Model architecture (unet, deeplabv3, etc.)
- `encoder_name`: Backbone encoder (resnet50, efficientnet-b4, etc.)
- `batch_size`: Training batch size
- `learning_rate`: Initial learning rate
- `num_epochs`: Number of training epochs
- `image_size`: Input image size (height, width)
- `use_augmentation`: Enable/disable data augmentation

## Model Architectures

### Supported Models

1. **U-Net (Custom)**: Custom implementation from scratch
2. **U-Net (Pre-trained)**: Using segmentation-models-pytorch
3. **DeepLabV3**: State-of-the-art segmentation model
4. **DeepLabV3+**: Enhanced version with decoder

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
# Quick test run
python src/train.py --config test --log-level DEBUG

# Production training with full monitoring
python src/train.py --config production

# Resume interrupted training
python src/train.py --config production --resume models/unet_production/checkpoint-50.pt
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
