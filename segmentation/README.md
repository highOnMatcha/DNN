# Semantic Segmentation Project

A comprehensive deep learning project for semantic segmentation using U-Net architecture with PyTorch, featuring WandB integration for experiment tracking and comprehensive evaluation metrics.

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