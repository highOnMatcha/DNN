# Pokemon Sprite Generation

Image-to-image translation pipeline that converts Ken Sugimori Pokemon artwork to Black/White sprite style using pix2pix and U-Net architectures.

## Dataset

The dataset consists of 898 paired Pokemon artwork-sprite samples and 127+ Pokemon with missing Black/White sprites. Artwork images are 475x475px Ken Sugimori official art, sprites are 96x96px from Pokemon Black/White games.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download and prepare dataset
jupyter notebook notebooks/01_data_analysis.ipynb
```

## Training

```bash
python src/train.py --model pix2pix-pretrained --config anti_overfitting --backbone resnet50

# Quick test (5 epochs)
python src/train.py --model pix2pix-small --config test

# Development training (50 epochs)
python src/train.py --model pix2pix-medium --config development

# Production training (200 epochs)
python src/train.py --model pix2pix-large --config production

# OPTIMAL: Best quality pixel art generation (250 epochs)
python src/train.py --model pix2pix-pixel-art --config pixel_art_optimal --augmentation production

# Anti-overfitting configuration (regularized model + early stopping)
python src/train.py --model pix2pix-regularized --config anti_overfitting --augmentation anti_overfitting

# Pretrained backbone options
python src/train.py --model pix2pix-pretrained --config development --backbone resnet50     # Best balance
python src/train.py --model pix2pix-pretrained --config development --backbone resnet34     # Lighter
python src/train.py --model pix2pix-pretrained --config development --backbone efficientnet_b0  # Most efficient
```

## Augmentation

Pixel art-optimized augmentation levels (configurable in `src/config/model_configs.json`):

- **light**: Minimal augmentation - horizontal flip (50%), light color jitter, minimal noise
- **standard**: Balanced augmentation - flip + rotation (±5°), moderate color jitter, small cutouts  
- **production**: Aggressive augmentation - flip (60%) + rotation (±8°), strategic cutouts
- **anti_overfitting**: Maximum augmentation - flip (70%) + rotation (±12°), elastic transforms, perspective changes
- **none**: No augmentation for testing

Blur is disabled for all levels to preserve pixel art sharpness.

## Model Options

- **pix2pix-pretrained**: Uses pretrained ResNet/EfficientNet backbone (15M trainable params) - **Best for small datasets**
- **pix2pix-regularized**: Anti-overfitting design with spectral norm and label smoothing (25M params)
- **pix2pix-small/medium/large**: Standard Pix2Pix models (8M/25M/54M params)
- **pix2pix-pixel-art**: Optimized for pixel art with attention and enhanced losses (65M params)

## Generation

```bash
# Generate sprites for missing Pokemon
python src/generate.py --model models/pix2pix_medium/checkpoints/best_model.pth \
                      --input data/pokemon_complete/artwork/ \
                      --output results/
```

## Testing

```bash
python tests/test_pipeline.py
python tests/test_integration.py
```

## Research References

- Isola et al. "Image-to-Image Translation with Conditional Adversarial Networks" (2017)
- Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)

## License

Educational and research use only. Pokemon artwork and sprites are property of Nintendo/Game Freak/Creatures Inc.