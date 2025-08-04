# Pokemon Sprite Generation

Image-to-image translation for converting Pokemon artwork to Black/White sprite style.

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install requirements
pip install -r requirements.txt
```

## Training

The training script automatically downloads and prepares the dataset on first run.

```bash
# Quick test (5 epochs)
python src/train.py --model lightweight-baseline --config test

# Recommended configuration for development
python src/train.py --model sprite-optimized --config development

# Production training
python src/train.py --model transformer-enhanced --config production
```

### Available Models

- `lightweight-baseline` - Fast training with minimal parameters for quick experimentation
- `sprite-optimized` - State-of-the-art configuration optimized specifically for pixel art sprite generation
- `transformer-enhanced` - Advanced transformer-enhanced architecture for complex artwork-to-sprite mappings

### Configuration Options

- `test` - 5 epochs, minimal augmentation, quick debugging configuration
- `development` - 50 epochs, standard augmentation, development configuration for experimentation
- `production` - 200 epochs, aggressive augmentation, production training with progressive augmentation and regularization

## Generation

Generate sprites from artwork using trained models:

```bash
python src/generate.py --model models/best_model.pth \
                      --input data/pokemon_complete/artwork/ \
                      --output results/
```

## Testing

```bash
python -m pytest tests/
```

## Dataset

The system automatically downloads 898 Pokemon artwork-sprite pairs from PokeAPI on first training run. No manual data preparation required.

## License

Educational and research use only. Pokemon content is property of Nintendo/Game Freak/Creatures Inc.