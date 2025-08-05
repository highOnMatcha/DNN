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
python src/train.py --model sprite-production  --config production --augmentation strong --wandb
```

## Generation

Generate sprites from artwork using trained models:

```bash
python src/generate.py --model models/best_model.pth \
                      --input data/pokemon_complete/artwork/ \
                      --output results/
```

## Testing

```bash
make test
```

## Results

Unfortunately, the results are not very good yet. The model struggles to yield satisfactory sprite outputs. The outputs are often blurry and do not capture the details of the original artwork well. Some positive aspects are that the overall shape and color palette of the Pokemon are retained, but finer details are lost. This is mainly due to a lack of training data and the complexity of the task. The task would have been much easier if the sprites were more similar to the artwork, but a lot of time the orignal artwork will be the pokemon looking in a different direction, or in a different pose, mouth open or closed, etc.

Yet we still publish one of the best results here:

## Dataset

The system automatically downloads 898 Pokemon artwork-sprite pairs from PokeAPI on first training run. No manual data preparation required.

## License

Educational and research use only. Pokemon content is property of Nintendo/Game Freak/Creatures Inc.