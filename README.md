# Deep Neural Network Projects

Multi-project repository for ML research and experimentation.

## Projects

### LLM (Language Models)
- Dialog model training 
- Custom and pre-trained architectures
- [llm/](./llm/)

### Segmentation 
- Image segmentation model
- [segmentation/](./segmentation/)

### Time Series
- Time series analysis and stock price forecasting
- [time_series/](./time_series/)

### Pokemon Sprites (NEW)
- Image-to-image translation for converting Pokemon artwork to Black/White sprite style
- [pokemon_sprites/](./pokemon_sprites/)

## Quick Setup

### 1. Install shared dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Choose the project you are interested in and install specific requirements
```bash
cd <project_name> && pip install -r requirements.txt
```

## Shared Infrastructure for llm project (mainly for practice)

### Database Configuration (Optional)
PostgreSQL setup for data storage across projects.

#### Setup:
```bash
# Copy and edit database config
cp set_db_env.example.sh set_db_env.sh
# Edit set_db_env.sh with your database credentials

# Load environment
source set_db_env.sh
```