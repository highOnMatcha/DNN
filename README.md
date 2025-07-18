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

## Quick Setup

### 1. Install shared dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Choose your project and install specific requirements
```bash
# For LLM project
cd llm && pip install -r requirements.txt

# For segmentation project  
cd segmentation && pip install -r requirements.txt

# For time series project
cd time_series && pip install -r requirements.txt
```

## Shared Infrastructure

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