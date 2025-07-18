# Deep Neural Network Projects

Multi-project repository for ML research and experimentation.

## Projects

### LLM (Language Models)
- Dialog model training and comparison
- Custom vs pre-trained architectures
- WandB experiment tracking
- [llm/](./llm/)

### Segmentation 
- Image segmentation models
- [segmentation/](./segmentation/)

### Time Series
- Time series analysis and forecasting
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

### Development Tools
- Jupyter notebooks for interactive development
- Black for code formatting
- Pytest for testing
- Shared database utilities

## Project Structure
```
dnn/
├── requirements.txt          # Shared dependencies
├── set_db_env.sh            # Database configuration  
├── llm/                     # Language model project
│   ├── requirements.txt     # LLM-specific deps
│   ├── notebooks/           # Jupyter notebooks
│   └── src/                 # Source code
├── segmentation/            # Image segmentation
└── time_series/             # Time series analysis
```

## Getting Started

1. Choose a project from the list above
2. Install dependencies following the setup instructions
3. Read the project README in each project folder
4. Start with notebooks for interactive development

Each project is self-contained but shares common infrastructure like database access and development tools.