"""
Training script for Alpaca-GPT4 dataset using PyTorch and Transformers.
Handles dataset loading, preprocessing, and fine-tuning of language models.
"""

import argparse
import os
import sys

# Add the parent directory to the path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loaders import get_dataset_manager
from src.core.trainer import DialogTrainer
from src.config.settings import (
    DEFAULT_MODEL_CONFIG, DEFAULT_TRAINING_CONFIG, DEFAULT_DATASET_CONFIG,
    get_production_config, get_test_config, get_model_config, list_available_models
)


def main(model_type="gpt2-small"):
    """Main training pipeline."""
    print("Alpaca-GPT4 PyTorch Training Pipeline")
    print("=" * 40)
    
    # Get model configuration
    model_config = get_model_config(model_type)
    print(f"Using model: {model_config.name}")
    print(f"Output directory: {model_config.output_dir}")
    
    # Initialize dataset manager
    dataset_manager = get_dataset_manager(data_dir=DEFAULT_DATASET_CONFIG.data_dir)
    
    # Show dataset size info
    print("\nDataset Information:")
    dataset_manager.print_size_info()
    
    # Load dataset
    print("\n1. Loading dataset...")
    df = dataset_manager.load_dataset()
    dataset_manager.print_dataset_summary(df)
    
    # Initialize trainer with configurations
    print("\n2. Initializing PyTorch model...")
    trainer = DialogTrainer(model_config=model_config)
    
    # Use test configuration for development
    training_config = get_test_config()
    
    # Prepare dataset
    print("\n3. Preparing dataset...")
    train_dataset, eval_dataset = trainer.prepare_dataset(
        df, 
        train_split=training_config.train_split,
        max_samples=training_config.max_samples
    )
    
    # Train the model
    print("\n4. Starting training...")
    trained_model = trainer.train(
        train_dataset, 
        eval_dataset,
        num_epochs=training_config.num_epochs,
        batch_size=training_config.batch_size,
        learning_rate=training_config.learning_rate
    )
    
    # Test generation
    print("\n5. Testing trained model...")
    test_instructions = [
        "Give three tips for staying healthy.",
        "What are the primary colors?",
        "Explain what machine learning is."
    ]
    
    for instruction in test_instructions:
        print(f"\nInstruction: {instruction}")
        response = trainer.generate_response(instruction)
        print(f"Response: {response}")
    
    print("\nTraining completed successfully!")


def analyze_dataset():
    """Analyze the dataset without training."""
    print("Dataset Analysis Mode")
    print("=" * 20)
    
    dataset_manager = get_dataset_manager(data_dir=DEFAULT_DATASET_CONFIG.data_dir)
    
    # Show dataset size info
    dataset_manager.print_size_info()
    
    # Load dataset
    print("\nLoading dataset...")
    df = dataset_manager.load_dataset()
    dataset_manager.print_dataset_summary(df)
    
    # Show examples of each field type
    print("\nField Examples:")
    print("-" * 30)
    
    print("Instruction example:")
    print(f"'{df['instruction'].iloc[0]}'")
    
    print(f"\nInput example (first non-null):")
    non_null_input = df[df['input'].notna() & (df['input'] != '')]['input'].iloc[0] if len(df[df['input'].notna() & (df['input'] != '')]) > 0 else "N/A"
    print(f"'{non_null_input}'")
    
    print(f"\nOutput example:")
    print(f"'{df['output'].iloc[0][:200]}...'")
    
    print(f"\nFormatted text example:")
    print(f"'{df['text'].iloc[0][:300]}...'")


def train_production(model_type="gpt2-medium"):
    """Run training with production configuration."""
    print("Production Training Mode")
    print("=" * 25)
    
    # Get model configuration
    model_config = get_model_config(model_type)
    print(f"Using model: {model_config.name}")
    
    # Initialize dataset manager
    dataset_manager = get_dataset_manager(data_dir=DEFAULT_DATASET_CONFIG.data_dir)
    
    # Load full dataset
    print("\n1. Loading full dataset...")
    df = dataset_manager.load_dataset()
    dataset_manager.print_dataset_summary(df)
    
    # Initialize trainer
    print("\n2. Initializing model for production training...")
    trainer = DialogTrainer(model_config=model_config)
    
    # Use production configuration
    training_config = get_production_config()
    
    # Prepare full dataset
    print("\n3. Preparing full dataset...")
    train_dataset, eval_dataset = trainer.prepare_dataset(
        df, 
        train_split=training_config.train_split,
        max_samples=training_config.max_samples  # None for full dataset
    )
    
    # Train with production settings
    print("\n4. Starting production training...")
    trained_model = trainer.train(
        train_dataset, 
        eval_dataset,
        num_epochs=training_config.num_epochs,
        batch_size=training_config.batch_size,
        learning_rate=training_config.learning_rate
    )
    
    print("\nProduction training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpaca-GPT4 Training Script")
    parser.add_argument("--analyze-only", action="store_true", 
                       help="Only analyze dataset without training")
    parser.add_argument("--production", action="store_true",
                       help="Run production training with full dataset")
    parser.add_argument("--model", type=str, default="gpt2-small",
                       help="Model type to use (see --list-models for options)")
    parser.add_argument("--list-models", action="store_true",
                       help="List available model configurations")
    parser.add_argument("--no-cuda", action="store_true",
                       help="Disable CUDA even if available")
    
    args = parser.parse_args()
    
    if args.list_models:
        list_available_models()
        sys.exit(0)
    
    if args.no_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    if args.analyze_only:
        analyze_dataset()
    elif args.production:
        train_production(args.model)
    else:
        main(args.model)
