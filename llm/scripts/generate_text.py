#!/usr/bin/env python3
"""
Text generation utility script.
"""

import argparse
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.trainer import DialogTrainer
from src.config.settings import get_model_config


def main():
    parser = argparse.ArgumentParser(description="Generate text using trained model")
    parser.add_argument("--model", type=str, default="gpt2-small",
                       help="Model type to use")
    parser.add_argument("--model-path", type=str,
                       help="Path to trained model (overrides --model)")
    parser.add_argument("--instruction", type=str, required=True,
                       help="Instruction for text generation")
    parser.add_argument("--max-length", type=int, default=150,
                       help="Maximum length of generated text")
    
    args = parser.parse_args()
    
    if args.model_path:
        # Load from specific path
        print(f"Loading model from: {args.model_path}")
        # TODO: Implement loading from checkpoint
    else:
        # Use configured model
        model_config = get_model_config(args.model)
        print(f"Using model: {model_config.name}")
        trainer = DialogTrainer(model_config=model_config)
    
    # Generate response
    print(f"\nInstruction: {args.instruction}")
    print("=" * 50)
    response = trainer.generate_response(args.instruction, max_length=args.max_length)
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
