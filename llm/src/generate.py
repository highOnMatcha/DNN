#!/usr/bin/env python3
"""
Text generation utility for trained models.

This script provides a command-line interface for generating text using trained
dialog models. It supports single generation, interactive mode, and listing 
available model configurations.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add project paths
current_dir = Path(__file__).parent  # This is /llm/src/
sys.path.insert(0, str(current_dir))

try:
    from core.trainer import DialogTrainer
    from config.settings import get_model_config, list_available_models
except ImportError as e:
    print("Error: Could not import LLM modules.")
    print(f"Import error: {e}")
    print("Make sure you have installed the requirements:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def generate_text(args) -> None:
    """Generate text using a trained model."""
    try:
        if args.model_path:
            print(f"Loading model from: {args.model_path}")
            print("Note: Loading from custom checkpoint paths not yet fully implemented")
            print("Using default model configuration for now")
            model_config = get_model_config("custom-tiny")
        else:
            model_config = get_model_config(args.model)
            print(f"Using model: {model_config.name}")
        
        trainer = DialogTrainer(model_config=model_config)
        
        print(f"\nInstruction: {args.instruction}")
        print("=" * 50)
        
        response = trainer.generate_response(
            args.instruction, 
            max_length=args.max_length
        )
        
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error generating text: {e}")
        sys.exit(1)


def list_models(args) -> None:
    """List available model configurations."""
    try:
        list_available_models()
    except Exception as e:
        print(f"Error listing models: {e}")
        sys.exit(1)


def interactive_mode(args) -> None:
    """Start an interactive session for text generation."""
    try:
        model_config = get_model_config(args.model)
        print(f"Loading model: {model_config.name}")
        trainer = DialogTrainer(model_config=model_config)
        
        print("\nInteractive Text Generation")
        print("=" * 30)
        print("Type 'quit' or 'exit' to end the session")
        print("Type 'help' for commands")
        print()
        
        while True:
            try:
                instruction = input("Enter instruction: ").strip()
                
                if instruction.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                elif instruction.lower() == 'help':
                    print("\nCommands:")
                    print("  quit, exit - End session")
                    print("  help - Show this help")
                    print("  Any other text - Generate response")
                    print()
                    continue
                elif not instruction:
                    continue
                
                print("\nGenerating response...")
                response = trainer.generate_response(
                    instruction, 
                    max_length=args.max_length
                )
                print(f"Response: {response}\n")
                
            except KeyboardInterrupt:
                print("\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
                
    except Exception as e:
        print(f"Error starting interactive mode: {e}")
        sys.exit(1)


def main():
    """Main entry point with command parsing for text generation."""
    parser = argparse.ArgumentParser(
        description="Text Generation CLI for trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single text generation
  python src/generate.py --model custom-tiny "Explain machine learning"
  python src/generate.py --model gpt2-small "Write a poem" --max-length 200
  
  # Interactive Mode
  python src/generate.py --interactive --model custom-tiny
  
  # List available models
  python src/generate.py --list-models
        """
    )
    
    # Main generation arguments
    parser.add_argument('instruction', nargs='?', type=str,
                       help='Instruction for text generation (required unless --interactive or --list-models)')
    parser.add_argument('--model', '-m', type=str, default='custom-tiny',
                       help='Model to use for generation (default: custom-tiny)')
    parser.add_argument('--model-path', type=str,
                       help='Path to trained model checkpoint')
    parser.add_argument('--max-length', type=int, default=150,
                       help='Maximum length of generated text (default: 150)')
    
    # Mode arguments
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Start interactive text generation mode')
    parser.add_argument('--list-models', action='store_true',
                       help='List available model configurations')
    
    args = parser.parse_args()
    
    # Handle different modes
    if args.list_models:
        list_models(args)
    elif args.interactive:
        interactive_mode(args)
    elif args.instruction:
        # Create a namespace with instruction for compatibility
        generation_args = argparse.Namespace(
            instruction=args.instruction,
            model=args.model,
            model_path=args.model_path,
            max_length=args.max_length
        )
        generate_text(generation_args)
    else:
        parser.error("Either provide an instruction, use --interactive, or --list-models")


if __name__ == "__main__":
    main()
