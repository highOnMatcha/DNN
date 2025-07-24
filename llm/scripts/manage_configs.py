#!/usr/bin/env python3
"""
Configuration management CLI utility.

This script provides a command-line interface for managing model configurations
including listing available models, adding new custom model configurations,
and displaying detailed information about specific model configurations.
"""

import argparse
import sys
import os
from typing import NoReturn

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import (
    list_available_models, 
    add_custom_model_to_config,
    get_model_config
)


def main() -> NoReturn:
    """
    Main entry point for configuration management utility.
    
    Parses command-line arguments and executes the requested configuration
    management operation (list, add, or show model configurations).
    """
    parser = argparse.ArgumentParser(description="Manage model configurations")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List models command
    list_parser = subparsers.add_parser('list', help='List available models')
    
    # Add model command
    add_parser = subparsers.add_parser('add', help='Add a new custom model')
    add_parser.add_argument('--key', required=True, help='Model key identifier')
    add_parser.add_argument('--name', required=True, help='Model name')
    add_parser.add_argument('--description', required=True, help='Model description')
    add_parser.add_argument('--output-dir', required=True, help='Output directory')
    add_parser.add_argument('--n-embd', type=int, default=768, help='Embedding dimension')
    add_parser.add_argument('--n-layer', type=int, default=12, help='Number of layers')
    add_parser.add_argument('--n-head', type=int, default=12, help='Number of attention heads')
    
    # Show model command
    show_parser = subparsers.add_parser('show', help='Show details for a specific model')
    show_parser.add_argument('model_key', help='Model key to show details for')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    if args.command == 'list':
        list_available_models()
    
    elif args.command == 'add':
        add_custom_model_to_config(
            model_key=args.key,
            name=args.name,
            description=args.description,
            output_dir=args.output_dir,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head
        )
    
    elif args.command == 'show':
        try:
            config = get_model_config(args.model_key)
            print(f"Model Configuration for '{args.model_key}':")
            print("=" * 40)
            print(f"Name: {config.name}")
            print(f"Output Directory: {config.output_dir}")
            print(f"From Scratch: {config.from_scratch}")
            if config.from_scratch:
                print(f"Architecture:")
                print(f"  Embedding Dimension: {config.n_embd}")
                print(f"  Layers: {config.n_layer}")
                print(f"  Attention Heads: {config.n_head}")
                print(f"  Vocabulary Size: {config.vocab_size}")
                print(f"  Max Sequence Length: {config.max_sequence_length}")
                print(f"  Dropout: {config.dropout}")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
