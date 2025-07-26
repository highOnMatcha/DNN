"""
Model export utilities for deployment and distribution.

Provides functions to export trained models in various formats for different deployment
scenarios including PyTorch files, ONNX, and HuggingFace format.
"""

import os
import torch
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
from transformers import AutoTokenizer, AutoModelForCausalLM

from core.models import CustomGPTModel, CustomGPTConfig


class ModelExporter:
    """Utility class for exporting trained models in various formats."""
    
    def __init__(self, model_path: str):
        """
        Initialize model exporter.
        
        Args:
            model_path: Path to the trained model directory or checkpoint
        """
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self._load_model_and_tokenizer()
    
    def _find_latest_checkpoint(self, model_dir: Path) -> Optional[Path]:
        """Find the latest checkpoint in a directory."""
        checkpoints = []
        for item in model_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    step = int(item.name.split("-")[1])
                    checkpoints.append((step, item))
                except ValueError:
                    continue
        
        if checkpoints:
            # Return the checkpoint with the highest step number
            return max(checkpoints, key=lambda x: x[0])[1]
        return None
    
    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer from the specified path."""
        print(f"Loading model from {self.model_path}...")
        
        # First, try to find the correct checkpoint directory
        checkpoint_dir = None
        
        # Check if the path directly contains model files
        if (self.model_path / "model.safetensors").exists() or (self.model_path / "pytorch_model.bin").exists():
            checkpoint_dir = self.model_path
        else:
            # Look for checkpoint directories
            checkpoint_dir = self._find_latest_checkpoint(self.model_path)
            if checkpoint_dir is None:
                # Check if it's a direct path to a checkpoint
                if self.model_path.name.startswith("checkpoint-") and self.model_path.is_dir():
                    checkpoint_dir = self.model_path
        
        if checkpoint_dir is None:
            raise FileNotFoundError(f"No valid checkpoint found in {self.model_path}")
        
        print(f"Using checkpoint: {checkpoint_dir}")
        
        try:
            # Try loading as HuggingFace model first
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
            print(f"✅ Loaded HuggingFace model from {checkpoint_dir}")
            return
        except Exception as e:
            print(f"⚠️  HuggingFace loading failed: {e}")
        
        # Try loading as custom model
        try:
            # Load model checkpoint
            checkpoint = None
            if (checkpoint_dir / "model.safetensors").exists():
                print("Loading from safetensors...")
                from safetensors.torch import load_file
                checkpoint = load_file(checkpoint_dir / "model.safetensors", device="cpu")
            elif (checkpoint_dir / "pytorch_model.bin").exists():
                print("Loading from pytorch_model.bin...")
                checkpoint = torch.load(checkpoint_dir / "pytorch_model.bin", map_location="cpu")
            else:
                raise FileNotFoundError("No model checkpoint file found")
            
            # Load config
            config_path = checkpoint_dir / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"No config.json found in {checkpoint_dir}")
            
            with open(config_path) as f:
                config_dict = json.load(f)
            
            # Create model
            config = CustomGPTConfig(**config_dict)
            self.model = CustomGPTModel(config)
            self.model.load_state_dict(checkpoint)
            print(f"✅ Loaded custom model from {checkpoint_dir}")
            
            # Load tokenizer from checkpoint directory
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
                print(f"✅ Loaded tokenizer from {checkpoint_dir}")
            except Exception:
                print("⚠️  Using default GPT-2 tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e2:
            raise ValueError(f"Could not load model: {e2}")
    
    def export_pytorch(self, output_path: str, include_tokenizer: bool = True) -> str:
        """
        Export model as PyTorch .pt file.
        
        Args:
            output_path: Output file path for the .pt file
            include_tokenizer: Whether to include tokenizer in a separate file
            
        Returns:
            Path to the exported file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare model for inference
        self.model.eval()
        
        # Create export dictionary
        export_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.config.to_dict() if hasattr(self.model, 'config') else None,
            'model_type': type(self.model).__name__,
        }
        
        # Save the model
        torch.save(export_dict, output_file)
        print(f"✅ Model exported to {output_file}")
        
        # Save tokenizer separately if requested
        if include_tokenizer and self.tokenizer:
            self.tokenizer.save_pretrained(output_file.parent)
            print(f"✅ Tokenizer saved to {output_file.parent}")
        
        return str(output_file)
    
    def export_huggingface(self, output_path: str) -> str:
        """
        Export model in HuggingFace format for easy sharing and deployment.
        
        Args:
            output_path: Output directory path
            
        Returns:
            Path to the exported directory
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer in HuggingFace format
        self.model.save_pretrained(output_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)
        
        print(f"✅ HuggingFace model exported to {output_dir}")
        return str(output_dir)
    
    def export_onnx(self, output_path: str, input_sample: Optional[torch.Tensor] = None) -> str:
        """
        Export model to ONNX format for optimized inference.
        
        Args:
            output_path: Output file path for the .onnx file
            input_sample: Sample input tensor for tracing
            
        Returns:
            Path to the exported file
        """
        try:
            import onnx
            import torch.onnx
        except ImportError:
            raise ImportError("ONNX export requires 'onnx' package. Install with: pip install onnx")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
        
        # Create sample input if not provided
        if input_sample is None:
            batch_size, seq_length = 1, 512
            vocab_size = getattr(self.tokenizer, 'vocab_size', 50257)
            input_sample = torch.randint(0, vocab_size, (batch_size, seq_length))
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            (input_sample,),
            output_file,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        
        print(f"✅ ONNX model exported to {output_file}")
        return str(output_file)
    
    def create_deployment_package(self, output_dir: str) -> str:
        """
        Create a complete deployment package with model, tokenizer, and inference script.
        
        Args:
            output_dir: Output directory for the deployment package
            
        Returns:
            Path to the deployment package directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export model in HuggingFace format
        self.export_huggingface(str(output_path / "model"))
        
        # Copy the core models module for standalone deployment
        self._create_standalone_inference_script(output_path)
        
        # Create requirements.txt
        requirements = '''torch>=1.9.0
transformers>=4.20.0
safetensors>=0.3.0
'''
        with open(output_path / "requirements.txt", "w") as f:
            f.write(requirements)
        
        # Create README
        readme = f'''# Model Deployment Package

This package contains your trained language model ready for deployment.

## Contents
- `model/`: HuggingFace format model and tokenizer
- `inference.py`: Simple inference script
- `requirements.txt`: Python dependencies

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run inference:
```python
from inference import ModelInference

model = ModelInference()
result = model.generate("Your prompt here:")
print(result)
```

## Model Info
- Model path: {self.model_path}
- Model type: {type(self.model).__name__}
- Tokenizer vocab size: {getattr(self.tokenizer, 'vocab_size', 'Unknown')}
'''
        
        with open(output_path / "README.md", "w") as f:
            f.write(readme)
        
        print(f"✅ Deployment package created at {output_path}")
        return str(output_path)
    
    def _create_standalone_inference_script(self, output_path: Path):
        """Create a standalone inference script that imports from the actual models module."""
        inference_script = '''"""
Simple inference script for the exported model.
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer

# Import our custom model classes
import sys
sys.path.append('..')  # Adjust path if needed
from src.core.models import CustomGPTModel, CustomGPTConfig

class ModelInference:
    def __init__(self, model_path="./model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Try loading as HuggingFace model first
        try:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path).to(self.device)
            print("✅ Loaded HuggingFace model")
        except Exception as e:
            print(f"⚠️  HuggingFace loading failed: {e}")
            # Load as custom model
            self._load_custom_model()
        
        self.model.eval()
    
    def _load_custom_model(self):
        """Load custom model from checkpoint."""
        import json
        
        # Load configuration
        config_path = self.model_path / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)
        config = CustomGPTConfig(**config_dict)
        
        # Create and load model
        self.model = CustomGPTModel(config).to(self.device)
        
        # Load model weights
        try:
            from safetensors.torch import load_file
            state_dict = load_file(self.model_path / "model.safetensors", device=str(self.device))
        except ImportError:
            state_dict = torch.load(self.model_path / "pytorch_model.bin", map_location=self.device)
        
        self.model.load_state_dict(state_dict)
        print("✅ Loaded custom model")
    
    def generate(self, prompt, max_length=100, temperature=0.8, do_sample=True, top_p=0.9):
        """Generate text from a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Get logits from model
                if hasattr(self.model, 'logits'):
                    # HuggingFace model
                    outputs = self.model(input_ids)
                    logits = outputs.logits
                else:
                    # Custom model
                    outputs = self.model(input_ids)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                next_token_logits = logits[0, -1, :]
                
                if do_sample:
                    # Apply temperature and top-p sampling
                    next_token_logits = next_token_logits / temperature
                    
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                    
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    print("Loading model...")
    inference = ModelInference()
    print("Model loaded successfully!")
    
    prompt = "Tell me about artificial intelligence:"
    print(f"\\nPrompt: {prompt}")
    print("Generating...")
    
    result = inference.generate(prompt, max_length=150, temperature=0.8)
    print(f"\\nGenerated text:\\n{result}")
'''
        
        with open(output_path / "inference.py", "w") as f:
            f.write(inference_script)


def export_model(model_path: str, output_path: str, format: str = "pytorch") -> str:
    """
    Convenience function to export a model.
    
    Args:
        model_path: Path to the trained model
        output_path: Output path for exported model
        format: Export format ('pytorch', 'huggingface', 'onnx', 'deployment')
        
    Returns:
        Path to exported model
    """
    exporter = ModelExporter(model_path)
    
    if format == "pytorch":
        return exporter.export_pytorch(output_path)
    elif format == "huggingface":
        return exporter.export_huggingface(output_path)
    elif format == "onnx":
        return exporter.export_onnx(output_path)
    elif format == "deployment":
        return exporter.create_deployment_package(output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export trained language model")
    parser.add_argument("model_path", help="Path to trained model")
    parser.add_argument("output_path", help="Output path for exported model")
    parser.add_argument("--format", choices=["pytorch", "huggingface", "onnx", "deployment"], 
                       default="deployment", help="Export format")
    
    args = parser.parse_args()
    
    result_path = export_model(args.model_path, args.output_path, args.format)
    print(f"✓ Model exported successfully to: {result_path}")
