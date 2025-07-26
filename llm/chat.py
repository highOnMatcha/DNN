#!/usr/bin/env python3
"""
Interactive Chat Interface for PyTorch .pt Model

This script provides a unified experience for loading and chatting with your exported .pt model.
Features:
- Automatic model and tokenizer loading
- Interactive chat interface with command support
- Configurable generation parameters
- Error handling and graceful exit
"""

import torch
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.append('./src')

from transformers import AutoTokenizer
from core.models import CustomGPTModel, CustomGPTConfig

class ModelChat:
    def __init__(self, model_path='./export/medium.pt', tokenizer_path='./export'):
        """
        Initialize the chat interface with model and tokenizer.
        
        Args:
            model_path: Path to the .pt model file
            tokenizer_path: Path to the tokenizer directory
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        
        # Generation parameters (can be modified during chat)
        self.max_length = 150
        self.temperature = 0.8
        self.top_p = 0.9
        self.do_sample = True
        
        self._load_model()
        self._load_tokenizer()
        self._print_model_info()
    
    def _load_model(self):
        """Load the PyTorch model from .pt file."""
        print(f"🔄 Loading model from {self.model_path}...")
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Extract model configuration and state
        self.model_config = checkpoint.get('model_config')
        if not self.model_config:
            raise ValueError("No model configuration found in .pt file")
        
        state_dict = checkpoint['model_state_dict']
        
        # Create and load model
        config = CustomGPTConfig(**self.model_config)
        self.model = CustomGPTModel(config)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully! Using {self.device}")
    
    def _load_tokenizer(self):
        """Load the tokenizer."""
        print(f"🔄 Loading tokenizer from {self.tokenizer_path}...")
        
        try:
            if Path(self.tokenizer_path).exists():
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            else:
                print("⚠️  Tokenizer directory not found, using default GPT-2 tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception as e:
            print(f"⚠️  Error loading tokenizer: {e}")
            print("Falling back to GPT-2 tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ Tokenizer loaded successfully!")
    
    def _print_model_info(self):
        """Print model information."""
        param_count = sum(p.numel() for p in self.model.parameters())
        print("\n" + "="*60)
        print("📊 MODEL INFORMATION")
        print("="*60)
        print(f"Architecture: Custom GPT")
        print(f"Layers: {self.model_config['n_layer']}")
        print(f"Embedding Dimension: {self.model_config['n_embd']}")
        print(f"Attention Heads: {self.model_config['n_head']}")
        print(f"Vocabulary Size: {self.model_config['vocab_size']}")
        print(f"Parameters: {param_count:,}")
        print(f"Device: {self.device}")
        print("="*60)
    
    def generate(self, prompt):
        """
        Generate text from a prompt using the model.
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Generated text string
        """
        print(f"🤔 Thinking...")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        with torch.no_grad():
            for _ in range(self.max_length - input_ids.size(1)):
                # Get model predictions
                output = self.model(input_ids)
                logits = output.logits
                next_token_logits = logits[0, -1, :]
                
                if self.do_sample:
                    # Apply temperature scaling
                    next_token_logits = next_token_logits / self.temperature
                    
                    # Apply top-p (nucleus) sampling
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > self.top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample from the filtered distribution
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy sampling
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Add the new token to the sequence
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
                
                # Stop if we hit the end token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode the full sequence
        result = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return result
    
    def _print_help(self):
        """Print help information."""
        print("""
🔧 AVAILABLE COMMANDS:
• help - Show this help message
• settings - Show current generation settings
• temp <value> - Set temperature (0.1-2.0, default: 0.8)
• length <value> - Set max length (10-500, default: 150)
• topp <value> - Set top-p value (0.1-1.0, default: 0.9)
• sample <on/off> - Toggle sampling (default: on)
• clear - Clear screen
• quit/exit/q - Exit the chat

💡 TIPS:
• Lower temperature = more focused responses
• Higher temperature = more creative responses
• Try instruction-style prompts like "Explain how to..."
        """)
    
    def _print_settings(self):
        """Print current generation settings."""
        print(f"""
⚙️  CURRENT SETTINGS:
• Temperature: {self.temperature}
• Max Length: {self.max_length}
• Top-p: {self.top_p}
• Sampling: {'On' if self.do_sample else 'Off'}
        """)
    
    def _handle_command(self, command):
        """Handle chat commands."""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == 'help':
            self._print_help()
        elif cmd == 'settings':
            self._print_settings()
        elif cmd == 'temp' and len(parts) == 2:
            try:
                temp = float(parts[1])
                if 0.1 <= temp <= 2.0:
                    self.temperature = temp
                    print(f"✅ Temperature set to {temp}")
                else:
                    print("❌ Temperature must be between 0.1 and 2.0")
            except ValueError:
                print("❌ Invalid temperature value")
        elif cmd == 'length' and len(parts) == 2:
            try:
                length = int(parts[1])
                if 10 <= length <= 500:
                    self.max_length = length
                    print(f"✅ Max length set to {length}")
                else:
                    print("❌ Max length must be between 10 and 500")
            except ValueError:
                print("❌ Invalid length value")
        elif cmd == 'topp' and len(parts) == 2:
            try:
                topp = float(parts[1])
                if 0.1 <= topp <= 1.0:
                    self.top_p = topp
                    print(f"✅ Top-p set to {topp}")
                else:
                    print("❌ Top-p must be between 0.1 and 1.0")
            except ValueError:
                print("❌ Invalid top-p value")
        elif cmd == 'sample' and len(parts) == 2:
            setting = parts[1].lower()
            if setting in ['on', 'true', '1']:
                self.do_sample = True
                print("✅ Sampling enabled")
            elif setting in ['off', 'false', '0']:
                self.do_sample = False
                print("✅ Sampling disabled (greedy)")
            else:
                print("❌ Use 'on' or 'off'")
        elif cmd == 'clear':
            os.system('clear' if os.name == 'posix' else 'cls')
            self._print_model_info()
        else:
            print("❌ Unknown command. Type 'help' for available commands.")
    
    def chat(self):
        """Start the interactive chat session."""
        print("\n🚀 CHAT INTERFACE READY")
        print("Type 'help' for commands, 'quit' to exit")
        print("="*60)
        
        while True:
            try:
                # Get user input
                user_input = input("\n💭 You: ").strip()
                
                # Handle exit commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Thanks for chatting! Goodbye!")
                    break
                
                # Handle empty input
                if not user_input:
                    print("Please enter a prompt or command.")
                    continue
                
                # Handle commands (start with special characters or known commands)
                if user_input.startswith('/') or user_input.split()[0].lower() in [
                    'help', 'settings', 'temp', 'length', 'topp', 'sample', 'clear'
                ]:
                    self._handle_command(user_input.lstrip('/'))
                    continue
                
                # Generate response
                try:
                    response = self.generate(user_input)
                    print(f"\n🤖 AI: {response}")
                except Exception as e:
                    print(f"❌ Generation error: {e}")
                
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function to start the chat interface."""
    import sys
    
    # Check for model argument
    available_models = {
        "nano": "55MB, 13.5M params, fastest inference",
        "tiny": "115MB, 29M params, good balance", 
        "small": "302MB, ~80M params, better quality",
        "medium": "633MB, ~160M params, best quality"
    }
    model_name = "tiny"  # default
    
    if len(sys.argv) > 1:
        requested_model = sys.argv[1].lower()
        if requested_model in available_models:
            model_name = requested_model
        else:
            print(f"❌ Unknown model '{requested_model}'.")
            print(f"\nAvailable models:")
            for model, desc in available_models.items():
                print(f"  • {model}: {desc}")
            print(f"\nUsage: python chat.py [model_name]")
            print(f"Example: python chat.py medium")
            return
    
    model_path = f"./export/{model_name}/{model_name}.pt"
    tokenizer_path = f"./export/{model_name}"
    
    try:
        print(f"🚀 Loading {model_name.upper()} model...")
        # Initialize the chat interface
        chat_interface = ModelChat(model_path, tokenizer_path)
        
        # Start chatting
        chat_interface.chat()
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print(f"Make sure you have exported the {model_name} model using export.py first.")
        print(f"\nAvailable models:")
        for model, desc in available_models.items():
            print(f"  • {model}: {desc}")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return

if __name__ == "__main__":
    main()
