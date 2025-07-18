"""
PyTorch trainer for dialog datasets.
Handles model initialization, training, and text generation for ML model comparison.
Works with pre-trained models and custom models built from scratch.
"""

import torch
import os
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset as HFDataset


class DialogTrainer:
    """Main trainer class for dialog datasets and ML model comparison."""
    
    def __init__(self, model_config=None):
        """
        Initialize the trainer.
        
        Args:
            model_config: ModelConfig object (optional, uses default config)
        """
        # Import here to avoid circular imports
        from ..config.settings import DEFAULT_MODEL_CONFIG
        
        self.config = model_config or DEFAULT_MODEL_CONFIG
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize based on model type
        if self.config.from_scratch:
            self._init_custom_model()
        else:
            self._init_pretrained_model()
        
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def _init_pretrained_model(self):
        """Initialize pre-trained model for fine-tuning."""
        print(f"Loading pre-trained model: {self.config.name}")
        print(f"Output directory: {self.config.output_dir}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(self.config.name)
        self.model.to(self.device)
    
    def _init_custom_model(self):
        """Initialize custom model built from scratch."""
        print(f"Building custom model from scratch: {self.config.name}")
        print(f"Architecture: {self.config.n_layer} layers, {self.config.n_embd} dim, {self.config.n_head} heads")
        print(f"Output directory: {self.config.output_dir}")
        
        # Use GPT-2 tokenizer as base (you can customize this)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create custom model
        from .models import create_custom_model
        self.model = create_custom_model(self.config)
        self.model.to(self.device)
    
    def prepare_dataset(self, data, train_split=0.9, max_samples=None):
        """
        Prepare the dataset for training using HuggingFace datasets.
        
        Args:
            data: Either a pandas DataFrame (with 'text' column) or list of text strings
            train_split: Fraction of data to use for training
            max_samples: Maximum number of samples to use (for testing)
            
        Returns:
            tuple: (train_dataset, eval_dataset)
        """
        # Handle different input types
        if hasattr(data, 'iloc'):  # pandas DataFrame
            texts = data['text'].tolist()
        elif isinstance(data, list):  # List of strings
            texts = data
        else:
            raise ValueError("Data must be a pandas DataFrame with 'text' column or list of strings")
        
        # Limit samples if specified (useful for testing)
        if max_samples:
            texts = texts[:max_samples]
            print(f"Using {len(texts)} samples for training")
        
        # Split into train and validation
        split_idx = int(len(texts) * train_split)
        train_texts = texts[:split_idx]
        eval_texts = texts[split_idx:]
        
        print(f"Training samples: {len(train_texts)}")
        print(f"Evaluation samples: {len(eval_texts)}")
        
        # Create HuggingFace datasets
        train_dataset = HFDataset.from_dict({"text": train_texts})
        eval_dataset = HFDataset.from_dict({"text": eval_texts})
        
        # Tokenize the datasets
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], 
                truncation=True, 
                padding=False,  # Let the data collator handle padding
                max_length=512
            )
        
        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        
        return train_dataset, eval_dataset
    
    def train(self, train_dataset, eval_dataset, 
              num_epochs=3, batch_size=4, learning_rate=5e-5,
              save_steps=500, eval_steps=500):
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            save_steps: Steps between saving checkpoints
            eval_steps: Steps between evaluations
        """
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            logging_steps=50,
            save_steps=save_steps,
            eval_steps=eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f"{self.config.output_dir}/logs",
            report_to=None,  # Disable wandb/tensorboard
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        print("Starting training...")
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        print(f"Training completed! Model saved to {self.config.output_dir}")
        
        return trainer
    
    def generate_response(self, instruction, max_length=150):
        """
        Generate a response for a given instruction.
        
        Args:
            instruction: The instruction/prompt
            max_length: Maximum length of generated response
            
        Returns:
            str: Generated response
        """
        # Format the input like in training
        input_text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Response:\n"
        )
        
        # Tokenize and generate
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            if self.config.from_scratch and hasattr(self.model, 'generate'):
                # Use custom generate method for scratch models
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    temperature=0.7,
                    top_k=50
                )
            else:
                # Use HuggingFace generate for pre-trained models
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        
        # Decode and extract just the response part
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(input_text):].strip()
        
        return response
