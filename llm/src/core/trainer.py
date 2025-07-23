"""
Dialog model trainer with PyTorch and Transformers integration.
Supports both pre-trained fine-tuning and custom model training from scratch.
"""

import torch
import os
import time
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from datasets import Dataset as HFDataset


class WandBCallback(TrainerCallback):
    """Centralized WandB logging callback."""
    
    def __init__(self, wandb_run=None):
        self.wandb_run = wandb_run
        self.start_time = None
        self.logged_model_info = False
        
    def on_train_begin(self, args, state, control, **kwargs):
        if self.wandb_run and not self.logged_model_info:
            self.start_time = time.time()
            model = kwargs.get('model')
            if model:
                self.wandb_run.watch(model, log="all", log_freq=1, log_graph=False)
                self.logged_model_info = True
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.wandb_run and logs:
            if torch.cuda.is_available():
                logs["system/gpu_memory_gb"] = torch.cuda.memory_allocated() / 1e9
            
            if self.start_time:
                logs["training/elapsed_minutes"] = (time.time() - self.start_time) / 60


class DialogTrainer:
    """Main trainer class for dialog datasets."""
    
    def __init__(self, model_config=None, wandb_run=None):
        try:
            from ..config.settings import DEFAULT_MODEL_CONFIG
        except ImportError:
            from src.config.settings import DEFAULT_MODEL_CONFIG
        
        self.config = model_config or DEFAULT_MODEL_CONFIG
        self.wandb_run = wandb_run
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.wandb_run:
            config_data = {
                "model_name": self.config.name,
                "output_dir": self.config.output_dir,
                "max_sequence_length": self.config.max_sequence_length,
                "from_scratch": self.config.from_scratch,
                "device": str(self.device),
            }
            
            if self.config.from_scratch:
                config_data.update({
                    "vocab_size": self.config.vocab_size,
                    "n_embd": self.config.n_embd,
                    "n_layer": self.config.n_layer,
                    "n_head": self.config.n_head,
                    "dropout": self.config.dropout,
                })
            
            self.wandb_run.config.update(config_data)
        
        if self.config.from_scratch:
            self._init_custom_model()
        else:
            self._init_pretrained_model()
        
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def _init_pretrained_model(self):
        """Initialize pre-trained model for fine-tuning."""
        print(f"Loading model: {self.config.name}")
        print(f"Output: {self.config.output_dir}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(self.config.name)
        self.model.to(self.device)
    
    def _init_custom_model(self):
        """Initialize custom model from scratch."""
        print(f"Building model: {self.config.name}")
        print(f"Architecture: {self.config.n_layer} layers, {self.config.n_embd} dim, {self.config.n_head} heads")
        print(f"Output: {self.config.output_dir}")
        
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        try:
            from .models import create_custom_model
        except ImportError:
            from src.core.models import create_custom_model
        self.model = create_custom_model(self.config)
        self.model.to(self.device)
    
    def prepare_dataset(self, data, train_split=0.9, max_samples=None):
        """Prepare dataset for training."""
        if hasattr(data, 'iloc'):
            texts = data['text'].tolist()
        elif isinstance(data, list):
            texts = data
        else:
            raise ValueError("Data must be DataFrame with 'text' column or list of strings")
        
        if max_samples:
            texts = texts[:max_samples]
            print(f"Using {len(texts)} samples")
        
        split_idx = int(len(texts) * train_split)
        train_texts = texts[:split_idx]
        eval_texts = texts[split_idx:]
        
        print(f"Train: {len(train_texts)}, Eval: {len(eval_texts)}")
        
        train_dataset = HFDataset.from_dict({"text": train_texts})
        eval_dataset = HFDataset.from_dict({"text": eval_texts})
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], 
                truncation=True, 
                padding=False,
                max_length=512
            )
        
        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        
        return train_dataset, eval_dataset
    
    def train(self, train_dataset, eval_dataset, 
              num_epochs=3, batch_size=4, learning_rate=5e-5,
              save_steps=500, eval_steps=500):
        """Train the model."""
        if self.wandb_run:
            training_params = {
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "train_dataset_size": len(train_dataset),
                "eval_dataset_size": len(eval_dataset),
            }
            self.wandb_run.config.update(training_params)
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            logging_steps=1,
            save_steps=save_steps,
            eval_steps=eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f"{self.config.output_dir}/logs",
            report_to=["wandb"] if self.wandb_run else [],
            run_name=self.wandb_run.name if self.wandb_run else None,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[WandBCallback(self.wandb_run)] if self.wandb_run else [],
        )
        
        print("Starting training...")
        start_time = time.time()
        trainer.train()
        training_duration = time.time() - start_time
        
        if self.wandb_run:
            final_metrics = {
                "training/total_duration_minutes": training_duration / 60,
                "training/completed": True,
            }
            self.wandb_run.log(final_metrics)
        
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        print(f"Training completed. Model saved to {self.config.output_dir}")
        print(f"Duration: {training_duration/60:.1f} minutes")
        
        return trainer
    
    def generate_response(self, instruction, max_length=150, log_to_wandb=False):
        """Generate response for given instruction."""
        input_text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Response:\n"
        )
        
        start_time = time.time()
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            if self.config.from_scratch and hasattr(self.model, 'generate'):
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    temperature=0.7,
                    top_k=50
                )
            else:
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        
        generation_time = time.time() - start_time
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(input_text):].strip()
        
        if log_to_wandb and self.wandb_run:
            self.wandb_run.log({
                "generation/time_seconds": generation_time,
                "generation/response_length": len(response),
                "generation/response_words": len(response.split()),
                "generation/input_length": len(instruction),
                "generation/tokens_generated": len(outputs[0]) - len(inputs[0]),
            })
        
        return response
