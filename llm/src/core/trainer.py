"""
Dialog model trainer with PyTorch and Transformers integration.

This module provides a comprehensive training framework for dialog datasets, supporting
both pre-trained model fine-tuning and custom model training from scratch. It includes
WandB integration for experiment tracking and monitoring.
"""

import os
import time
from typing import Optional, Tuple, Any, Union

import torch
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.trainer_callback import TrainerCallback


class WandBCallback(TrainerCallback):
    """
    Centralized WandB logging callback for training metrics and system monitoring.
    
    This callback integrates with Weights & Biases to track training progress,
    model parameters, and system resources during training. It automatically
    logs GPU memory usage and training duration.
    """
    
    def __init__(self, wandb_run: Optional[Any] = None) -> None:
        """
        Initialize the WandB callback.
        
        Args:
            wandb_run: Active WandB run instance for logging metrics. If None,
                      no logging will be performed.
        """
        self.wandb_run = wandb_run
        self.start_time: Optional[float] = None
        self.logged_model_info: bool = False
        
    def on_train_begin(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        """
        Called at the beginning of training to initialize model watching.
        
        Args:
            args: Training arguments from the Trainer.
            state: Current training state.
            control: Training control object.
            **kwargs: Additional keyword arguments including the model.
        """
        if self.wandb_run and not self.logged_model_info:
            self.start_time = time.time()
            model = kwargs.get('model')
            if model:
                self.wandb_run.watch(model, log="all", log_freq=1, log_graph=False)
                self.logged_model_info = True
    
    def on_log(self, args: TrainingArguments, state: Any, control: Any, 
               logs: Optional[dict] = None, **kwargs: Any) -> None:
        """
        Called on each logging step to add system metrics.
        
        Args:
            args: Training arguments from the Trainer.
            state: Current training state.
            control: Training control object.
            logs: Dictionary of logged metrics to be augmented.
            **kwargs: Additional keyword arguments.
        """
        if self.wandb_run and logs:
            if torch.cuda.is_available():
                logs["system/gpu_memory_gb"] = torch.cuda.memory_allocated() / 1e9
            
            if self.start_time:
                logs["training/elapsed_minutes"] = (time.time() - self.start_time) / 60


class DialogTrainer:
    """
    Main trainer class for dialog datasets.
    
    This class handles the complete training pipeline for dialog models, including
    dataset preparation, model initialization (both pre-trained and custom), training
    with WandB integration, and text generation capabilities.
    
    Attributes:
        config: Model configuration object containing architecture parameters.
        wandb_run: Optional WandB run instance for experiment tracking.
        device: PyTorch device (CPU or CUDA) for model computation.
        tokenizer: Tokenizer for text preprocessing.
        model: The neural network model (pre-trained or custom).
    """
    
    def __init__(self, model_config: Optional[Any] = None, wandb_run: Optional[Any] = None) -> None:
        """
        Initialize the DialogTrainer with model configuration and optional WandB tracking.
        
        Args:
            model_config: Configuration object containing model parameters. If None,
                         uses the default configuration from settings.
            wandb_run: Optional WandB run instance for experiment tracking.
        """
        try:
            from ..config.settings import DEFAULT_MODEL_CONFIG
        except ImportError:
            from config.settings import DEFAULT_MODEL_CONFIG
        
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
    
    def _init_pretrained_model(self) -> None:
        """
        Initialize pre-trained model for fine-tuning.
        
        Loads a pre-trained model and tokenizer from HuggingFace model hub,
        sets up padding tokens, and moves the model to the appropriate device.
        """
        print(f"Loading model: {self.config.name}")
        print(f"Output: {self.config.output_dir}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(self.config.name)
        self.model.to(self.device)
    
    def _init_custom_model(self) -> None:
        """
        Initialize custom model from scratch.
        
        Creates a custom GPT-style model using the architecture parameters
        specified in the configuration. Uses GPT-2 tokenizer for compatibility.
        """
        print(f"Building model: {self.config.name}")
        print(f"Architecture: {self.config.n_layer} layers, {self.config.n_embd} dim, {self.config.n_head} heads")
        print(f"Output: {self.config.output_dir}")
        
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        try:
            from .models import create_custom_model
        except ImportError:
            from core.models import create_custom_model
        self.model = create_custom_model(self.config)
        self.model.to(self.device)
    
    def prepare_dataset(self, data: Union[Any, list], train_split: float = 0.9, 
                       max_samples: Optional[int] = None) -> Tuple[HFDataset, HFDataset]:
        """
        Prepare dataset for training by tokenizing and splitting into train/eval sets.
        
        Args:
            data: Input data as either a pandas DataFrame with 'text' column or 
                 a list of strings.
            train_split: Fraction of data to use for training (remainder for evaluation).
            max_samples: Maximum number of samples to use. If None, uses all data.
        
        Returns:
            Tuple of (train_dataset, eval_dataset) as HuggingFace Dataset objects.
        
        Raises:
            ValueError: If data format is not supported.
        """
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
        
        def tokenize_function(examples: dict) -> dict:
            """Tokenize text examples for model input."""
            return self.tokenizer(
                examples["text"], 
                truncation=True, 
                padding=False,
                max_length=512
            )
        
        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        
        return train_dataset, eval_dataset
    
    def train(self, train_dataset: HFDataset, eval_dataset: HFDataset, 
              num_epochs: int = 3, batch_size: int = 4, learning_rate: float = 5e-5,
              save_steps: int = 500, eval_steps: int = 500, 
              resume_from_checkpoint: Optional[str] = None) -> Trainer:
        """
        Train the model using the provided datasets.
        
        Args:
            train_dataset: Tokenized training dataset.
            eval_dataset: Tokenized evaluation dataset.
            num_epochs: Number of training epochs.
            batch_size: Batch size for training and evaluation.
            learning_rate: Learning rate for optimization.
            save_steps: Frequency of model checkpointing.
            eval_steps: Frequency of evaluation during training.
            resume_from_checkpoint: Path to checkpoint to resume training from.
        
        Returns:
            Trained Trainer object.
        """
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
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
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
    
    def generate_response(self, instruction: str, max_length: int = 150, 
                         log_to_wandb: bool = False) -> str:
        """
        Generate response for given instruction using the trained model.
        
        This method formats the instruction using Alpaca-style prompting,
        generates a response using the model, and optionally logs metrics to WandB.
        
        Args:
            instruction: Input instruction or question for the model.
            max_length: Maximum length of the generated response.
            log_to_wandb: Whether to log generation metrics to WandB.
        
        Returns:
            Generated response text, cleaned of special tokens and formatting.
        """
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
                # For pre-trained models, use transformers generate method
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
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
