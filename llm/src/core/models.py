"""
Custom model architectures for building transformer models from scratch.

This module implements a custom GPT-style transformer architecture with multi-head
attention, position embeddings, and causal language modeling capabilities. It provides
both the configuration class and the complete model implementation compatible with
the HuggingFace transformers library.
"""

import math
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput


class CustomGPTConfig(PretrainedConfig):
    """
    Configuration class for custom GPT model.
    
    This class stores all the configuration parameters needed to define a custom
    GPT-style transformer model, including architecture dimensions, vocabulary size,
    and training hyperparameters.
    
    Attributes:
        vocab_size: Size of the vocabulary.
        n_embd: Dimensionality of the embeddings and hidden states.
        n_layer: Number of transformer layers.
        n_head: Number of attention heads in each layer.
        max_sequence_length: Maximum sequence length for position embeddings.
        dropout: Dropout probability for regularization.
    """
    
    model_type = "custom_gpt"
    
    def __init__(
        self,
        vocab_size: int = 50257,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        max_sequence_length: int = 1024,
        dropout: float = 0.1,
        **kwargs
    ) -> None:
        """
        Initialize the configuration.
        
        Args:
            vocab_size: Size of the vocabulary.
            n_embd: Dimensionality of the embeddings and hidden states.
            n_layer: Number of transformer layers.
            n_head: Number of attention heads in each layer.
            max_sequence_length: Maximum sequence length for position embeddings.
            dropout: Dropout probability for regularization.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.max_sequence_length = max_sequence_length
        self.dropout = dropout


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism with causal masking.
    
    This implementation follows the scaled dot-product attention mechanism
    from "Attention is All You Need" with causal masking for autoregressive
    language modeling.
    
    Attributes:
        n_embd: Embedding dimension.
        n_head: Number of attention heads.
        head_dim: Dimension per attention head.
        qkv: Linear layer for query, key, value projections.
        proj: Output projection layer.
        dropout: Dropout layer for attention weights.
        causal_mask: Lower triangular mask for causal attention.
    """
    
    def __init__(self, config: CustomGPTConfig) -> None:
        """
        Initialize the multi-head attention layer.
        
        Args:
            config: Model configuration containing architecture parameters.
        
        Raises:
            AssertionError: If n_embd is not divisible by n_head.
        """
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = self.n_embd // self.n_head
        
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.proj = nn.Linear(self.n_embd, self.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        # Causal mask - register as buffer so it moves with model device
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_sequence_length, config.max_sequence_length))
            .view(1, 1, config.max_sequence_length, config.max_sequence_length)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_embd).
        
        Returns:
            Output tensor of shape (batch_size, sequence_length, n_embd).
        """
        B, T, C = x.size()  # batch_size, sequence_length, n_embd
        
        # Calculate Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, n_embd)
        
        # Final projection
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    This implements the feed-forward component of a transformer block,
    consisting of two linear transformations with a GELU activation
    and dropout for regularization.
    
    Attributes:
        net: Sequential network with linear layers, activation, and dropout.
    """
    
    def __init__(self, config: CustomGPTConfig) -> None:
        """
        Initialize the feed-forward network.
        
        Args:
            config: Model configuration containing architecture parameters.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_embd).
        
        Returns:
            Output tensor of the same shape as input.
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and feed-forward components.
    
    This implements a single transformer layer using pre-norm architecture
    with residual connections around both the attention and feed-forward
    sub-layers.
    
    Attributes:
        ln1: Layer normalization before attention.
        attn: Multi-head self-attention layer.
        ln2: Layer normalization before feed-forward.
        ffn: Feed-forward network.
    """
    
    def __init__(self, config: CustomGPTConfig) -> None:
        """
        Initialize the transformer block.
        
        Args:
            config: Model configuration containing architecture parameters.
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_embd).
        
        Returns:
            Output tensor of the same shape as input.
        """
        # Pre-norm architecture with residual connections
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class CustomGPTModel(PreTrainedModel):
    """
    Custom GPT model built from scratch with transformer architecture.
    
    This model implements a GPT-style autoregressive transformer for causal
    language modeling. It includes token and position embeddings, multiple
    transformer layers, and a language modeling head for next-token prediction.
    
    The model is compatible with HuggingFace's training infrastructure and
    supports standard transformer operations like training and generation.
    
    Attributes:
        config: Model configuration object.
        token_embedding: Token embedding layer.
        position_embedding: Position embedding layer.
        dropout: Dropout layer for embeddings.
        blocks: List of transformer blocks.
        ln_f: Final layer normalization.
        lm_head: Language modeling head for token prediction.
    """
    
    config_class = CustomGPTConfig
    
    def __init__(self, config: CustomGPTConfig) -> None:
        """
        Initialize the custom GPT model.
        
        Args:
            config: Configuration object containing model architecture parameters.
        """
        super().__init__(config)
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final layer norm and language modeling head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        print(f"Custom GPT model initialized:")
        print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"  Layers: {config.n_layer}")
        print(f"  Embedding dim: {config.n_embd}")
        print(f"  Attention heads: {config.n_head}")
    
    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize model weights using standard transformer initialization.
        
        This method applies proper weight initialization for different layer types:
        - Linear layers: Normal distribution with std=0.02
        - Embedding layers: Normal distribution with std=0.02  
        - LayerNorm: Zero bias, unit weight
        
        Args:
            module: PyTorch module to initialize.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None) -> CausalLMOutput:
        """
        Forward pass through the model for causal language modeling.
        
        Args:
            input_ids: Token indices of shape (batch_size, sequence_length).
            attention_mask: Attention mask (currently unused in this implementation).
            labels: Target token indices for loss computation. If provided, loss
                   will be computed using teacher forcing.
        
        Returns:
            CausalLMOutput containing logits and optional loss.
        """
        B, T = input_ids.size()
        
        # Token embeddings
        token_emb = self.token_embedding(input_ids)  # (B, T, n_embd)
        
        # Position embeddings
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)  # (1, T)
        pos_emb = self.position_embedding(pos)  # (1, T, n_embd)
        
        # Combine embeddings
        x = self.dropout(token_emb + pos_emb)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        loss = None
        if labels is not None:
            # Shift labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        # Return in format expected by HuggingFace Trainer
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 50, 
                temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """
        Simple generation method for autoregressive text generation.
        
        This method generates text token by token using top-k sampling.
        It's a basic implementation suitable for inference and testing.
        
        Args:
            input_ids: Input token sequence of shape (batch_size, sequence_length).
            max_length: Maximum number of tokens to generate.
            temperature: Sampling temperature for controlling randomness.
            top_k: Number of top tokens to consider for sampling.
        
        Returns:
            Generated token sequence including the input prompt.
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self(input_ids)
                logits = outputs.logits
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k sampling
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if we reach max length
                if input_ids.size(1) >= max_length:
                    break
        
        return input_ids


def create_custom_model(config: Any) -> CustomGPTModel:
    """
    Factory function to create a custom model instance.
    
    This function creates a CustomGPTModel with the specified configuration
    parameters, handling the conversion from the training configuration format
    to the model configuration format.
    
    Args:
        config: Configuration object containing model architecture parameters.
    
    Returns:
        Initialized CustomGPTModel instance.
    """
    model_config = CustomGPTConfig(
        vocab_size=config.vocab_size,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        max_sequence_length=config.max_sequence_length,
        dropout=config.dropout
    )
    
    return CustomGPTModel(model_config)
