"""
LSTM model architectures for stock price prediction.

This module provides various LSTM-based neural network architectures
optimized for time series forecasting, specifically stock price prediction.
Includes basic LSTM, bidirectional LSTM, and attention-enhanced models.

Author: Time Series Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math
import logging

logger = logging.getLogger(__name__)

# Model architecture constants
MIN_HIDDEN_SIZE = 8
MAX_HIDDEN_SIZE = 2048
MIN_LAYERS = 1
MAX_LAYERS = 10
MIN_DROPOUT = 0.0
MAX_DROPOUT = 0.9


class BasicLSTM(nn.Module):
    """
    Basic LSTM model for stock price prediction.
    
    This class implements a standard LSTM architecture with configurable
    layers, dropout, and bidirectional support. Suitable for most time
    series forecasting tasks.
    
    Attributes:
        input_size (int): Number of input features per timestep
        hidden_size (int): Number of hidden units in LSTM layers
        num_layers (int): Number of stacked LSTM layers
        output_size (int): Number of output predictions
        bidirectional (bool): Whether to use bidirectional LSTM
        dropout_rate (float): Dropout probability for regularization
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 bidirectional: bool = False) -> None:
        """
        Initialize BasicLSTM model.
        
        Args:
            input_size: Number of input features per timestep
            hidden_size: Number of hidden units in LSTM layers
            num_layers: Number of stacked LSTM layers  
            output_size: Number of output predictions (default: 1)
            dropout: Dropout probability for regularization (default: 0.2)
            bidirectional: Whether to use bidirectional LSTM (default: False)
            
        Raises:
            ValueError: If any parameter is outside valid range
        """
        super(BasicLSTM, self).__init__()
        
        # Validate parameters
        self._validate_parameters(input_size, hidden_size, num_layers, dropout)
        
        # Store configuration
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.dropout_rate = dropout
        
        # Build LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Build fully connected layers
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc1 = nn.Linear(lstm_output_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized BasicLSTM with {self.count_parameters():,} parameters")
    
    def _validate_parameters(self, input_size: int, hidden_size: int, 
                           num_layers: int, dropout: float) -> None:
        """
        Validate model parameters.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            
        Raises:
            ValueError: If any parameter is invalid
        """
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError(f"input_size must be a positive integer, got {input_size}")
        
        if not MIN_HIDDEN_SIZE <= hidden_size <= MAX_HIDDEN_SIZE:
            raise ValueError(f"hidden_size must be between {MIN_HIDDEN_SIZE} and {MAX_HIDDEN_SIZE}")
        
        if not MIN_LAYERS <= num_layers <= MAX_LAYERS:
            raise ValueError(f"num_layers must be between {MIN_LAYERS} and {MAX_LAYERS}")
        
        if not MIN_DROPOUT <= dropout <= MAX_DROPOUT:
            raise ValueError(f"dropout must be between {MIN_DROPOUT} and {MAX_DROPOUT}")
    
    def _init_weights(self) -> None:
        """
        Initialize model weights using appropriate schemes.
        
        Uses Xavier uniform initialization for input weights,
        orthogonal initialization for hidden weights, and
        zero initialization for biases.
        """
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        # Initialize fully connected layers
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, output_size)
            
        Raises:
            ValueError: If input tensor has incorrect shape
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor, got {x.dim()}D")
        
        batch_size, seq_len, features = x.shape
        if features != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {features}")
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Extract final output based on architecture
        if self.bidirectional:
            # Concatenate the last output from both directions
            last_output = torch.cat((
                lstm_out[:, -1, :self.hidden_size], 
                lstm_out[:, 0, self.hidden_size:]
            ), dim=1)
        else:
            last_output = lstm_out[:, -1, :]
        
        # Apply fully connected layers
        out = F.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
    def count_parameters(self) -> int:
        """
        Count the total number of trainable parameters.
        
        Returns:
            int: Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AttentionLSTM(nn.Module):
    """LSTM model with attention mechanism."""
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 attention_size: Optional[int] = None) -> None:
        """
        Initialize AttentionLSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            output_size: Number of output features
            dropout: Dropout probability
            attention_size: Size of attention mechanism
        """
        super(AttentionLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.attention_size = attention_size or hidden_size
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention_linear = nn.Linear(hidden_size, self.attention_size)
        self.attention_vector = nn.Parameter(torch.randn(self.attention_size))
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        nn.init.xavier_uniform_(self.attention_linear.weight)
        self.attention_linear.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.attention_vector.unsqueeze(0))
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
    
    def attention_mechanism(self, lstm_outputs: torch.Tensor) -> torch.Tensor:
        """
        Apply attention mechanism to LSTM outputs.
        
        Args:
            lstm_outputs: LSTM outputs of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Attended output of shape (batch_size, hidden_size)
        """
        # Compute attention scores
        attention_scores = torch.tanh(self.attention_linear(lstm_outputs))
        attention_scores = torch.matmul(attention_scores, self.attention_vector)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention weights
        attended_output = torch.sum(lstm_outputs * attention_weights.unsqueeze(2), dim=1)
        
        return attended_output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attended_output = self.attention_mechanism(lstm_out)
        
        # Fully connected layers
        out = F.relu(self.fc1(attended_output))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class CNN_LSTM(nn.Module):
    """Hybrid CNN-LSTM model for stock price prediction."""
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 cnn_channels: int = 64,
                 kernel_size: int = 3):
        """
        Initialize CNN-LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            output_size: Number of output features
            dropout: Dropout probability
            cnn_channels: Number of CNN channels
            kernel_size: CNN kernel size
        """
        super(CNN_LSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # CNN layers
        self.conv1 = nn.Conv1d(input_size, cnn_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size, padding=kernel_size//2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        
        # Initialize bias if it exists
        if self.conv1.bias is not None:
            self.conv1.bias.data.fill_(0)
        if self.conv2.bias is not None:
            self.conv2.bias.data.fill_(0)
        
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN then LSTM.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Reshape for CNN: (batch_size, input_size, sequence_length)
        x = x.transpose(1, 2)
        
        # CNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Reshape back for LSTM: (batch_size, sequence_length, cnn_channels)
        x = x.transpose(1, 2)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use the last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = F.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class MultiHeadAttentionLSTM(nn.Module):
    """LSTM model with multi-head attention mechanism."""
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 num_heads: int = 4):
        """
        Initialize Multi-Head Attention LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            output_size: Number of output features
            dropout: Dropout probability
            num_heads: Number of attention heads
        """
        super(MultiHeadAttentionLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_heads = num_heads
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Multi-head attention
        attn_out, _ = self.multihead_attn(lstm_out, lstm_out, lstm_out)
        
        # Residual connection and layer normalization
        attn_out = self.layer_norm(lstm_out + attn_out)
        
        # Use the last output
        last_output = attn_out[:, -1, :]
        
        # Fully connected layers
        out = F.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


def create_model(model_config) -> nn.Module:
    """
    Create a model based on configuration.
    
    Args:
        model_config: Model configuration object
        
    Returns:
        Initialized model
    """
    model_type = model_config.model_type.lower()
    
    if model_type == "lstm":
        if model_config.use_attention:
            return AttentionLSTM(
                input_size=model_config.input_size,
                hidden_size=model_config.hidden_size,
                num_layers=model_config.num_layers,
                output_size=model_config.output_size,
                dropout=model_config.dropout
            )
        else:
            return BasicLSTM(
                input_size=model_config.input_size,
                hidden_size=model_config.hidden_size,
                num_layers=model_config.num_layers,
                output_size=model_config.output_size,
                dropout=model_config.dropout,
                bidirectional=model_config.bidirectional
            )
    elif model_type == "cnn_lstm":
        return CNN_LSTM(
            input_size=model_config.input_size,
            hidden_size=model_config.hidden_size,
            num_layers=model_config.num_layers,
            output_size=model_config.output_size,
            dropout=model_config.dropout
        )
    elif model_type == "multihead_lstm":
        return MultiHeadAttentionLSTM(
            input_size=model_config.input_size,
            hidden_size=model_config.hidden_size,
            num_layers=model_config.num_layers,
            output_size=model_config.output_size,
            dropout=model_config.dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
