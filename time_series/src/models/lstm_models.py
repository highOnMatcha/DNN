"""LSTM model architectures for stock price prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class BasicLSTM(nn.Module):
    """Basic LSTM model for stock price prediction."""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 bidirectional: bool = False):
        super(BasicLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Fully connected layers
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc1 = nn.Linear(lstm_output_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # Initialize weights
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
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output
        if self.bidirectional:
            # Concatenate the last output from both directions
            last_output = torch.cat((lstm_out[:, -1, :self.hidden_size], 
                                   lstm_out[:, 0, self.hidden_size:]), dim=1)
        else:
            last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = F.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class AttentionLSTM(nn.Module):
    """LSTM model with attention mechanism."""
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 attention_size: int = None):
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
        self.conv1.bias.data.fill_(0)
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
