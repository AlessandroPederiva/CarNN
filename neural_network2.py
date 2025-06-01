"""
Recurrent Neural Network module with PyTorch implementation.
Supports LSTM/GRU cells, attention mechanism, and weight saving/loading.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Optional, Union, Tuple
from pathlib import Path
from collections import deque

logger = logging.getLogger(__name__)

class MemoryBuffer:
    """Buffer for storing and managing temporal sequences for recurrent networks."""
    def __init__(self, buffer_size: int = 10):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.is_initialized = False
    
    def add(self, observation: np.ndarray):
        self.buffer.append(observation)
        self.is_initialized = True
    
    def get_sequence(self) -> np.ndarray:
        if not self.is_initialized:
            return np.array([])
        return np.array(self.buffer)
    
    def clear(self):
        self.buffer.clear()
        self.is_initialized = False
    
    def is_ready(self) -> bool:
        return len(self.buffer) > 0
    
    def __len__(self) -> int:
        return len(self.buffer)

class RecurrentNeuralNetwork(nn.Module):
    """
    Enhanced neural network with recurrent capabilities (LSTM/GRU).
    """
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        recurrent_type: str = 'lstm',
        num_recurrent_layers: int = 1,
        sequence_length: int = 5,
        use_attention: bool = False,
        dropout_rate: float = 0.0,
        bidirectional: bool = False
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.recurrent_type = recurrent_type.lower()
        self.num_recurrent_layers = num_recurrent_layers
        self.sequence_length = sequence_length
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        self.memory_buffer = MemoryBuffer(sequence_length)
        
        if self.recurrent_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_sizes[0],
                num_layers=num_recurrent_layers,
                batch_first=True,
                dropout=dropout_rate if num_recurrent_layers > 1 else 0,
                bidirectional=bidirectional
            )
        elif self.recurrent_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_sizes[0],
                num_layers=num_recurrent_layers,
                batch_first=True,
                dropout=dropout_rate if num_recurrent_layers > 1 else 0,
                bidirectional=bidirectional
            )
        else:
            raise ValueError(f"Unsupported recurrent type: {recurrent_type}")
        
        if use_attention:
            self.attention = nn.Linear(hidden_sizes[0] * self.directions, 1)
        
        fc_layers = []
        fc_input_size = hidden_sizes[0] * self.directions
        for i in range(1, len(hidden_sizes)):
            fc_layers.append(nn.Linear(fc_input_size, hidden_sizes[i]))
            fc_layers.append(nn.ReLU())
            if dropout_rate > 0:
                fc_layers.append(nn.Dropout(dropout_rate))
            fc_input_size = hidden_sizes[i]
        
        fc_layers.append(nn.Linear(fc_input_size, output_size))
        fc_layers.append(nn.Tanh())
        self.fc_layers = nn.Sequential(*fc_layers)
        self.hidden = None
        self.weight_count = self._calculate_weight_count()
        logger.debug(f"Created recurrent neural network: {self}")
    
    def _calculate_weight_count(self) -> int:
        count = 0
        for name, param in self.rnn.named_parameters():
            count += param.numel()
        if self.use_attention:
            for name, param in self.attention.named_parameters():
                count += param.numel()
        for name, param in self.fc_layers.named_parameters():
            count += param.numel()
        return count
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.recurrent_type == 'lstm':
            rnn_out, (h_n, c_n) = self.rnn(x)
        else:
            rnn_out, h_n = self.rnn(x)
        
        if self.use_attention:
            attention_weights = F.softmax(self.attention(rnn_out), dim=1)
            context = torch.sum(attention_weights * rnn_out, dim=1)
        else:
            context = rnn_out[:, -1, :]
        
        output = self.fc_layers(context)
        return output
    
    def init_hidden(self, batch_size: int = 1):
        weight = next(self.parameters()).data
        hidden_size = self.hidden_sizes[0]
        if self.recurrent_type == 'lstm':
            self.hidden = (
                weight.new(self.num_recurrent_layers * self.directions, batch_size, hidden_size).zero_(),
                weight.new(self.num_recurrent_layers * self.directions, batch_size, hidden_size).zero_()
            )
        else:
            self.hidden = weight.new(self.num_recurrent_layers * self.directions, batch_size, hidden_size).zero_()
    
    def reset_memory(self):
        self.memory_buffer.clear()
        self.hidden = None
    
    def process_inputs(
        self, 
        inputs: Union[List[float], np.ndarray], 
        noise_std: float = 0.0
    ) -> np.ndarray:
        if len(inputs) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(inputs)}")
        
        self.memory_buffer.add(np.array(inputs, dtype=np.float32))
        
        if len(self.memory_buffer) < 1:
            return np.zeros(self.output_size)
        
        if len(self.memory_buffer) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(self.memory_buffer), self.input_size))
            sequence = np.vstack([padding, self.memory_buffer.get_sequence()])
        else:
            sequence = self.memory_buffer.get_sequence()[-self.sequence_length:]
        
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            self.eval()
            outputs = self.forward(x)
            
            if noise_std > 0.0:
                noise = torch.normal(0, noise_std, size=outputs.shape)
                outputs += noise
        
        return outputs.squeeze(0).numpy()
    
    def get_weights_flattened(self) -> np.ndarray:
        params = []
        for param in self.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)
    
    def set_weights_flattened(self, weights: np.ndarray):
        if len(weights) != self.weight_count:
            raise ValueError(f"Expected {self.weight_count} weights, got {len(weights)}")
        
        idx = 0
        for param in self.parameters():
            param_size = param.numel()
            param_shape = param.data.shape
            param.data = torch.from_numpy(
                weights[idx:idx + param_size].reshape(param_shape)
            ).float()
            idx += param_size
    
    def set_random_weights(self, min_val: float = -1.0, max_val: float = 1.0):
        for param in self.parameters():
            param.data.uniform_(min_val, max_val)
    
    def save_weights(self, filepath: Union[str, Path]):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'state_dict': self.state_dict(),
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'recurrent_type': self.recurrent_type,
            'num_recurrent_layers': self.num_recurrent_layers,
            'sequence_length': self.sequence_length,
            'use_attention': self.use_attention,
            'dropout_rate': self.dropout_rate,
            'bidirectional': self.bidirectional
        }
        
        torch.save(state, filepath)
        logger.info(f"Saved recurrent neural network weights to {filepath}")
    
    def load_weights(self, filepath: Union[str, Path]):
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Weight file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location='cpu')
        
        if checkpoint['input_size'] != self.input_size:
            raise ValueError(f"Input size mismatch. Expected {self.input_size}, got {checkpoint['input_size']}")
        if checkpoint['output_size'] != self.output_size:
            raise ValueError(f"Output size mismatch. Expected {self.output_size}, got {checkpoint['output_size']}")
        
        self.load_state_dict(checkpoint['state_dict'])
        logger.info(f"Loaded recurrent neural network weights from {filepath}")
    
    def clone(self) -> 'RecurrentNeuralNetwork':
        clone = RecurrentNeuralNetwork(
            input_size=self.input_size,
            hidden_sizes=self.hidden_sizes.copy(),
            output_size=self.output_size,
            recurrent_type=self.recurrent_type,
            num_recurrent_layers=self.num_recurrent_layers,
            sequence_length=self.sequence_length,
            use_attention=self.use_attention,
            dropout_rate=self.dropout_rate,
            bidirectional=self.bidirectional
        )
        
        clone.load_state_dict(self.state_dict())
        return clone
    
    def __str__(self) -> str:
        return (f"RecurrentNeuralNetwork(input_size={self.input_size}, "
                f"hidden_sizes={self.hidden_sizes}, "
                f"output_size={self.output_size}, "
                f"recurrent_type={self.recurrent_type}, "
                f"num_layers={self.num_recurrent_layers}, "
                f"sequence_length={self.sequence_length}, "
                f"attention={self.use_attention}, "
                f"bidirectional={self.bidirectional})")
