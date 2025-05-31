"""
Improved Neural Network module with PyTorch implementation.
Supports multiple activation functions, weight saving/loading, and normalization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Optional, Union, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class TorchNeuralNetwork(nn.Module):
    """
    Enhanced feedforward neural network with PyTorch.
    
    Features:
    - Output continuo per la sterzata ([-1, 1])
    - Multiple activation functions (ReLU, Sigmoid, Tanh)
    - Optional input normalization
    - Weight initialization strategies
    - Save/load functionality
    - Dropout for regularization
    - Noise injection for robustness
    """
    
    def __init__(
        self,
        topology: List[int],
        activation: str = 'relu',
        input_norm: bool = True,
        dropout_rate: float = 0.0,
        weight_init: str = 'xavier'
    ):
        """
        Initialize neural network.
        
        Args:
            topology: List of layer sizes [input, hidden1, hidden2, ..., output]
            activation: Activation function ('relu', 'sigmoid', 'tanh', 'leaky_relu')
            input_norm: Whether to normalize inputs
            dropout_rate: Dropout probability (0.0 = no dropout)
            weight_init: Weight initialization strategy ('xavier', 'he', 'uniform')
        """
        super().__init__()
        
        if len(topology) < 2:
            raise ValueError("Network must have at least input and output layers")
        
        self.topology = topology
        self.input_norm = input_norm
        self.dropout_rate = dropout_rate
        self.activation_name = activation
        
        # Create layers
        self.layers = nn.ModuleList()
        for i in range(len(topology) - 1):
            self.layers.append(nn.Linear(topology[i], topology[i + 1]))
        
        # Add dropout if specified
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # Set activation function
        self.activation = self._get_activation_function(activation)
        
        # Initialize weights
        self._initialize_weights(weight_init)
        
        # Calculate total weight count for genetic algorithm compatibility
        self.weight_count = sum(
            (topology[i] + 1) * topology[i + 1] 
            for i in range(len(topology) - 1)
        )
        
        logger.debug(f"Created neural network with topology {topology}, "
                    f"activation={activation}, weight_count={self.weight_count}")
    
    def _get_activation_function(self, activation: str) -> callable:
        """Get activation function by name."""
        activations = {
            'relu': F.relu,
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'leaky_relu': lambda x: F.leaky_relu(x, 0.01),
            'swish': lambda x: x * torch.sigmoid(x),
            'elu': F.elu
        }
        
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}. "
                           f"Available: {list(activations.keys())}")
        
        return activations[activation]
    
    def _initialize_weights(self, method: str):
        """Initialize network weights using specified method."""
        for layer in self.layers:
            if method == 'xavier':
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif method == 'he':
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)
            elif method == 'uniform':
                nn.init.uniform_(layer.weight, -1.0, 1.0)
                nn.init.uniform_(layer.bias, -1.0, 1.0)
            else:
                raise ValueError(f"Unknown weight initialization: {method}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Input normalization
        if self.input_norm and x.numel() > 1:
            x = (x - x.mean()) / (x.std() + 1e-8)
        
        # Pass through hidden layers
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            
            # Apply dropout during training
            if self.dropout is not None and self.training:
                x = self.dropout(x)
        
        # Output layer (tanh per output continuo)
        x = self.layers[-1](x)
        x = torch.tanh(x)  # output in [-1, 1]
        
        return x
    
    def process_inputs(
        self, 
        inputs: Union[List[float], np.ndarray], 
        noise_std: float = 0.0
    ) -> np.ndarray:
        """
        Process inputs through the network (compatible with genetic algorithm).
        
        Args:
            inputs: Input values
            noise_std: Standard deviation of noise to add to outputs
            
        Returns:
            Network outputs as numpy array
        """
        if len(inputs) != self.topology[0]:
            raise ValueError(f"Expected {self.topology[0]} inputs, got {len(inputs)}")
        
        # Convert to tensor
        x = torch.tensor(inputs, dtype=torch.float32)
        
        # Forward pass without gradients
        with torch.no_grad():
            self.eval()  # Set to evaluation mode
            outputs = self.forward(x)
            
            # Add noise if specified
            if noise_std > 0.0:
                noise = torch.normal(0, noise_std, size=outputs.shape)
                outputs += noise
        
        return outputs.numpy()
    
    def get_weights_flattened(self) -> np.ndarray:
        """Get all weights and biases as a flattened array."""
        params = []
        for layer in self.layers:
            params.append(layer.weight.data.cpu().numpy().flatten())
            params.append(layer.bias.data.cpu().numpy().flatten())
        
        return np.concatenate(params)
    
    def set_weights_flattened(self, weights: np.ndarray):
        """Set weights and biases from a flattened array."""
        if len(weights) != self.weight_count:
            raise ValueError(f"Expected {self.weight_count} weights, got {len(weights)}")
        
        idx = 0
        for layer in self.layers:
            # Set weights
            w_shape = layer.weight.data.shape
            w_size = w_shape[0] * w_shape[1]
            layer.weight.data = torch.from_numpy(
                weights[idx:idx + w_size].reshape(w_shape)
            ).float()
            idx += w_size
            
            # Set biases
            b_shape = layer.bias.data.shape
            b_size = b_shape[0]
            layer.bias.data = torch.from_numpy(
                weights[idx:idx + b_size].reshape(b_shape)
            ).float()
            idx += b_size
    
    def set_random_weights(self, min_val: float = -1.0, max_val: float = 1.0):
        """Set random weights in specified range."""
        for layer in self.layers:
            nn.init.uniform_(layer.weight, a=min_val, b=max_val)
            nn.init.uniform_(layer.bias, a=min_val, b=max_val)
    
    def save_weights(self, filepath: Union[str, Path]):
        """Save network weights to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'state_dict': self.state_dict(),
            'topology': self.topology,
            'activation': self.activation_name,
            'input_norm': self.input_norm,
            'dropout_rate': self.dropout_rate
        }
        
        torch.save(state, filepath)
        logger.info(f"Saved neural network weights to {filepath}")
    
    def load_weights(self, filepath: Union[str, Path]):
        """Load network weights from file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Weight file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Verify compatibility
        if checkpoint['topology'] != self.topology:
            raise ValueError(f"Topology mismatch. Expected {self.topology}, "
                           f"got {checkpoint['topology']}")
        
        self.load_state_dict(checkpoint['state_dict'])
        logger.info(f"Loaded neural network weights from {filepath}")
    
    def clone(self) -> 'TorchNeuralNetwork':
        """Create a deep copy of this network."""
        clone = TorchNeuralNetwork(
            topology=self.topology.copy(),
            activation=self.activation_name,
            input_norm=self.input_norm,
            dropout_rate=self.dropout_rate
        )
        
        # Copy weights
        clone.load_state_dict(self.state_dict())
        
        return clone
    
    def get_layer_activations(self, inputs: Union[List[float], np.ndarray]) -> List[np.ndarray]:
        """Get activations from all layers (for debugging/analysis)."""
        activations = []
        
        x = torch.tensor(inputs, dtype=torch.float32)
        
        with torch.no_grad():
            self.eval()
            
            # Input normalization
            if self.input_norm and x.numel() > 1:
                x = (x - x.mean()) / (x.std() + 1e-8)
            
            activations.append(x.numpy())
            
            # Pass through layers
            for layer in self.layers:
                x = layer(x)
                activations.append(x.numpy())
                
                # Apply activation (except for last layer)
                if layer != self.layers[-1]:
                    x = self.activation(x)
        
        return activations
    
    def __str__(self) -> str:
        """String representation of the network."""
        return (f"TorchNeuralNetwork(topology={self.topology}, "
                f"activation={self.activation_name}, "
                f"input_norm={self.input_norm}, "
                f"dropout={self.dropout_rate})")


# Legacy compatibility class
class NeuralNetwork:
    """Legacy wrapper for backward compatibility."""
    
    def __init__(self, topology: List[int]):
        logger.warning("NeuralNetwork is deprecated. Use TorchNeuralNetwork instead.")
        self.network = TorchNeuralNetwork(topology, activation='sigmoid', input_norm=False)
        self.topology = topology
        self.weight_count = self.network.weight_count
    
    def process_inputs(self, inputs: List[float]) -> np.ndarray:
        return self.network.process_inputs(inputs)
    
    def get_weights_flattened(self) -> np.ndarray:
        return self.network.get_weights_flattened()
    
    def set_weights_flattened(self, weights: np.ndarray):
        self.network.set_weights_flattened(weights)
    
    def set_random_weights(self, min_val: float, max_val: float):
        self.network.set_random_weights(min_val, max_val)


if __name__ == "__main__":
    # Test the improved neural network
    import time
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create network
    topology = [5, 8, 6, 3]
    nn = TorchNeuralNetwork(
        topology=topology,
        activation='relu',
        input_norm=True,
        dropout_rate=0.1
    )
    
    print(f"Network: {nn}")
    print(f"Weight count: {nn.weight_count}")
    
    # Test processing
    inputs = np.random.randn(5)
    outputs = nn.process_inputs(inputs)
    print(f"Inputs: {inputs}")
    print(f"Outputs: {outputs}")
    
    # Test weight operations
    weights = nn.get_weights_flattened()
    print(f"Flattened weights shape: {weights.shape}")
    
    # Test cloning
    clone = nn.clone()
    clone_outputs = clone.process_inputs(inputs)
    print(f"Clone outputs match: {np.allclose(outputs, clone_outputs)}")
    
    # Performance test
    start_time = time.time()
    for _ in range(1000):
        _ = nn.process_inputs(inputs)
    end_time = time.time()
    
    print(f"1000 forward passes took {end_time - start_time:.4f} seconds")
    
    # Test save/load
    nn.save_weights("test_weights.pth")
    
    new_nn = TorchNeuralNetwork(topology, activation='relu', input_norm=True)
    new_nn.load_weights("test_weights.pth")
    
    new_outputs = new_nn.process_inputs(inputs)
    print(f"Loaded network outputs match: {np.allclose(outputs, new_outputs)}")
