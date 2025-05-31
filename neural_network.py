import numpy as np

class NeuralLayer:
    def __init__(self, input_count, neuron_count):
        # Initialize weights with small random values (including bias)
        self.weights = np.random.uniform(-1.0, 1.0, (input_count + 1, neuron_count))
        
    def process_inputs(self, inputs):
        # Add bias input (always 1.0)
        biased_inputs = np.append(inputs, 1.0)
        
        # Calculate weighted sum
        sums = np.dot(biased_inputs, self.weights)
        
        # Apply activation function (sigmoid)
        return self.sigmoid(sums)
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def set_random_weights(self, min_value, max_value):
        self.weights = np.random.uniform(min_value, max_value, self.weights.shape)
    
    def deep_copy(self):
        new_layer = NeuralLayer(self.weights.shape[0] - 1, self.weights.shape[1])
        new_layer.weights = self.weights.copy()
        return new_layer


class NeuralNetwork:
    def __init__(self, topology):
        """
        Initialize a neural network with the given topology.
        
        Args:
            topology: List of integers representing the number of neurons in each layer
                     (including input and output layers)
        """
        self.topology = topology
        self.layers = []
        
        # Create layers based on topology
        for i in range(len(topology) - 1):
            self.layers.append(NeuralLayer(topology[i], topology[i + 1]))
        
        # Calculate total weight count for genetic algorithm
        self.weight_count = sum((topology[i] + 1) * topology[i + 1] for i in range(len(topology) - 1))
    
    def process_inputs(self, inputs):
        """Process inputs through the network and return outputs."""
        if len(inputs) != self.topology[0]:
            raise ValueError("Input count does not match network input size")
        
        outputs = inputs
        for layer in self.layers:
            outputs = layer.process_inputs(outputs)
        
        return outputs
    
    def set_random_weights(self, min_value, max_value):
        """Set all weights to random values within the specified range."""
        for layer in self.layers:
            layer.set_random_weights(min_value, max_value)
    
    def get_weights_flattened(self):
        """Return all weights as a flattened array for genetic algorithm."""
        weights = []
        for layer in self.layers:
            weights.extend(layer.weights.flatten())
        return np.array(weights)
    
    def set_weights_flattened(self, flattened_weights):
        """Set weights from a flattened array (used by genetic algorithm)."""
        index = 0
        for layer in self.layers:
            size = layer.weights.size
            layer.weights = flattened_weights[index:index+size].reshape(layer.weights.shape)
            index += size
    
    def deep_copy(self):
        """Create a deep copy of this neural network."""
        new_net = NeuralNetwork(self.topology)
        for i, layer in enumerate(self.layers):
            new_net.layers[i] = layer.deep_copy()
        return new_net


# Test the neural network
if __name__ == "__main__":
    # Create a neural network with topology [5, 4, 3, 2]
    nn = NeuralNetwork([5, 4, 3, 2])
    
    # Set random weights
    nn.set_random_weights(-1.0, 1.0)
    
    # Test with random inputs
    inputs = np.random.random(5)
    outputs = nn.process_inputs(inputs)
    
    print(f"Inputs: {inputs}")
    print(f"Outputs: {outputs}")
    print(f"Total weights: {nn.weight_count}")
