import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.relu = nn.ReLU()                         # Activation function
        self.fc2 = nn.Linear(hidden_size, output_size) # Second fully connected layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Example usage:
if __name__ == "__main__":
    input_dim = 10  # Number of input features
    hidden_dim = 50 # Number of neurons in the hidden layer
    output_dim = 2  # Number of output classes or values

    # Create an instance of the model
    model = SimpleNN(input_dim, hidden_dim, output_dim)
    print("Model structure:")
    print(model)

    # Create a dummy input tensor
    dummy_input = torch.randn(1, input_dim) # Batch size of 1

    # Perform a forward pass
    output = model(dummy_input)
    print("\nOutput from the model:")
    print(output)
    print("Output shape:", output.shape)
    