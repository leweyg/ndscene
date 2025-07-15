import torch
import torch.nn as nn
import torch.nn.functional as F

# Define some simple image tools
class SimpleImages():
    def __init__(self):
        size = 4
        self.target_list = SimpleImages._simpleImageData(size)
        self.target_tensor = torch.tensor( self.target_list ).float()
        self.target_flat = self.target_tensor.flatten()
        self.input_coords = SimpleImages._simpleCoordData(size)
        self.input_tensor = torch.tensor(self.input_coords).float()
        self.input_tensor = self.input_tensor.reshape( [size*size, 2] )
        pass
    @staticmethod
    def _simpleCoordData(size):
        rows = []
        for y in range(size):
            row = []
            rows.append(row)
            for x in range(size):
                col = [ x * 1.0, y * 1.0 ]
                row.append(col)
        return rows
    @staticmethod
    def _simpleImageData(size):
        rows = []
        for y in range(size):
            row = []
            rows.append(row)
            for x in range(size):
                col = (1.0 if (x > y) else 0.0)
                row.append(col)
        return rows

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
    input_dim = 2  # Number of input features
    hidden_dim = 50 # Number of neurons in the hidden layer
    output_dim = 1  # Number of output classes or values

    # Create an instance of the model
    model = SimpleNN(input_dim, hidden_dim, output_dim)
    print("Model structure:")
    print(model)

    # Create dummy input
    img = SimpleImages()
    dummy_input = img.input_tensor[0]

    # Perform a forward pass
    output = model(dummy_input)
    print("\nOutput from the model:")
    print(output)
    print("Output shape:", output.shape)
    