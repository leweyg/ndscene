import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio

RESULTS_PATH="tests/.results/"

# Define some simple image tools
class SimpleImages():
    def __init__(self):
        exImg = SimpleImages._loadExampleImage()
        size = exImg.shape[0]
        self.target_list = exImg
        self.target_tensor = torch.tensor( self.target_list ).float() / 255.0
        if True:
            mx,_mxInds = self.target_tensor.max(0)
            mx,_mxInds = mx.max(0)
            print("Max1:", mx.shape, mx )
        self.target_tensor = self.target_tensor.reshape( [size*size, 4] )
        SimpleImages.saveResult(self.target_tensor, RESULTS_PATH + "target.png")
        self.input_coords = SimpleImages._simpleCoordData(size)
        self.input_tensor = torch.tensor(self.input_coords).float()
        self.input_tensor = self.input_tensor.reshape( [size*size, 2] )
        pass
    @staticmethod
    def saveResult(result, path):
        assert(result.shape[-1] == 4)
        if (len(result.shape) == 2):
            import math
            side = int(math.sqrt( result.shape[0] ))
            result = result.reshape( [side,side,4] )
        result = ( result * 255.0 ).byte().cpu().numpy()
        print("Result:", result.shape, result.dtype)
        print("Writing to:", path)
        imageio.v3.imwrite( path, result )
    @staticmethod
    def _loadExampleImage():
        import os
        cwd = os.getcwd()
        print("cwd=", cwd)
        info = imageio.v3.imread("tests/test_image.png")
        print("InfoShape=", info.shape)
        return info
    @staticmethod
    def _simpleCoordData(size):
        rows = []
        invSize = 1.0 / float(size-1)
        for y in range(size):
            row = []
            rows.append(row)
            for x in range(size):
                col = [ invSize * x, invSize * y ]
                row.append(col)
        return rows

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.relu = nn.ReLU()                         # Activation function
        self.ml1 = nn.Linear(hidden_size, hidden_size)
        self.mr1 = nn.ReLU()
        self.ml2 = nn.Linear(hidden_size, hidden_size)
        self.mr2 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size) # Second fully connected layer

    def forward(self, x):
        x = self.fc1(x)
        #x = self.relu(x)
        x = self.ml1(x)
        x = self.mr1(x)
        #x = self.ml2(x)
        #x = self.mr2(x)
        x = self.fc2(x)
        return x
    
    def printParameters(self):
        params = [ self.fc1, self.relu, self.fc2 ]
        for p in params:
            if (isinstance(p,nn.Linear)):
                print("Model Param", p.weight, p.bias)
            else:
                print("ModelParam:", p)
        #print("Model Parameters:", self.fc1.weight, self.fc1.bias)

# Simple optimization loop:
class SimpleSolverLoop:
    def __init__(self, model):
        self.model = model
        self.criterion = nn.MSELoss()
        #self.optimizer = torch.optim.Rprop(model.parameters()) #, lr=0.05)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        pass

    def loss_as_out_minus_target(self, sampleOut, targetOut):
        #print("sampleOut.shape=", sampleOut.shape)
        #print("targetOut.shape=", targetOut.shape)
        loss = sampleOut - targetOut
        loss = ( loss * loss ).sum()
        return loss

    def solve_step(self, sampleIn, targetOut, prevLossItem=0, showInfo=False):
        sampleOut = self.model(sampleIn)
        loss = self.loss_as_out_minus_target(sampleOut, targetOut); # self.criterion(y_pred, sampleOutput)
        lossItem = loss.item()
        if (showInfo):
            print("Loss:", lossItem, lossItem - prevLossItem)
        # Zero gradients, perform a backward pass, and update the weights.
        self.optimizer.zero_grad()
        debugCreateGraph = False;
        loss.backward(create_graph=debugCreateGraph)
        self.optimizer.step()
        return lossItem

# Example usage:
if __name__ == "__main__":
    input_dim = 2  # Number of input features
    hidden_dim = 128 # Number of neurons in the hidden layer
    output_dim = 4  # Number of output classes or values

    # Create an instance of the model
    model = SimpleNN(input_dim, hidden_dim, output_dim)
    print("Model structure:")
    print(model)

    # Create dummy input
    img = SimpleImages()
    if True:
        # test model at first:
        dummy_input = img.input_tensor
        output = model(dummy_input)
        SimpleImages.saveResult(output, RESULTS_PATH + "result_before.png")

    print("Building solver loop...")
    solver = SimpleSolverLoop(model)
    solverSteps = 300
    prevLossItem = 0
    for i in range(solverSteps):
        showInfo = False
        if ((i%10)==0):
            showInfo = True
            print("Step:", i)
        prevLossItem = solver.solve_step(img.input_tensor, img.target_tensor, prevLossItem, showInfo=showInfo)

    if True:
        # test model at last:
        dummy_input = img.input_tensor
        output = model(dummy_input)
        SimpleImages.saveResult(output, RESULTS_PATH + "result_after.png")

    #model.printParameters();

    print("Done.")
    