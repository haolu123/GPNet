import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F


class SimpleFNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleFNN, self).__init__()
        # Define the layers and dropout
        self.fc1 = nn.Linear(input_size, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 200)
        self.fc4 = nn.Linear(200, 300)
        self.fc5 = nn.Linear(300, 200)
        self.fc6 = nn.Linear(200, 100)
        self.fc7 = nn.Linear(100, output_size)  # Output layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply layers with ReLU activation and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        x = F.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.fc7(x)  # No activation and no dropout on the output layer
        return x