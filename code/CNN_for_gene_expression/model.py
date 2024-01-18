import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, class_num):
        super(CustomCNN, self).__init__()
        
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        # Max Pooling Layer 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        # Max Pooling Layer 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(32 * 16 * 20, 64)  # Adjust the input features to match your image size
        
        # Fully Connected Layer class_num (Output layer)
        self.fc2 = nn.Linear(64, class_num)

    def forward(self, x):
        # Apply Convolutional layer 1 followed by ReLU and Max Pooling
        x = self.pool1(F.relu(self.conv1(x)))
        
        # Apply Convolutional layer 2 followed by ReLU and Max Pooling
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Apply Convolutional layer 3 followed by ReLU
        x = F.relu(self.conv3(x))
        
        # Flatten the output for the fully connected layer
        x = torch.flatten(x, 1)
        
        # Apply Fully Connected layer 1 followed by ReLU
        x = F.relu(self.fc1(x))
        
        # Apply Fully Connected layer 2 (Output layer)
        x = self.fc2(x)
        
        return x

# # Create the CNN model instance
# model = CustomCNN()
# print(model)