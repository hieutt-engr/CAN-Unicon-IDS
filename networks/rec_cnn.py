import torch
import torch.nn as nn
import torch.nn.functional as F

class RecCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(RecCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        
        # Pooling layer (MaxPooling)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Adaptive pooling to ensure the same spatial size (4x4)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)  # 256 * 4 * 4 = 4096
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Convolutional layers with pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Adaptive pooling to make sure output size is 4x4
        x = self.adaptive_pool(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)  # Flatten: [batch_size, 256 * 4 * 4]
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
