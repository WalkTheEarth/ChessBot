import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessNet(nn.Module):
    """Convolutional Neural Network for chess position evaluation"""
    
    def __init__(self):
        super(ChessNet, self).__init__()
        
        # Convolutional layers for spatial features
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)  # Q-value output
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x shape: (batch, 12, 8, 8)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
