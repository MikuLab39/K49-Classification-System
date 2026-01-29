import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    [Production Ver] 
    """
    def __init__(self, num_classes=49):
        super(SimpleCNN, self).__init__()
        
        # Layer 1: 28x28 -> 26x26 -> 13x13
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Layer 2: 13x13 -> 11x11 -> 5x5
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # FC Layers: Input 64 * 5 * 5
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
