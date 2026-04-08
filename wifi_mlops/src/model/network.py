import torch
import torch.nn as nn

class WiFiShapePredictor(nn.Module):
    def __init__(self, input_dim=100, num_classes=3):
        super(WiFiShapePredictor, self).__init__()
        # A lightweight Multi-Layer Perceptron (MLP) for feature arrays
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)
