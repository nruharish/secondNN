import torch
import torch.nn as nn


class Model(nn.Module):
    
    def __init__(self, num_features):
        super().__init__()
        self.linear_layer1 = nn.Linear(num_features, 3)
        self.relu = nn.ReLU()
        self.linear_layer2 = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, features):
        out = self.linear_layer1(features)
        out = self.relu(out)
        out = self.linear_layer2(out)
        out = self.sigmoid(out)
        return out
