import torch
import torch.nn as nn


class Model(nn.Module):
    
    def __init__(self, num_features):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, features):
        out = self.linear(features)
        out = self.sigmoid(out)
        return out
