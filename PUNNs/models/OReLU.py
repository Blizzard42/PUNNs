import torch
import torch.nn as nn

class OReLU(nn.Module):
    def forward(self, x):
        return torch.relu(x - 1) + 1
