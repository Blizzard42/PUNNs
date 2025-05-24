import torch
import torch.nn as nn

class SmoothProductUnitLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SmoothProductUnitLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(output_dim, input_dim))  # Exponents
        self.biases = nn.Parameter(torch.randn(output_dim))

    def forward(self, x):
        # Ensure x is strictly positive to prevent log-domain issues
        log_x = torch.log(x + 1)
        mean_of_squares = self.weights.square().mean(dim=1)
        activation = torch.matmul(log_x, self.weights.square().t()) / mean_of_squares
        out = torch.exp(activation + self.biases)
        return out

class SmoothPUNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SmoothPUNN, self).__init__()
        self.product_layer = SmoothProductUnitLayer(input_dim, output_dim)

    def forward(self, x):
        return self.product_layer(x)
