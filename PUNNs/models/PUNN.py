import torch
import torch.nn as nn

class ProductUnitLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProductUnitLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(output_dim, input_dim))  # Exponents
        self.biases = nn.Parameter(torch.randn(output_dim))

    def forward(self, x):
        # Ensure x is strictly positive to prevent log-domain issues
        x = torch.clamp(x, min=1e-6)
        log_x = torch.log(x)
        out = torch.exp(torch.matmul(log_x, self.weights.t()) + self.biases)
        return out

class PUNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PUNN, self).__init__()
        self.product_layer = ProductUnitLayer(input_dim, output_dim)

    def forward(self, x):
        return self.product_layer(x)
