import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ...models.PUNN import PUNN
from ...models.smoothPUNN import SmoothPUNN
from ...models.OReLU import OReLU

# Parameters
n = 10 # Dimension of input vectors
batch_size = 32
epochs = 400
learning_rate = 1e-2 # Often needs to be higher for PUNNs
num_samples = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("USING DEVICE: ", device)

# Model
# model = PUNN(input_dim=2 * n, output_dim=n).to(device)
# model = MLP(input_dim=2 * n, output_dim=n).to(device)

class MLP2(nn.Module):
    def __init__(self):
        super(MLP2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2* n, 4 * n),
            nn.ReLU(),
            nn.Linear(4 * n, 1)
        )

    def forward(self, x):
        return self.model(x)

class PUNN2(nn.Module):
    def __init__(self):
        super(PUNN2, self).__init__()
        self.model = nn.Sequential(
            PUNN(2 * n, 4 * n),
            OReLU(),
            PUNN(4 *n, 1)
        )

    def forward(self, x):
        return self.model(x)

class SmoothPUNN2(nn.Module):
    def __init__(self):
        super(SmoothPUNN2, self).__init__()
        self.model = nn.Sequential(
            SmoothPUNN(2 * n, 4 * n),
            OReLU(),
            SmoothPUNN(4 *n, 1)
        )

    def forward(self, x):
        return self.model(x)

class Hybrid(nn.Module):
    def __init__(self):
        super(Hybrid, self).__init__()
        self.model = nn.Sequential(
            PUNN(2 * n, 4 * n),
            OReLU(),
            nn.Linear(4 * n, 1)
        )
    
    def forward(self, x):
        return self.model(x)

model = MLP2().to(device)

def test_function(x1, x2):
    x2_sum = torch.sum(x2, dim=1, keepdim=True)
    return torch.sum(x1 * x2_sum, dim=1, keepdim=True)

# Generate dataset
x1 = torch.rand((num_samples, n)) + 1
x2 = torch.rand((num_samples, n)) + 1
y = test_function(x1, x2)

# Create DataLoader
dataset = TensorDataset(torch.cat([x1, x2], dim=1), y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    total_loss = 0.0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    if epoch % 50 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

# Test on a few samples
test_x1 = torch.rand((5, n)) + 1
test_x2 = torch.rand((5, n)) + 1
test_input = torch.cat([test_x1, test_x2], dim=1).to(device)
with torch.no_grad():
    pred = model(test_input)
    print("Predictions vs Ground Truth:")
    print(torch.stack([pred.cpu(), test_function(test_x1, test_x2)], dim=1))

breakpoint()
