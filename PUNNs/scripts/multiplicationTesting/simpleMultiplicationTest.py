import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ...models.PUNN import PUNN
from ...models.MLP import MLP
from ...models.smoothPUNN import SmoothPUNN

# Parameters
n = 1 # Dimension of input vectors
batch_size = 32
epochs = 400
learning_rate = 1e-2  # Often needs to be higher for PUNNs
num_samples = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = MLP(input_dim=2 * n, output_dim=n).to(device)
# model = PUNN(input_dim=2 * n, output_dim=n).to(device)
# model = SmoothPUNN(input_dim=2 * n, output_dim=n).to(device)

# test_function = lambda x1, x2: x1 * x2 + x1 + x2 + 1
test_function = lambda x1, x2: x1 + x2

# Generate dataset
x1 = 100 * torch.rand((num_samples, n)) + 1
x2 = 100 * torch.rand((num_samples, n)) + 1
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
test_x1 = 100 * torch.rand((5, n)) + 1
test_x2 = 100 * torch.rand((5, n)) + 1
test_input = torch.cat([test_x1, test_x2], dim=1).to(device)
with torch.no_grad():
    pred = model(test_input)
    print("Predictions vs Ground Truth:")
    print(torch.stack([pred.cpu(), test_function(test_x1, test_x2)], dim=1))

breakpoint()
