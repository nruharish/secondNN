import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Create input and target arrays using numpy
X_np = np.linspace(-10, 10, 100).reshape(-1, 1)
Y_np = X_np ** 2

# 2. Convert numpy arrays to PyTorch tensors
X = torch.tensor(X_np, dtype=torch.float32)
Y = torch.tensor(Y_np, dtype=torch.float32)

# 3. Define a simple neural network
class SquareNet(nn.Module):
    def __init__(self):
        super(SquareNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# 4. Initialize model, loss function and optimizer
model = SquareNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. Train the model
epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# 6. Test and visualize
model.eval()
with torch.no_grad():
    predicted = model(X).numpy()

# Plot the results
plt.plot(X_np, Y_np, label='Actual y = x²')
plt.plot(X_np, predicted, label='Predicted', linestyle='--')
plt.legend()
plt.title('Fitting y = x² using PyTorch NN')
plt.show()
