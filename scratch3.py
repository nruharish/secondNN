import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Generate training data
np.random.seed(0)
num_samples = 100

# Random input features a, b, c
a = np.random.rand(num_samples, 1)
b = np.random.rand(num_samples, 1)
c = np.random.rand(num_samples, 1)

# Stack into a single input matrix X: shape (100, 3)
X_np = np.hstack([a, b, c])

# Compute y = a + 2b + 3c
Y_np = a + 2 * b + 3 * c  # shape: (100, 1)

# 2. Convert to PyTorch tensors
X = torch.tensor(X_np, dtype=torch.float32)
Y = torch.tensor(Y_np, dtype=torch.float32)

# 3. Define a simple linear model: y = w1*a + w2*b + w3*c + b
model = nn.Linear(3, 1)  # 3 input features → 1 output

# 4. Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 5. Train the model
epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.6f}')

# 6. Show learned weights
with torch.no_grad():
    weights = model.weight.data.numpy().flatten()
    bias = model.bias.item()
    print(f"\nLearned weights: a: {weights[0]:.2f}, b: {weights[1]:.2f}, c: {weights[2]:.2f}")
    print(f"Learned bias: {bias:.2f}")
