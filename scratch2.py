import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Generate data for y = 3x + 2
X_np = np.linspace(-10, 10, 100).reshape(-1, 1)
Y_np = 3 * X_np + 2

# 2. Convert to PyTorch tensors
X = torch.tensor(X_np, dtype=torch.float32)
Y = torch.tensor(Y_np, dtype=torch.float32)

# 3. Define a simple linear model: y = wx + b
model = nn.Linear(1, 1)  # input: 1 feature, output: 1 target

# 4. Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 5. Train the model
epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# 6. Evaluate
model.eval()
with torch.no_grad():
    predicted = model(X).numpy()

# 7. Visualize
plt.plot(X_np, Y_np, label='Actual y = 3x + 2')
plt.plot(X_np, predicted, label='Predicted', linestyle='--')
plt.legend()
plt.title('Learning y = mx + b using PyTorch')
plt.show()

# 8. Show learned parameters
params = list(model.parameters())
w_learned = params[0].item()
b_learned = params[1].item()
print(f"Learned m (slope): {w_learned:.2f}, Learned b (intercept): {b_learned:.2f}")
