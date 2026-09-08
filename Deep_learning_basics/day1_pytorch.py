import torch
import torch.nn as nn

x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0], [11.0]])

model = nn.Linear(1, 1)

loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

epochs = 1000

for epoch in range(epochs):

    # Forward pass
    prediction = model(x)

    # Calculate loss
    loss = loss_fn(prediction, y)

    # Clear old gradients
    optimizer.zero_grad()

    # Calculate new gradients
    loss.backward()

    # Update weights
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")

print("\nLearned parameters:")

print("Weight:", model.weight.item())
print("Bias:", model.bias.item())