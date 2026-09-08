import torch
import torch.nn as nn
import torch.optim as optim

# Node features
X = torch.tensor([
    [1., 2.],
    [2., 3.],
    [4., 1.]
])

# Adjacency matrix
A = torch.tensor([
    [0., 1., 1.],
    [1., 0., 1.],
    [1., 1., 0.]
])

# Add self-loops
I = torch.eye(3)
A_hat = A + I

# Normalize
D = torch.diag(torch.sum(A_hat, dim=1))
D_inv_sqrt = torch.linalg.inv(torch.sqrt(D))

A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt


class GCN(nn.Module):

    def __init__(self):
        super().__init__()

        self.W = nn.Linear(2, 2)

    def forward(self, X):
        H = A_norm @ X
        H = self.W(H)
        H = torch.relu(H)

        return H


model = GCN()

optimizer = optim.Adam(model.parameters(), lr=0.01)

# Target output
Y = torch.tensor([
    [3., 3.],
    [3., 3.],
    [3., 3.]
])

for epoch in range(1000):

    prediction = model(X)

    loss = torch.mean((prediction - Y) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(epoch, loss.item())


print("\nPrediction:")
print(model(X))