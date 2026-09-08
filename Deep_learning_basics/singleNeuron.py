x = [1, 2, 3, 4, 5]
y = [3, 5, 7, 9, 11]

w = 0.0
b = 0.0

learning_rate = 0.01
epochs = 1000

n = len(x)

for epoch in range(epochs):

    # Forward pass
    predictions = [w * xi + b for xi in x]

    # Calculate loss
    loss = sum((yi - pi) ** 2 for yi, pi in zip(y, predictions)) / n

    # Calculate gradients
    dw = sum(-2 * xi * (yi - pi)
             for xi, yi, pi in zip(x, y, predictions)) / n

    db = sum(-2 * (yi - pi)
             for yi, pi in zip(y, predictions)) / n

    # Update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss={loss:.4f}, w={w:.4f}, b={b:.4f}")

print("\nFinal:")
print("w =", w)
print("b =", b)