import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return A1, A2

def compute_loss(y, y_pred):
    m = y.shape[0]
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def backward_propagation(X, y, A1, A2, W2):
    m = y.shape[0]

    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * A1 * (1 - A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    return W1, b1, W2, b2

def train(X, y, hidden_size, learning_rate, epochs):
    input_size = X.shape[1]
    output_size = 1
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        A1, A2 = forward_propagation(X, W1, b1, W2, b2)
        loss = compute_loss(y, A2)
        dW1, db1, dW2, db2 = backward_propagation(X, y, A1, A2, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch} \t Loss: {loss:.4f}")

    return W1, b1, W2, b2

def predict(X, W1, b1, W2, b2):
    _, A2 = forward_propagation(X, W1, b1, W2, b2)
    return (A2 > 0.5).astype(int)

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

W1, b1, W2, b2 = train(X, y, hidden_size=2, learning_rate=0.1, epochs=10000)

predictions = predict(X, W1, b1, W2, b2)
print("\nPredictions:")
print(predictions)
