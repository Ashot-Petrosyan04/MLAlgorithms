import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, max_epochs=1000, patience=5, tol=1e-4):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.tol = tol
        self.W = None
        self.best_W = None
        self.classes_ = None
        
    def _softmax(self, z):
        # Numerically stable softmax
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / e_z.sum(axis=1, keepdims=True)
    
    def _cross_entropy(self, y_true, y_probs):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_probs[range(m), y_true])
        return np.sum(log_likelihood) / m
    
    def fit(self, X, y):
        # Preprocess data
        X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add bias term
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # Initialize parameters
        self.W = np.random.randn(n_features, n_classes) * 0.01
        best_loss = np.inf
        no_improvement = 0
        
        # Split validation set
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        
        for epoch in range(self.max_epochs):
            # Forward pass
            scores = X_train @ self.W
            probs = self._softmax(scores)
            
            # Compute loss
            train_loss = self._cross_entropy(y_train, probs)
            
            # Backward pass
            grad = np.zeros_like(self.W)
            m = X_train.shape[0]
            
            # Create one-hot encoded labels
            y_one_hot = np.eye(n_classes)[y_train]
            
            # Compute gradient
            error = probs - y_one_hot
            grad = (X_train.T @ error) / m
            
            # Update weights
            self.W -= self.learning_rate * grad
            
            # Validation
            val_scores = X_val @ self.W
            val_probs = self._softmax(val_scores)
            val_loss = self._cross_entropy(y_val, val_probs)
            
            # Early stopping check
            if val_loss < best_loss - self.tol:
                best_loss = val_loss
                self.best_W = self.W.copy()
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                    
            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Restore best weights
        self.W = self.best_W
    
    def predict_proba(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        scores = X @ self.W
        return self._softmax(scores)
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# Example usage
# Generate sample data
X, y = make_classification(n_samples=1000, n_features=4, n_classes=3, n_informative=3)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize and train model
model = SoftmaxRegression(learning_rate=0.1, patience=5)
model.fit(X, y)

# Predictions
y_pred = model.predict(X)
print("Accuracy:", np.mean(y == y_pred))
