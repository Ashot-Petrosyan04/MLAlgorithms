import numpy as np

class SoftmaxRegression:
    def __init__(self, learning_rate = 0.01, max_epochs = 1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.W = None
        self.classes = None
        
    def _softmax(self, z):
        z = z - np.max(z, axis = 1, keepdims = True)
        e_z = np.exp(z)

        return e_z / np.sum(e_z, axis = 1, keepdims = True)
    
    def _one_hot(self, y):
        one_hot = np.zeros((len(y), len(self.classes)))
        one_hot[np.arange(len(y)), y] = 1

        return one_hot
    
    def fit(self, X, y, tol = 1e-4, patience = 5):
      X = np.insert(X, 0, 1, axis = 1)
      self.classes = np.unique(y)
      n_classes = len(self.classes)
      n_features = X.shape[1]
      
      self.W = np.zeros((n_features, n_classes))
      y_one_hot = self._one_hot(y)
      
      prev_loss = float('inf')
      patience_count = 0
      
      for epoch in range(self.max_epochs):
          scores = X @ self.W
          y_hat = self._softmax(scores)
          
          loss = -np.sum(y_one_hot * np.log(y_hat)) / X.shape[0]
          
          if abs(prev_loss - loss) < tol:
              break
              
          if loss > prev_loss:
              patience_count += 1

              if patience_count >= patience:
                  break
          else:
              patience_count = 0 
              
          prev_loss = loss
          
          gradient = (X.T @ (y_one_hot - y_hat)) / X.shape[0]
          self.W += self.learning_rate * gradient

    def predict(self, X):
        X = np.insert(X, 0, 1, axis = 1)
        scores = X @ self.W

        return self.classes[np.argmax(scores, axis = 1)]
