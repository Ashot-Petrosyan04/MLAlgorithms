import numpy as np
import pandas as pd

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = 0.0
        self.b = 0.0
        self.cost_history = []

    def fit(self, X, y):
        n = len(X)
        self.w = 0.0
        self.b = 0.0
        self.cost_history = []
        
        for _ in range(self.epochs):
            y_pred = self.w * X + self.b
            cost = (1 / n) * np.sum((y_pred - y) ** 2)
            self.cost_history.append(cost)
            
            dw = (2 / n) * np.sum((y_pred - y) * X)
            db = (2 / n) * np.sum(y_pred - y)
            
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, years_experience):
        return self.w * years_experience + self.b

data = pd.read_csv('salary_pred.txt', header=None)
X = data[0]
y = data[1]
model = LinearRegressionGD()
model.fit(X, y)
prediction = model.predict(5)
