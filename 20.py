import numpy as np
import matplotlib.pyplot as plt

# Load data (replace 'salary_data.csv' with your file path)
data = np.genfromtxt('salary_data.csv', delimiter=',', skip_header=1)
X = data[:, 0]  # Years of Experience
y = data[:, 1]  # Salary

# Feature Scaling (Standardization)
X_mean = np.mean(X)
X_std = np.std(X)
X_scaled = (X - X_mean) / X_std

# Hyperparameters
learning_rate = 0.1
iterations = 1000

# Initialize parameters
w = 0.0  # weight (slope)
b = 0.0  # bias (intercept)
n = len(X)

# Gradient Descent
cost_history = []
for i in range(iterations):
    # Predictions
    y_pred = w * X_scaled + b
    
    # Compute cost (MSE)
    cost = (1/(2*n)) * np.sum((y_pred - y)**2)
    cost_history.append(cost)
    
    # Compute gradients
    dw = (1/n) * np.sum((y_pred - y) * X_scaled)
    db = (1/n) * np.sum(y_pred - y)
    
    # Update parameters
    w -= learning_rate * dw
    b -= learning_rate * db
    
    # Print progress
    if i % 100 == 0:
        print(f"Iteration {i}: Cost = {cost:.2f}")

# Training results
print(f"\nFinal parameters: w = {w:.2f}, b = {b:.2f}")
print(f"Feature scaling parameters: mean = {X_mean:.2f}, std = {X_std:.2f}")

# Prediction function
def predict(years_experience):
    scaled_input = (years_experience - X_mean) / X_std
    return w * scaled_input + b

# Plot results
plt.figure(figsize=(12, 5))

# Plot cost history
plt.subplot(1, 2, 1)
plt.plot(cost_history)
plt.title('Cost Function Convergence')
plt.xlabel('Iterations')
plt.ylabel('MSE')

# Plot regression line
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', label='Actual Data')
x_values = np.linspace(min(X), max(X), 100)
y_values = predict(x_values)
plt.plot(x_values, y_values, color='red', label='Regression Line')
plt.title('Years of Experience vs Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()

plt.tight_layout()
plt.show()

# Example prediction
experience = 5.5
predicted_salary = predict(experience)
print(f"\nPredicted salary for {experience} years experience: ${predicted_salary:.2f}")
