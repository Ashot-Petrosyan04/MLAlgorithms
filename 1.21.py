import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
x = np.linspace(0, 5, 100)
y = 2 * x ** 2 - 3 * x + np.random.normal(0, 3, x.shape)

X_train, X_test, y_train, y_test = train_test_split(x.reshape(100, 1), y.reshape(100), test_size = 0.2, random_state = 42)

poly = PolynomialFeatures(degree = 2, include_bias = False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

plt.scatter(X_test, y_test, color = 'blue', label = 'True')
plt.scatter(X_test, y_pred, color = 'red', label = 'Predicted')
plt.legend()
plt.show()
