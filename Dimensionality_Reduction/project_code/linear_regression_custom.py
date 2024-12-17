import numpy as np

class LinearRegressionCustom:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        # Add bias term
        X = np.c_[np.ones(X.shape[0]), X]
        # Closed-form solution (Normal Equation)
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        # Add bias term
        X = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X, self.weights)
    