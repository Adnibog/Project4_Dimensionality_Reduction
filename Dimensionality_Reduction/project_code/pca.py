import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Covariance, function needs samples as columns
        cov = np.cov(X.T)

        # Eigenvalues, eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # Store first n eigenvectors
        self.components = eigenvectors[0:self.n_components]

    def transform(self, X):
        # Project data
        X = X - self.mean
        return np.dot(X, self.components.T)

    def inverse_transform(self, X):
        return np.dot(X, self.components) + self.mean

    def plot_eigenfaces(self, image_shape):
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        for i, ax in enumerate(axes.flat):
            ax.imshow(self.components[i].reshape(image_shape), cmap='gray')
            ax.set_title(f"Eigenface {i+1}")
            ax.axis('off')
        plt.show()
