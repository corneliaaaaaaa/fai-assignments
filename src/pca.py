import numpy as np
import ipdb


"""
Implementation of Principal Component Analysis.
"""


class PCA:
    """
    I referred to Vivian's class slides and ChatGPT to implement the code.
    """

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X: np.ndarray) -> None:
        """
        Find the top n principal components from input data.
        """
        # TODO: 10%
        # center the data to create the factor for covariance computation
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # compute the covariance matrix * (N - 1)
        cov_matrix = np.dot(X_centered.T, X_centered)

        # compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigenvalues)[
            ::-1
        ]  # sort eigenvalues in descending order then return the index after sorting
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # choose the n eigenvectors with top n large eigenvalues as principal components
        self.components = eigenvectors[:, : self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project the data onto the base formed by principal components.
        """
        # TODO: 2%
        X_centered = X - self.mean

        return np.dot(X_centered, self.components)

    def reconstruct(self, X):
        """
        Project the data back to the original space.
        """
        # TODO: 2%
        X_centered = X - self.mean
        X_transformed = np.dot(X_centered, self.components)

        return np.dot(self.components, X_transformed) + self.mean
