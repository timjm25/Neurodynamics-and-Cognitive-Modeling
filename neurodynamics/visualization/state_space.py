"""State-space explorer."""
import numpy as np


class StateSpaceExplorer:
    def __init__(self, trajectory: np.ndarray):
        self.trajectory = np.asarray(trajectory)

    def pca_projection(self, n_components: int = 2) -> np.ndarray:
        from sklearn.decomposition import PCA  # type: ignore
        return PCA(n_components=n_components).fit_transform(self.trajectory)

    def simple_pca(self, n_components: int = 2) -> np.ndarray:
        """PCA without sklearn dependency."""
        X = self.trajectory - self.trajectory.mean(axis=0)
        C = X.T @ X / (len(X) - 1)
        eigvals, eigvecs = np.linalg.eigh(C)
        idx = np.argsort(eigvals)[::-1]
        return X @ eigvecs[:, idx[:n_components]]

    def compute_umap(self, n_components: int = 2) -> np.ndarray:
        try:
            import umap  # type: ignore
            return umap.UMAP(n_components=n_components).fit_transform(self.trajectory)
        except ImportError:
            return self.simple_pca(n_components)
