from typing import Tuple

import numpy as np
from scipy.spatial.distance import cdist


def fuzzy_c_means(
    X: np.ndarray,
    U: np.ndarray,
    m: float = 2.0,
    max_iter: int = 1000,
    tdist: str = "euclidean",
    error: float = 1e-7,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Fuzzy C-Means (FCM) clustering method
    (https://www.geeksforgeeks.org/ml-fuzzy-clustering/)
    Args:
        X (np.ndarray): input data with shape (n_samples, n_features)
        U (np.ndarray): membership matrix with shape (n_samples, n_clusters)
        m (float, optional): fuzziness parameter. Defaults to 2.0
        max_iter (int, optional): limit on the number of iterations. Defaults to 1000.
        tdist (str, optional): type of distance. Defaults to "euclidean".
        error (float, optional): accuracy. Defaults to 1e-7.

    Raises:
        ValueError: _description_

    Returns:
        Tuple[np.ndarray, np.ndarray, int]: output U-matrix, centroids, final number of iterations
    """

    n_samples = X.shape[0]
    if n_samples != U.shape[0]:
        raise ValueError(
            "in X(n_samples, n_features) and U(n_samples, n_clusters) n_samples is not the same!"
        )
    U /= np.sum(U, axis=1)[:, np.newaxis]  # Normalize the sum to 1

    num_iter = 0
    for _ in range(max_iter):
        num_iter += 1
        U_prev = U.copy()
        # Calculating cluster centers
        centroids = np.dot((U**m).T, X) / np.sum(U**m, axis=0)[:, np.newaxis]

        # Updating the membership matrix
        distances = cdist(X, centroids, tdist)
        # Avoid division by 0
        distances = np.fmax(distances, np.finfo(np.float64).eps)
        U = 1 / (distances ** (2 / (m - 1)))
        U /= np.sum(U, axis=1)[:, np.newaxis]  # Normalization

        # Convergence check
        if np.linalg.norm(U - U_prev) < error:
            break

    return U, centroids, num_iter
