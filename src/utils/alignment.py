import numpy as np

from utils.normalization import normalize_matrix


def make_binary(
    matrix: np.ndarray, threshold: float = None, top_k: int = None, norm: str = None
):
    if norm is not None:
        matrix = normalize_matrix(matrix, norm)

    if threshold is None and top_k is None:
        raise ValueError(
            "Both 'threshold' and 'top_k' cannot be None. Please provide at least one."
        )

    # Threshold Mask
    if threshold is not None and not np.isnan(threshold):
        threshold_mask = matrix > threshold
    else:
        threshold_mask = np.ones_like(matrix, dtype=bool)

    # Top-K Mask
    if top_k is not None and not np.isnan(top_k):
        top_k = int(top_k)
        top_k_indices = np.argsort(matrix, axis=0)[-top_k:, :]
        top_k_mask = np.zeros_like(matrix, dtype=bool)
        cols = np.arange(matrix.shape[1])[np.newaxis, :]
        top_k_mask[top_k_indices, cols] = True
    else:
        top_k_mask = np.ones_like(matrix, dtype=bool)

    binary_matrix = (threshold_mask & top_k_mask).astype(int)
    return binary_matrix
