import numpy as np


NORMALIZATION_METHODS = ["minmax", "zscore"]


def normalize_matrix(mat: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Normalize matrix with either z-score or min-max."""
    if not method in NORMALIZATION_METHODS:
        raise ValueError("Invalid normalization method")

    if mat.size == 0:
        return mat

    if method == "zscore":
        mean, std = np.mean(mat), np.std(mat)
        if std == 0:
            return np.zeros_like(mat)
        return (mat - mean) / std
    elif method == "minmax":
        min_val, max_val = np.min(mat), np.max(mat)
        if max_val - min_val == 0:
            return np.zeros_like(mat)
        return (mat - min_val) / (max_val - min_val)
    raise ValueError()
