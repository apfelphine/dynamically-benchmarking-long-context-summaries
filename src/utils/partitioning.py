from typing import List, Callable

import numpy as np


def aggregate_partition_usage(
    n_partitions: int,
    sim_matrix: np.ndarray,
    check_chunk: Callable[[int, int], bool],
    func=np.any,
    normalize=False,
) -> np.ndarray:
    res = np.zeros(n_partitions)
    for i in range(n_partitions):
        mask = np.array([check_chunk(j, i) for j in range(sim_matrix.shape[0])])
        if not np.any(mask):
            res[i] = 0
            continue
        val = func(sim_matrix[mask])
        if normalize:
            val /= np.sum(mask)
        res[i] = val
    return res


def partition_by_cumulative_sum(data: List[int], n_partitions: int = 25) -> np.ndarray:
    cumulative_sum = np.cumsum(np.array(data))
    total_sum = cumulative_sum[-1]
    edges = np.linspace(0, total_sum, n_partitions + 1)
    bin_indices = np.digitize(cumulative_sum, edges) - 1
    return np.clip(bin_indices, 0, n_partitions - 1)


def partition_centered(data: List[int], n_partitions: int = 25) -> np.ndarray:
    data_arr = np.array(data)

    cumulative_sum = np.cumsum(data_arr)
    total_sum = cumulative_sum[-1]

    prev_cumsum = np.insert(cumulative_sum[:-1], 0, 0)
    midpoints = (cumulative_sum + prev_cumsum) / 2
    edges = np.linspace(0, total_sum, n_partitions + 1)

    bin_indices = np.digitize(midpoints, edges) - 1

    # Force the last item into the last bin if it fell short
    bin_indices[-1] = n_partitions - 1
    return np.clip(bin_indices, 0, n_partitions - 1)
