import numpy as np


def create_pairs(data, cycle_start_indices):
    """
    Create pairs of data points within detected cycles.

    This function generates pairs of data points from the provided dataset.
    Each pair consists of a data point and another data point that is 10 steps
    ahead within the same cycle.

    Args:
        data (numpy.ndarray):
                The dataset from which pairs are created.
        cycle_start_indices (list of int):
                The starting indices of detected cycles.

    Returns:
        numpy.ndarray: An array of data point pairs.
    """
    pairs = []
    for start_idx in cycle_start_indices:
        cycle_end_idx = start_idx + 100
        for i in range(start_idx, cycle_end_idx, 10):
            if i + 10 < cycle_end_idx:
                pair = (data[i], data[i + 10])
                pairs.append(pair)
    return np.array(pairs)
