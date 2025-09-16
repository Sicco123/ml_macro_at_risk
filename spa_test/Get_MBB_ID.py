import numpy as np

def get_mbb_id(T, L):
    """
    Obtains ids of resampled observations using a moving block bootstrap with blocks of length L.
    :param T: Total number of observations.
    :param L: Length of each block.
    :return: An array of indices corresponding to the resampled observations.
    """
    id = np.zeros(T, dtype=int)
    id[0] = np.ceil(T * np.random.rand()).astype(int) - 1  # Adjust for 0-based indexing in Python
    for t in range(1, T):
        if t % L == 0:
            id[t] = np.ceil(T * np.random.rand()).astype(int) - 1
        else:
            id[t] = id[t - 1] + 1
        if id[t] >= T:  # Ensure the index wraps around if it exceeds the range
            id[t] = 0
    return id