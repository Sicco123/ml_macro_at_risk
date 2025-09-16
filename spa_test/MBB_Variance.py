import numpy as np

def mbb_variance(y, L):
    """
    Computes the 'natural variance' estimator of the resampled data using moving block bootstrap method.
    :param y: A 2D numpy array where each column represents a series of observations.
    :param L: The length of each block in the bootstrap.
    :return: A numpy array containing the variance estimates for each series.
    """
    T, N = y.shape
    omega = np.zeros(N)
    y_dem = y - np.tile(np.mean(y, axis=0), (T, 1))  # Demean each series
    K = T // L  # Number of complete blocks

    for n in range(N):
        temp = np.reshape(y_dem[:K * L, n], (K, L))
        omega[n] = np.mean(np.sum(temp, axis=1) ** 2) / L

    return omega