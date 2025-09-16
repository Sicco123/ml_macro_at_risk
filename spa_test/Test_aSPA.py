import numpy as np
from .Bootstrap_aSPA import Bootstrap_aSPA
def Test_aSPA(LossDiff, weights, L):
    """
    Implements the test for Average SPA using the forecast path loss differential.
    LossDiff is the [TxH] forecast path loss differential.
    weights is the [1xH] vector of weights for the losses at different horizons.
    L is the parameter for the moving block bootstrap.
    Returns the statistic as well as the p-value.
    """

    t_aSPA, t_aSPA_b = Bootstrap_aSPA(LossDiff, weights, L)
    p_value = np.mean(t_aSPA < t_aSPA_b)
    return t_aSPA[0], p_value


