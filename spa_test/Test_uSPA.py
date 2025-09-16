import numpy as np
from .Bootstrap_uSPA import Bootstrap_uSPA

def Test_uSPA(LossDiff, L):
    """
    Implements the test for Unconditional SPA using forecast path loss differential.
    LossDiff is the [TxH] forecast path loss differential.
    L is the parameter for the moving block bootstrap.
    Returns the statistic as well as the p-value.
    """
    t_uSPA, t_uSPA_b = Bootstrap_uSPA(LossDiff,  L)
    p_value = np.mean(t_uSPA < t_uSPA_b)
    return t_uSPA, p_value