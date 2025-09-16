import numpy as np
from .Get_MBB_ID import get_mbb_id
from .MBB_Variance import mbb_variance
from .QS import QS
def Bootstrap_uSPA(loss_diff, L):
    """
    Implements Bootstrap Algorithm 1 for the test for Uniform SPA of Quaedvlieg (2018).
    Bootstrap is based on a moving block bootstrap with length L.
    """
    B = 999
    T = loss_diff.shape[0]

    # Calculate means of loss differences
    d_ij = np.mean(loss_diff, axis=0)
    t_uspa = np.min(np.sqrt(T) * d_ij / np.sqrt(QS(loss_diff)), axis=0)

    t_uspa_b = np.zeros(B)
    demeaned_loss_diff = loss_diff - np.tile(d_ij, (T, 1))

    for b in range(B):
        id = get_mbb_id(T, L)
        b_lossdiff = demeaned_loss_diff[id, :]
        omega_b = mbb_variance(b_lossdiff, L)
        t_uspa_b[b] = np.min(np.sqrt(T) * np.mean(b_lossdiff, axis=0) / np.sqrt(omega_b))

    return t_uspa, t_uspa_b