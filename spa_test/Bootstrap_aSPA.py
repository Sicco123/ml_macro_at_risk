import numpy as np
from .MBB_Variance import mbb_variance
from .Get_MBB_ID import get_mbb_id
from .QS import QS

def Bootstrap_aSPA(loss_diff, weights, L):
    """
    Implements Bootstrap Algorithm 1 for the test for Average SPA of Quaedvlieg (2018).
    Bootstrap is based on a Moving Block Bootstrap with parameter L.
    Weights are the weights on the various horizons, size conforming to loss_diff.
    """
    B = 999
    T = loss_diff.shape[0]

    # Weighted loss differences
    
    weighted_loss_diff = np.sum(np.tile(weights, (T, 1)) * loss_diff, axis=1, keepdims=True)

    d_ij = np.mean(weighted_loss_diff)
    t_aspa = np.sqrt(T) * d_ij / np.sqrt(QS(weighted_loss_diff))

    t_aspa_b = np.zeros(B)
   
    demeaned_loss_diff = weighted_loss_diff - np.tile(d_ij, T)[:,None]

    for b in range(B):
        id = get_mbb_id(T, L)
        b_lossdiff = demeaned_loss_diff[id]
        zeta_b = mbb_variance(b_lossdiff, L)
      
        t_aspa_b[b] = np.sqrt(T) * np.mean(b_lossdiff) / np.sqrt(zeta_b)

    return t_aspa, t_aspa_b