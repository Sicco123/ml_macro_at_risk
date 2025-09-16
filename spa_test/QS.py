import numpy as np
def QS(y):
    """
    Returns columnwise QS HAC estimator of the variance for a given matrix y.
    """
    T, N = y.shape
    bw = 1.3 * T ** (1 / 5)
    weight = QS_weights(np.arange(1, T) / bw)
    omega = np.zeros(N)
    for i in range(N):
        workdata = y[:, i] - np.mean(y[:, i])
        omega[i] = np.dot(workdata, workdata) / T
        for j in range(1, T):
            omega[i] += 2 * weight[j - 1] * np.dot(workdata[:-j], workdata[j:]) / T
    return omega

def QS_weights(x):
    """
    Calculates Quadratic Spectral (QS) weights used in handling autocorrelation in time series data.
    
    :param x: A numpy array of input values, typically representing lags divided by the maximum lag.
    :return: A numpy array of QS weights.
    """
    arg_qs = 6 * np.pi * x / 5
    w1 = 3. / (arg_qs ** 2)
    w2 = (np.sin(arg_qs) / arg_qs) - np.cos(arg_qs)
    w_qs = w1 * w2
    w_qs[x == 0] = 1  # Handle the case where x is zero to avoid division by zero in the formula
    return w_qs

