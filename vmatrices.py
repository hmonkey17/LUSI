import numpy as np
from numba import jit
import math
#from scipy import special



@jit(nopython=True)
def V_matrix_heaviside_add(X, c_max_offset):
    # X: (n_samples, n_features)
    # c_max_offset: (n_features,) or scalar
    # c_max
    (n, D) = X.shape # D: the number of features;   n: the number of samples
    
    # Since numba doesn't support np.max with axis argument, we create
    # a for loop to compute c_max vector
    c_max = np.empty(shape=(D,), dtype=np.float64)
    for i in range(D):
        c_max[i] = np.max(X[:, i]) + c_max_offset
    V = np.empty(shape=(n, n), dtype=np.float64)
    for j in range(n):
        for i in range(n):
            V[i, j] = np.sum( c_max - np.maximum(X[i, :], X[j, :]) )
    return V / np.abs(V.min())


@jit(nopython=True)
def V_matrix_kde1(X, sigma = None):

    n = X.shape[0]
    if sigma is None: 
        sigma = n * np.std(X) * (n**(-0.2))
    G_mtx = np.empty(shape=(n, n), dtype=np.float64)
    for j in range(n):
        for i in range(n):
            G_mtx[i, j] = G(X[i, :] - X[j, :], sigma)
    V = G_mtx.T @ G_mtx / n
    return V / np.abs(V.min())


@jit(nopython=True)
def G(x, sigma):
    return np.exp( - np.sum(x**2) / (2 * sigma**2) )


'''
# Not stable
@jit(nopython=True)
def V_matrix_kde2(X, sigma = None):
    # X: (n_samples, n_features)
    (n, D) = X.shape
    V = np.empty(shape=(n, n), dtype=np.float64)
    
    if sigma is None:
        sigma = n * np.std(X) * n**(-0.2)
    for i in range(n):
        for j in range(n):
            x_hat = (X[i, :] + X[j, :]) / 2
            V[i, j] = ( np.exp( - np.sum((X[i, :] - X[j, :])**2) / (sigma**2) ) * 
                       np.prod( erf_vec( (1+x_hat)/sigma ) + erf_vec( (1-x_hat)/sigma ) ) )
    return V

@jit(nopython=True)
def erf_vec(x):
    output = np.empty(shape=x.shape, dtype=np.float64)
    for i in range(x.shape[0]):
        output[i] = math.erf(x[i])
    return output
'''
