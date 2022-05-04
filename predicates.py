import numpy as np
from numba import jit, prange
from scipy import signal

def f1(X):
    # constant predicate
    n_samples = X.shape[0]
    return np.ones((n_samples, 1))

def f2(X):
    # identity predicate
    return X

def f3(X, n_channels = 1):
    # tangent distance based predicate
    n_samples, n_features = X.shape
    output = np.empty(shape=(n_samples, 3), dtype=np.float64)
    
    for i in prange(n_samples):
        H = int(np.sqrt(n_features/n_channels))
        if n_channels > 1:
            img = np.reshape(X[i], (H, -1, n_channels))
        else:
            img = np.reshape(X[i], (H, -1))
        output[i, 0] = tangentDistance(img, np.flip(img, axis=1))[0]
        output[i, 1] = tangentDistance(img, np.flip(img, axis=0))[0]
        output[i, 2] = tangentDistance(img, np.flip(np.flip(img, axis=0), axis=1))[0]
    return output

    
@jit(nopython=True)
def gaussianFunction(x, y, sigma):
    return np.exp(- (x ** 2 + y ** 2) / (2.0 * sigma ** 2))

@jit(nopython=True)
def dgdx(x, y, sigma):
    return -(  x * np.exp(-(x**2/2 + y**2/2)/sigma**2)  )/sigma**2

@jit(nopython=True)
def dgdy(x, y, sigma):
    return -(  y * np.exp(-(x**2/2 + y**2/2)/sigma**2)  )/sigma**2

@jit(nopython=True)
def meshgrid(x, y):
    # a numblifiable version of np.meshgrid
    xx = np.empty(shape=(x.size, y.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size), dtype=y.dtype)
    for i in range(x.size):
        for j in range(y.size):
            xx[i, j] = x[j]
            yy[i, j] = y[i]                
    return xx, yy

def conv2d(img, kernel):
    # img: (H, W, C) or (H, W)
    # kernel: (HH, WW)
    if len(img.shape) == 2:
        return signal.convolve2d(img, kernel, mode='same')
    elif len(img.shape) == 3:
        (H, W, C) = img.shape
        output = np.empty(shape=(H, W, C), dtype=np.float64)
        for i in range(C):
            output[:, :, i] = signal.convolve2d(img[:, :, i], kernel, mode='same')
        return output
    else:
        raise ValueError("The img must be a 2D or 3D numpy array")
    

def tangentVector(img, filter_size):
    # Return the tangent vector of 
    # img: (H, W, C) or (H, W)
    # six transformations
    #      1: Horizontal translation
    #      2: Vertical translation
    #      3: Rotation
    #      4: Scaling
    #      5: Parallel hyperbolic
    #      6: Diagonal hyperbolic
    
    H = img.shape[0]
    W = img.shape[1]
    if len(img.shape) == 2:
        output = np.empty(shape=(7, H, W), dtype=np.float64)
    elif len(img.shape) == 3:
        C = img.shape[2]
        output = np.empty(shape=(7, H, W, C), dtype=np.float64)
    
    xx_filter, yy_filter = meshgrid(np.arange(filter_size), np.arange(filter_size))
    xx, yy = meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W))
    
    center_loc = np.floor(filter_size / 2)
    gaussian_filter_1 = dgdx(xx_filter - center_loc, yy_filter - center_loc, 0.9)
    gaussian_filter_2 = dgdy(xx_filter - center_loc, yy_filter - center_loc, 0.9)
    conv_1 = conv2d(img, gaussian_filter_1)
    conv_2 = conv2d(img, gaussian_filter_2)
    output[0] = conv_1 # Horizontal translation
    output[1] = conv_2 # Vertical translation
    output[2] = yy*conv_1 - xx*conv_2 # Rotation
    output[3] = xx*conv_1 + yy*conv_2 # Scaling
    output[4] = xx*conv_1 - yy*conv_2 # Parallel hyperbolic
    output[5] = yy*conv_1 + xx*conv_2 # Diagonal hyperbolic
    output[6] = np.sqrt(conv_1**2 + conv_2**2)
    
    return output

def tangentDistance(img1, img2):
    # img1: (H, W) or (H, W, C)
    # img2: (H, W) or (H, W, C)
    tvec1 = tangentVector(img1, 17)
    tvec2 = tangentVector(img2, 17)
    
    # reshape into vector
    P = np.reshape(img1, (-1))
    E = np.reshape(img2, (-1))
    
    # Equation (7)
    L_P = np.reshape(tvec1, (7, -1)).T #(n_features, 6)
    L_E = np.reshape(tvec2, (7, -1)).T #(n_features, 6)
    
    L_EE = L_E.T @ L_E
    L_PE = L_P.T @ L_E
    L_EP = L_PE.T
    L_PP = L_P.T @ L_P
    
    k = 0.001
    
    alpha_P = np.linalg.solve(L_PE @ np.linalg.inv(L_EE) @ L_EP - (1+k)**2 * L_PP, 
                             (L_PE @ np.linalg.inv(L_EE) @ L_E.T - (1+k)*L_P.T) @ (E - P))
    
    alpha_E = np.linalg.solve((1+k)**2 * L_EE - L_EP @ np.linalg.inv(L_PP) @ L_PE, 
                             (L_EP @ np.linalg.inv(L_PP) @ L_P.T - (1+k)*L_E.T) @ (E - P))
    
    E_transform = E + L_E @ alpha_E
    P_transform = P + L_P @ alpha_P
    return np.sqrt(np.sum((E_transform - P_transform)**2)), P_transform, E_transform, alpha_P, alpha_E, np.sqrt(np.sum((E - P)**2))