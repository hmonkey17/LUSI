import numpy as np
import cvxpy as cp
import scipy.stats
from vmatrices import V_matrix_heaviside_add, V_matrix_kde1

import os
#print(os.listdir("data/input"))

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import rbf_kernel


class LUSI_exact(BaseEstimator, ClassifierMixin):
    
    def __init__(
        self,
        tau = 1.0, # The parameter of regularization term
        kernel = 'rbf', # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’
        degree = 3, # degree of the polynomial kernel, 
        gamma = 'scale', # {‘scale’, ‘auto’} or float, default=’scale’
        V_type = 'add', # The type of the V_matrix, 'add', 'kde1' or 'I'
        predicate_size = 0
                ):
        
        self.tau = tau
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.V_type = V_type
        self.predicate_size = predicate_size
        
    def fit(self, X, y):
        # X: (n_samples, n_features)
        # y: (n_samples,)

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Step 1: set up Phi (n_samples, m) and several other useful matrix
        if self.predicate_size > 0:
            Phi = X[:, -self.predicate_size:] # The last m columns
            X = X[:, :-self.predicate_size] # The first n_features columns
        
        n_samples = X.shape[0]
        self.n_features_in_ = X.shape[1]
        
        # Step 2: set up the V matrix
        if self.V_type == 'add':
            V = V_matrix_heaviside_add(X, 0.05)
        elif self.V_type == 'kde1':
            V = V_matrix_kde1(X)
        elif self.V_type == 'I':
            V = np.eye(n_samples)
        
        # Step 3: set up Kernel Matrix
        if self.gamma == 'scale':
            self.gamma_num_ =  1.0 / (self.n_features_in_ * X.var())
        else:
            self.gamma_num_ = self.gamma
        
        self.K_ = rbf_kernel(X = X, Y = None, gamma = self.gamma_num_)
        
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        
        I_n_s = np.eye(n_samples)
        one_n_s = np.ones(n_samples)
        coe_mtx = V @ self.K_ + self.tau * I_n_s
        
        if self.predicate_size == 0:
            self.A_list_, self.c_list_ = LUSI_exact_solver_no_predicates(self.K_, V, coe_mtx, y, one_n_s, self.classes_)
        else:
            # fit
            self.A_list_, self.c_list_ = LUSI_exact_solver(self.K_, V, 
                                                                      Phi, coe_mtx, y, I_n_s, 
                                                                      one_n_s, self.classes_)    
        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self)
        
        if self.predicate_size > 0:
            X = X[:, :-self.predicate_size] # The first n_features columns

        # Input validation
        X = check_array(X)
        
        K_star = rbf_kernel(X, self.X_, gamma = self.gamma_num_)
        
        scores = np.stack([K_star @ self.A_list_[c] + self.c_list_[c] for c in range(len(self.classes_))], axis = 1)
        pred = np.argmax(scores, axis = 1)
        
        return self.classes_[pred]
    
    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"tau": self.tau, 
                "kernel": self.kernel, 
                "degree": self.degree, 
                "gamma": self.gamma,
                "V_type": self.V_type,
                "predicate_size": self.predicate_size
                }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    

def setup_A_s(Phi, coe_mtx):
    # Phi: matrix of predicates --- (n_samples, m)
    # coe_mtx: VK + tau * I --- (n_samples, n_samples)
    n_samples, m = Phi.shape
    A_s = np.zeros((n_samples, m))

    for t in range(m):
        A_s[:, t] = np.linalg.solve(coe_mtx, Phi[:, t])

    return A_s
    
def setup_COE(V, K, A_c, A_s, Phi, one_n_s):
    n_samples, m = Phi.shape
    COE = np.zeros((m+1, m+1)) # The coefficient of the linear system

    COE[0, 0] = one_n_s[None, :] @ V @ K @ A_c - one_n_s[None, :] @ V @ one_n_s

    for s in np.arange(m):
        COE[0, 1+s] = one_n_s[None, :] @ V @ K @ A_s[:, s] - one_n_s[None, :] @ Phi[:, s]
        COE[1+s, 0] = A_c[None, :] @ K @ Phi[:, s] - one_n_s[None, :] @ Phi[:, s]
        for k in np.arange(m):
            COE[k+1, s+1] = A_s[:, s][None, :] @ K @ Phi[:, k]

    return COE

def setup_RHS(V, K, A_v, Phi, Y, one_n_s):
    n_samples, m = Phi.shape
    RHS = np.zeros(m+1)
    RHS[0] = one_n_s[None, :] @ V @ K @ A_v - one_n_s[None, :] @ V @ Y
    for s in np.arange(m):
        RHS[1+s] = A_v[None, :] @ K @ Phi[:, s] - Y[None, :] @ Phi[:, s]

    return RHS

def solve_coef(COE, RHS, A_v, A_c, A_s):
    m = COE.shape[0] - 1
    try:
        coef = np.linalg.solve(COE, RHS)
    # handles singular exception    
    except:   
        coef = np.linalg.pinv(COE) @ RHS
    

    c = coef[0]
    A = A_v - c * A_c
    for s in np.arange(m):
        A -= coef[s+1] * A_s[:, s]

    return A, c


def LUSI_exact_solver(K, V, Phi, coe_mtx, Y, I_n_s, one_n_s, classes_):
    # K: kernel matrix. --- (n_samples, n_samples)
    # V: V_matrix --- (n_samples, n_samples)
    # Phi: matrix of predicates --- (n_samples, m)
    # coe_mtx: VK + tau * I --- (n_samples, n_samples)
    # Y --- (n_samples, )
    # I_n_s: Identity matrix --- (n_samples, n_samples)
    # one_n_s: One matrix --- (n_samples,)
    # classes_: unique_labels(Y)


    A_c = np.linalg.solve(coe_mtx, V @ one_n_s)
    A_s = setup_A_s(Phi, coe_mtx)
    COE = setup_COE(V, K, A_c, A_s, Phi, one_n_s)

    num_classes = classes_.shape[0]

    A_list = [] # The parameters of each binary classifier
    c_list = []

    for c in range(num_classes):
        Y_temp = (Y == classes_[c]).astype(np.float64)

        A_v_temp = np.linalg.solve(coe_mtx, V @ Y_temp)
        RHS_temp = setup_RHS(V, K, A_v_temp, Phi, Y_temp, one_n_s)
        A_temp, c_temp = solve_coef(COE, RHS_temp, A_v_temp, A_c, A_s)
        A_list.append(A_temp)
        c_list.append(c_temp)

    return A_list, c_list  
    

def LUSI_exact_solver_no_predicates(K, V, coe_mtx, Y, one_n_s, classes_):
    # K: kernel matrix. --- (n_samples, n_samples)
    # V: V_matrix --- (n_samples, n_samples)
    # coe_mtx: VK + tau * I --- (n_samples, n_samples)
    # Y --- (n_samples, )
    # I_n_s: Identity matrix --- (n_samples, n_samples)
    # one_n_s: One matrix --- (n_samples,)
    # classes_: unique_labels(Y)


    A_c = np.linalg.solve(coe_mtx, V @ one_n_s)

    num_classes = classes_.shape[0]

    A_list = [] # The parameters of each binary classifier
    c_list = []

    for c in range(num_classes):
        Y_temp = (Y == classes_[c]).astype(np.float64)

        A_b_temp = np.linalg.solve( coe_mtx, V @ Y_temp ) # 
        c_temp = (one_n_s[None, :] @ V @ K @ A_b_temp - 
                  one_n_s[None, :] @ V @ Y_temp) / (one_n_s[None, :] @ V @ K @ A_c - 
                                                    one_n_s[None, :] @ V @ one_n_s)
        A_temp = A_b_temp - c_temp * A_c 

        A_list.append(A_temp)
        c_list.append(c_temp)


    return A_list, c_list
    
class LUSI_approx(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        tau = 0.5, # The parameter of regularization term
        lamda = 0.01,
        kernel = 'rbf', # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’
        degree = 3, # degree of the polynomial kernel, 
        gamma = 'scale', # {‘scale’, ‘auto’} or float, default=’scale’
        V_type = 'add', # The type of the V_matrix, 'add' or 'multiply'
        predicate_size = 0
                ):
 
        self.tau = tau
        self.lamda = lamda
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.V_type = V_type
        self.predicate_size = predicate_size
        
    def fit(self, X, y):
        # X: (n_samples, n_features)
        # y: (n_samples,)

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Step 1: set up Phi (n_samples, m) and several other useful matrix
        if self.predicate_size > 0:
            Phi = X[:, -self.predicate_size:] # The last m columns
            X = X[:, :-self.predicate_size] # The first n_features columns

        
        n_samples, self.n_features_in_ = X.shape
        
        # Step 2: set up the V matrix
        if self.V_type == 'add':
            V = V_matrix_heaviside_add(X, 0.05)
        elif self.V_type == 'kde1':
            V = V_matrix_kde1(X)
        elif self.V_type == 'I':
            V = np.eye(n_samples)
            
        # Step 3: set up Kernel Matrix
        if self.gamma == 'scale':
            self.gamma_num_ =  1.0 / (self.n_features_in_ * X.var())
        else:
            self.gamma_num_ = self.gamma
        
        self.K_ = rbf_kernel(X = X, Y = None, gamma = self.gamma_num_)   

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        
        I_n_s = np.eye(n_samples)
        one_n_s = np.ones(n_samples)
        
        if self.predicate_size == 0:
            tau = 1
            Phi = None
            self.A_list_, self.c_list_ = LUSI_approx_solver(self.K_, V, tau, self.lamda, 
                                                                        Phi, y, I_n_s, 
                                                                        one_n_s, self.classes_)
        else:
            # fit
            self.A_list_, self.c_list_ = LUSI_approx_solver(self.K_, V, self.tau, self.lamda, 
                                                                        Phi, y, I_n_s, 
                                                                        one_n_s, self.classes_)
        self.y_ = y
        self.X_ = X
        # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self)

        if self.predicate_size > 0:
            X = X[:, :-self.predicate_size] # The first n_features columns
        # Input validation
        X = check_array(X)
        
        K_star = rbf_kernel(X, self.X_, gamma = self.gamma_num_)
        
        scores = np.stack([K_star @ self.A_list_[c] + self.c_list_[c] for c in range(len(self.classes_))], axis = 1)
        pred = np.argmax(scores, axis = 1)
        
        return self.classes_[pred]
    
    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"tau": self.tau, 
                "lamda": self.lamda,
                "kernel": self.kernel, 
                "degree": self.degree, 
                "gamma": self.gamma,
                "V_type": self.V_type,
                "predicate_size": self.predicate_size
                }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    

def LUSI_approx_solver(K, V, tau, lamda, Phi, Y, I_n_s, one_n_s, classes_):
        # K: kernel matrix. --- (n_samples, n_samples)
        # V: V_matrix --- (n_samples, n_samples)
        # Phi: matrix of predicates --- (n_samples, m)
        # Y --- (n_samples, )
        # I_n_s: Identity matrix --- (n_samples, n_samples)
        # one_n_s: One matrix --- (n_samples,)
        # classes_: unique_labels(Y)
        
        # setup P
        if Phi is None:
            coe_mtx = tau * V
        else:
            m = Phi.shape[1]
            P = Phi @ Phi.T / m
            coe_mtx = tau * V + (1 - tau) * P
        
        num_classes = classes_.shape[0]
        
        A_list = [] # The parameters of each binary classifier
        c_list = []
        
        for c in range(num_classes):
            Y_temp = (Y == classes_[c]).astype(np.float64)
            
            c_temp = (one_n_s[None, :] @ coe_mtx @ Y_temp) / (one_n_s[None, :] @ coe_mtx @ one_n_s)
            A_temp = np.linalg.solve( coe_mtx @ K + lamda * I_n_s, coe_mtx @ (Y_temp - c_temp * one_n_s) ) # 
            
            A_list.append(A_temp)
            c_list.append(c_temp)
        return A_list, c_list 
    
    
class LUSI_SVM(BaseEstimator, ClassifierMixin):
    
    def __init__(
        self,
        C = 1.0, # The coefficient of the second term in the loss function, 
        tau = 0.5, # The parameter of SVM loss versus LUSI loss
        epsilon_star = 0.001,
        eta = 0.95,
        kernel = 'rbf', # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’
        degree = 3, # degree of the polynomial kernel, 
        gamma = 'scale', # {‘scale’, ‘auto’} or float, default=’scale’
        predicate_size = 0
                ):
        
        self.C = C
        self.tau = tau
        self.epsilon_star = epsilon_star
        self.eta = eta
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.predicate_size = predicate_size
        
    def fit(self, X, y):
        # X: (n_samples, n_features)
        # y: (n_samples,)

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Step 1: set up Phi (n_samples, m) and several other useful matrix
        if self.predicate_size > 0:
            Phi = X[:, -self.predicate_size:] # The last m columns
            X = X[:, :-self.predicate_size] # The first n_features columns
        else:
            Phi = None

        
        n_samples, self.n_features_in_ = X.shape
        
        
        # Step 2: set up Kernel Matrix
        if self.gamma == 'scale':
            self.gamma_num_ =  1.0 / (self.n_features_in_ * X.var())
        else:
            self.gamma_num_ = self.gamma
        
        self.K_ = rbf_kernel(X = X, Y = None, gamma = self.gamma_num_)
        

        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        
        self.A_list_, self.c_list_ = LUSI_SVM_solver(self.K_, self.C, self.tau, self.eta, 
                                                                y, self.epsilon_star, self.classes_, Phi)
        
        self.y_ = y
        self.X_ = X
        # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self)
        if self.predicate_size > 0:
            X = X[:, :-self.predicate_size] # The first n_features columns

        # Input validation
        X = check_array(X)
        
        K_star = rbf_kernel(X, self.X_, gamma = self.gamma_num_)
        
        scores = np.stack([K_star @ self.A_list_[c] + self.c_list_[c] for c in range(len(self.classes_))], axis = 1)
        pred = np.argmax(scores, axis = 1)
        
        return self.classes_[pred]
    
    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"C": self.C,
                "tau": self.tau, 
                "epsilon_star": self.epsilon_star,
                "eta": self.eta, 
                "kernel": self.kernel, 
                "degree": self.degree,
                "gamma": self.gamma,
                "predicate_size": self.predicate_size
                }


    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
def LUSI_SVM_solver(K, C, tau, eta, Y, epsilon_star, classes_, Phi = None):
    # K: the kernel matrix
    # C: hyper parameter
    # tau: hyperparameter
    # Y: multi-class labels
    # epsilon_star: hyperparameter
    # classes_: the set of labels
    # Phi: optitional, predicates evaluated on the training set

    num_classes = classes_.shape[0]

    n_s = K.shape[0]
    one_n_s = np.ones(n_s)

    # Set up P
    if Phi is not None:
        m = Phi.shape[1]
        one_m = np.ones(m)
    else:
        m = 0

    P_np = np.zeros((2 * n_s + 1 + m, 2 * n_s + 1 + m))
    P_np[0:n_s, 0:n_s] = K

    try:
        P = cp.Parameter(shape=(2*n_s+1+m, 2*n_s+1+m), value = P_np , PSD=True)
    except:
        P_psd = nearestPD(P_np.copy())
        P = cp.Parameter(shape=(2*n_s+1+m, 2*n_s+1+m), value = P_psd , PSD=True)
    #print("Shape of P:", 2*n_s+1+m)

    # Set up D
    d = np.zeros(2 * n_s + 1 + m)
    if Phi is not None:
        d[n_s+1:n_s+1+m] = C * (1 - tau) * one_m

    d[-n_s:] = C * tau * one_n_s

    # set up epsilon_s
    if Phi is not None: #Phi: (n_l, m)
        Phi_max = np.max(Phi, axis = 0)
        Phi_min = np.min(Phi, axis = 0)
        epsilon_s = (Phi_max - Phi_min) * np.sqrt( -n_s * np.log(eta / 2.0) /2.0 )
        #epsilon_s =  np.sqrt( -n_s * np.log(eta / 2.0) /2.0 )
    else:
        epsilon_s = None



    A_list = [] # The parameters of each binary classifier
    c_list = []

    for c in range(num_classes):
        Y_temp = (Y == classes_[c]).astype(int)

        A_temp, c_temp = LUSI_SVM_case_optimizer(one_n_s, P, K, d, epsilon_star, epsilon_s, Y_temp, Phi)

        A_list.append(A_temp)
        c_list.append(c_temp)

    return A_list, c_list
        
    
def LUSI_SVM_case_optimizer(one_n_s, P, K, d, 
                            epsilon_star, epsilon_s, y, Phi = None
                           ):
    # n_s: the number of training examples
    # n_f: the number of features
    # m: the number of predicates
    # P: cvxpy parameter, the psd matrix of the quadratic programming
    # Kernel Matrix
    # d: the coef of the linear term of the loss function
    # Phi: (n_samples ,m)
    # epsilon_star: hyperparameter, 
    # epsilon_s: hyperparameter,
    # y: (0, 1) binary labels 

    n_s = one_n_s.shape[0]
    if Phi is not None:
        m = Phi.shape[1]
        one_m = np.zeros(m)
        dim_variable = 2 * n_s + m + 1
    else: 
        m = 0
        dim_variable = 2 * n_s  + 1

    y_hat = 2 * y - 1 #labels: -1 or 1

    G1 = np.zeros((n_s, dim_variable))
    G1[:, :n_s] = -np.diag(y_hat) @ K
    G1[:, n_s] = -y_hat
    G1[:, -n_s:] = np.eye(n_s)

    h1 = - epsilon_star * one_n_s - 0.5 * y_hat

    if Phi is not None:
        G2 = np.zeros((m, 2 * n_s + m + 1))
        G2[:, :n_s] = Phi.T @ K
        G2[:, n_s] = Phi.T @ one_n_s
        G2[:, n_s + 1 : n_s + m + 1] = np.eye(m)

        h2 = epsilon_s * one_m + Phi.T @ y
        G3 = -G2
        h3 = epsilon_s * one_m - Phi.T @ y

        G4 = np.zeros((m, 2 * n_s + m + 1))
        G4[:, n_s + 1: n_s + m + 1] = -np.eye(m)
        h4 = np.zeros(m)

    G5 = np.zeros((n_s, dim_variable))
    G5[:, -n_s:] = -np.eye(n_s)
    h5 = np.zeros(n_s)

    if Phi is not None:
        G = np.concatenate((G1, G2, G3, G4, G5), axis = 0)
        h = np.concatenate((h1, h2, h3, h4, h5), axis = 0)
    else:
        G = np.concatenate((G1,  G5), axis = 0)
        h = np.concatenate((h1,  h5), axis = 0)

    x = cp.Variable(dim_variable) # The l + 1 + m + 1 variable

    #print("Shape of x:", dim_variable)

    objective_fn = cp.quad_form(x, P) + d @ x

    constraints = [
        G @ x <= h
    ]

    problem = cp.Problem(cp.Minimize(objective_fn), constraints)
    problem.solve()

    A = x.value[:n_s]
    c = x.value[n_s]

    return A, c

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD2(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD2(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False
    
    
def isPD2(B):
    """Returns true when input is positive-definite, via cvxpy"""
    
    try:
        _ = cp.Parameter(shape=(B.shape[0], B.shape[1]), value = B , PSD=True)
        return True
    except:
        return False