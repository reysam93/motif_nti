import numpy as np
from sklearn.covariance import graphical_lasso

import spectral_nti as snti
import baselines

# Laplacian operator    
def L_op(w):
    k = w.size
    N = int((1 + np.sqrt(1+8*k))/2)
    
    idx = np.triu_indices(N, k=1)
    A = np.zeros((N,N))
    A[idx] = w
    A = A + A.T
    return np.diag(np.sum(A,0)) - A


# Inverse of the Laplacian operator
def L_inv_op(L):
    N = L.shape[0]
    idx = np.triu_indices(N, k=1)
    return -L[idx]


# Adjoint of the Laplacian operator (from the paper of Palomar)
def Lstar_op(Y):
    N = Y.shape[1]
    K = int(N*(N-1)/2)
    j = 0
    i = 1
    w = np.zeros(K)
    for k in range(K):
        w[k] = Y[i,i] + Y[j,j] - (Y[i,j] + Y[j,i])
        if (i == (N-1)):
            j += 1
            i = j + 1
        else:
            i += 1
    return w


# Adjacency operator    
def A_op(w):
    k = w.size
    N = int((1 + np.sqrt(1+8*k))/2)
    
    idx = np.triu_indices(N, k=1)
    A = np.zeros((N,N))
    A[idx] = w
    return A + A.T


# Inverse of the Adjacency operator
def A_inv_op(A):
    N = A.shape[0]
    idx = np.triu_indices(N, k=1)
    return A[idx]


# Adjoint of the Adjacency operator
def Astar_op(Y):
    N = Y.shape[1]
    K = int(N*(N-1)/2)
    j = 0
    i = 1
    w = np.zeros(K)
    for k in range(K):
        w[k] = Y[i,j] + Y[j,i]
        if (i == (N-1)):
            j += 1
            i = j + 1
        else:
            i += 1
    return w


def compute_cs(gs, lambdas0, lambdas, verbose=False):
    N0 = lambdas0.shape[0]
    N = lambdas.shape[0]

    if not isinstance(gs, list):
        gs = [gs]

    cs = np.zeros(len(gs))
    errs = np.zeros(len(gs))
    for i, g in enumerate(gs):
        cs[i] = g(lambdas0[1:], N0).value
        c_aux = g(lambdas[1:], N).value
        errs[i] = c_aux-cs[i]
        if verbose:
            print('\tc-{}: c: {:.3f}\tc0: {:.3f}\terr: {:.6f}\terr norm: {:.6f}'
                .format(i, c_aux, cs[i], errs[i], errs[i]/cs[i]))
    return cs, np.abs(errs)


def est_graph(C_hat, model, iters):
    if model['name'] == 'Pinv':
        L_hat = np.linalg.pinv(C_hat)

    elif model['name'] == 'GLasso':
        if 'alpha' in model:
            alpha = model['alpha']
        else:
            alpha = model['regs']['alpha']
        try:
            _, L_hat = graphical_lasso(C_hat, alpha, max_iter=iters)
        except FloatingPointError:
            print('WARNING: Floating Point Error: Non SPD result.')
            return np.ones(C_hat.shape), np.ones(C_hat.shape[0])
     
    elif model['name'] == 'SGL':
        L_hat, _ = baselines.SGL(C_hat, model['regs'], max_iters=iters)

    else:
        L_hat, _ = snti.MGL(C_hat, model['gs'], model['bounds'], model['cs'],
                            model['regs'], max_iters=iters)

    lamd_hat, _ = np.linalg.eigh(L_hat)

    return L_hat, lamd_hat