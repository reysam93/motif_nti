
import cvxpy as cp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import time
from os import cpu_count
import sys 

sys.path.insert(0, '..')
import utils
import spectral_nti as snti


# CONSTANTS
N_CPUS = cpu_count()
SEED = 28
WEIGHTED = True

GS = [
    lambda a, b : cp.sum(a)/b,
    lambda a, b : cp.sum(a**2)/b,
    lambda a, b : cp.sum(cp.exp(-a))/b,
    lambda a, b : cp.sum(cp.sqrt(a))/b,
    lambda a, b : cp.sum(.25*a**2-.75*a)/b,
]
BOUNDS = [
    lambda lamd, lamd_t, b : -2/b*lamd_t.T@lamd,
    lambda lamd, lamd_t, b : 1/b*cp.exp(-lamd_t).T@lamd,
    lambda lamd, lamd_t, b : cp.sum(lamd/cp.sqrt(lamd_t))/(2*b),
    lambda lamd, lamd_t, b: 1/b*(0.75-2*0.25*lamd_t).T@lamd,
]

DELTAS  = [.04, .27, .003, .02, 0.05]

MODELS = [
    # Ours
    {'name': 'MGL-Tr', 'gs': GS[0], 'bounds': [], 'regs': {'deltas': DELTAS[0]}},
    {'name': 'MGL-Sq', 'gs': GS[1], 'bounds': BOUNDS[0], 'regs': {'deltas': DELTAS[1]}},
    {'name': 'MGL-Heat', 'gs': GS[2], 'bounds': BOUNDS[1], 'regs': {'deltas': DELTAS[2]}},
    {'name': 'MGL-Sqrt', 'gs': GS[3], 'bounds': BOUNDS[2], 'regs': {'deltas': DELTAS[3]}},
    {'name': 'MGL-Poly', 'gs': GS[4], 'bounds': BOUNDS[3], 'regs': {'deltas': DELTAS[4]}},

    # Baselines
    {'name': 'GLasso', 'gs': [], 'bounds': [], 'regs': {}},
    {'name': 'MGL-Tr=1', 'gs': GS[0], 'bounds': [], 'regs': {'deltas': DELTAS[0]}},
    {'name': 'SGL', 'gs': [], 'regs': {'c1': .001, 'c2': 20, 'conn_comp': 1}},  # c1 and c2 obtained from min/max eigenvals 
]


def create_C(lambdas, M, V):
    lambdas_aux = np.insert(1/np.sqrt(lambdas[1:]),0,0)
    C_inv_sqrt = V@np.diag(lambdas_aux)@V.T
    X = C_inv_sqrt@np.random.randn(lambdas.shape[0], M)
    return X@X.T/M


def est_graph(id, alphas, betas, gammas, C_hat,
              model, iters, A, lambdas):
    A_n = np.linalg.norm(A, 'fro')
    lambs_n = np.linalg.norm(lambdas, 2)
    
    err_A = np.zeros((len(gammas), len(betas), len(alphas)))
    err_lam = np.zeros((len(gammas), len(betas), len(alphas)))
    for k, alpha in enumerate(alphas):
        model['regs']['alpha'] = alpha
        for j, beta in enumerate(betas):
            model['regs']['beta'] = beta
            for i, gamma in enumerate(gammas):
                model['regs']['gamma'] = gamma
                
                L_hat, lamd_hat = utils.est_graph(C_hat, model, iters)

                A_hat = np.diag(np.diag(L_hat)) - L_hat
                lamd_hat, _ = np.linalg.eigh(L_hat)

                err_A[i,j,k] = np.linalg.norm(A-A_hat,'fro')**2/A_n**2
                err_lam[i,j,k] = np.linalg.norm(lambdas-lamd_hat)**2/lambs_n**2

                # A_hat /= np.linalg.norm(A_hat, 'fro')
                # lamd_hat /= np.linalg.norm(lamd_hat, 2)
                # err_A[i,j,k] = np.linalg.norm(A/A_n-A_hat,'fro')**2
                # err_lam[i,j,k] = np.linalg.norm(lambdas/lambs_n-lamd_hat)**2

                print('Cov-{}: Alpha {}, Beta {}, Gamma, {}: ErrA: {:.3f}'.
                      format(id, alpha, beta, gamma, err_A[i,j,k]))
    return err_A, err_lam


def plot_err(err, alphas, betas, gammas, label='A'):
    if len(gammas) == 1:
        # Without upper bounds, gamma does not matter
        plt.figure()
        plt.imshow(err[0,:,:])
        plt.colorbar()
        plt.xlabel('Alpha')
        plt.xticks(np.arange(len(alphas)), alphas)
        plt.ylabel('Beta')
        plt.yticks(np.arange(len(betas)), betas)
        plt.title('Err {}, Gamma: {}'.format(label, gammas[0]))
    else:
        # With upper bounds, gamma matter
        for k, alpha in enumerate(alphas):
            plt.figure()
            plt.imshow(err[:,:,k])
            plt.colorbar()
            plt.xlabel('Beta')
            plt.xticks(np.arange(len(betas)), betas)
            plt.ylabel('Gamma')
            plt.yticks(np.arange(len(gammas)), gammas)
            plt.title('Err {}, Alpha: {}'.format(label, alpha))


if __name__ == "__main__":
    np.random.seed(SEED)

    # Regs
    model = MODELS[5]
    alphas = [.001, .005, .01, .05, .1]
    betas = [0]  #np.arange(.25, 3.1, .25)
    gammas = [1000]
    print('Target model:', model['name'])

    # Model params
    n_covs = 10
    iters = 100
    M = 1000

    # Graph params
    n01 = 15
    n02 = 10
    N0 = n01*n02
    n1 = 20
    n2 = 10
    N = n1*n2

    # Create graphs
    A0 = nx.to_numpy_array(nx.grid_2d_graph(n01, n02))
    if WEIGHTED:
        W0 = np.triu(np.random.rand(N0, N0)*3 + .1)
        A0 = A0*(W0 + W0.T)
    L0 = np.diag(np.sum(A0, 0)) - A0
    lambdas0, _ = np.linalg.eigh(L0)
    A = nx.to_numpy_array(nx.grid_2d_graph(n1, n2))
    if WEIGHTED:
        W = np.triu(np.random.rand(N, N)*3 + .1)
        A = A*(W + W.T)
    L = np.diag(np.sum(A, 0)) - A
    lambdas, V = np.linalg.eigh(L)

    if model['name'] == 'MGL-Tr=1':
        model['cs'] = 1
    else:
         model['cs'] = utils.compute_cs(model['gs'], lambdas0, lambdas)

    t = time.time()
    print("CPUs used:", N_CPUS)
    err_A = np.zeros((len(gammas), len(betas), len(alphas), n_covs))
    err_lam = np.zeros((len(gammas), len(betas), len(alphas), n_covs))
    
    pool = Parallel(n_jobs=N_CPUS)
    errs = pool(delayed(est_graph)(i, alphas, betas, gammas, create_C(lambdas, M, V),
                                   model, iters, A, lambdas) for i in range(n_covs))
    for i, err in enumerate(errs):
        err_A[:,:,:,i], err_lam[:,:,:,i] = err
    print('----- {} mins -----'.format((time.time()-t)/60))

    mean_errA = np.mean(err_A, 3)
    mean_errl = np.mean(err_lam, 3)

    # Print
    idx = np.unravel_index(np.argmin(mean_errA), mean_errA.shape)
    print('Min err A (Alpha: {:.3g}, Beta: {:.3g}, Gamma: {:.3g}): {:.6f}\t Err in Lamb: {:.6f}'
        .format(alphas[idx[2]], betas[idx[1]], gammas[idx[0]], mean_errA[idx], mean_errl[idx]))
    idx = np.unravel_index(np.argmin(mean_errl), mean_errl.shape)
    print('Min err Lambd (Alpha: {:.3g}, Beta: {:.3g}, Gamma: {:.3g}): {:.6f}\t Err in A: {:.6f}'
        .format(alphas[idx[2]], betas[idx[1]], gammas[idx[0]], mean_errl[idx], mean_errA[idx]))
    print()

    # Print error with gamma=0
    idx = np.unravel_index(np.argmin(mean_errA[0,:,:]), mean_errA[0,:,:].shape)
    print('Min err A (Alpha: {:.3g}, Beta: {:.3g}, Gamma: {:.3g}): {:.6f}\t Err in Lamb: {:.6f}'
        .format(alphas[idx[1]], betas[idx[0]], 0, mean_errA[0,:,:][idx], mean_errl[0,:,:][idx]))
    idx = np.unravel_index(np.argmin(mean_errl[0,:,:]), mean_errl[0,:,:].shape)
    print('Min err Lambd (Alpha: {:.3g}, Beta: {:.3g}, Gamma: {:.3g}): {:.6f}\t Err in A: {:.6f}'
        .format(alphas[idx[1]], betas[idx[0]], 0, mean_errl[0,:,:][idx], mean_errA[0,:,:][idx]))

    plot_err(mean_errA, alphas, betas, gammas)
    plot_err(mean_errl, alphas, betas, gammas, label='Lambd2')
    plt.show()

    data = {
        'alphas': alphas,
        'betas': betas,
        'gammas': gammas,
        'iters': iters,
        'err_A': err_A,
        'err_lam': err_lam
    }

    # np.save('tmp\params_heat_M1000_i200_err', data)
