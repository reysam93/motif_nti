
import cvxpy as cp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import time
import sys
from os import cpu_count

sys.path.insert(0, '..')
import utils
import spectral_nti as snti


# CONSTANTS
N_CPUS = cpu_count()
SEED = 28
SEED2 = 14
SAME_GRAPHS = False
WEIGHTED = True

GS = [
    lambda a, b : cp.sum(a)/b,
    lambda a, b : cp.sum(a**2)/b,
    lambda a, b : cp.sum(cp.exp(-a))/b,
    lambda a, b : cp.sum(.25*a**2-.75*a)/b,
]
BOUNDS = [
    lambda lamd, lamd_t, b : -2/b*lamd_t.T@lamd,
    lambda lamd, lamd_t, b : 1/b*cp.exp(-lamd_t).T@lamd,
    lambda lamd, lamd_t, b: 1/b*(0.75-2*0.25*lamd_t).T@lamd,
]

# Create deltas as a dict indexed by k?
DELTAS = [1e-3, .3, 0.005, .1]

MODELS = [
    # Ours
    {'name': 'MGL-Tr', 'gs': GS[0], 'bounds': [], 'regs': {'deltas': DELTAS[0]}},
    {'name': 'MGL-Sq', 'gs': GS[1], 'bounds': BOUNDS[0], 'regs': {'deltas': DELTAS[1]}},
    {'name': 'MGL-Heat', 'gs': GS[2], 'bounds': BOUNDS[1], 'regs': {'deltas': DELTAS[2]}},
    {'name': 'MGL-Poly', 'gs': GS[3], 'bounds': BOUNDS[2], 'regs': {'deltas': DELTAS[3]}},

    # Baselines
    {'name': 'GLasso', 'gs': [], 'bounds': [], 'regs': {}},
    {'name': 'MGL-Tr=1', 'gs': GS[0], 'bounds': [], 'regs': {'deltas': DELTAS[0]}},
    {'name': 'SGL', 'gs': [], 'regs': {'c1': .1, 'c2': 30, 'conn_comp': 1}},  # c1 and c2 obtained from min/max eigenvals 
]


def create_C(lambdas, M, V):
    N = lambdas.size
    lambdas_aux = np.insert(1/np.sqrt(lambdas[1:]),0,0)
    C_inv_sqrt = V@np.diag(lambdas_aux)@V.T
    # The shape of X is MxN, so it is X.T according to our notation
    X = np.random.multivariate_normal(np.zeros(N), C_inv_sqrt, M)
    return X.T@X/M


def est_graph(id, alphas, betas, gammas, model, N, k, p, M, 
              iters, lambdas0):
    # Create graph
    if SAME_GRAPHS:
        A = nx.to_numpy_array(nx.watts_strogatz_graph(N, k, p, seed=SEED2))
    else:
        A = nx.to_numpy_array(nx.watts_strogatz_graph(N, k, p))
    if WEIGHTED:
        W = np.triu(np.random.rand(N, N)*3 + .1)
        A = A*(W + W.T)
    L = np.diag(np.sum(A, 0)) - A
    lambdas, V = np.linalg.eigh(L)
    L_n = np.linalg.norm(L, 'fro')
    lambs_n = np.linalg.norm(lambdas, 2)
    C_hat = create_C(lambdas, M, V)

    if model['name'] == 'MGL-Tr=1':
        model['cs'] = 1
    else:
        model['cs'] = utils.compute_cs(model['gs'], lambdas0, lambdas)    

    err_L = np.zeros((len(gammas), len(betas), len(alphas)))
    err_lam = np.zeros((len(gammas), len(betas), len(alphas)))
    for k, alpha in enumerate(alphas):
        model['regs']['alpha'] = alpha
        for j, beta in enumerate(betas):
            model['regs']['beta'] = beta
            for i, gamma in enumerate(gammas):
                model['regs']['gamma'] = gamma

                L_hat, lamd_hat = utils.est_graph(C_hat, model, iters)

                # err_L[i,j,k] = np.linalg.norm(L-L_hat,'fro')**2/L_n**2
                # err_lam[i,j,k] = np.linalg.norm(lambdas-lamd_hat)**2/lambs_n**2

                L_hat /= np.linalg.norm(L_hat, 'fro')
                lamd_hat /= np.linalg.norm(lamd_hat, 2)
                err_L[i,j,k] = np.linalg.norm(L/L_n-L_hat,'fro')**2
                err_lam[i,j,k] = np.linalg.norm(lambdas/lambs_n-lamd_hat)**2

                print('Cov-{}: Alpha {}, Beta {}, Gamma, {}: ErrL: {:.3f}'.
                      format(id, alpha, beta, gamma, err_L[i,j,k]))
    plt.show()
    return err_L, err_lam


def plot_err(err, alphas, betas, gammas, label='L'):
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
    model = MODELS[1]
    alphas = [0, .001, .01, .1]
    betas = np.concatenate((np.arange(.1, .6, .1), [1, 5, 10]))
    gammas = [0, 1, 5, 10, 25, 50, 100]
    print('Target model:', model['name'])

    # Model params
    n_graphs = 10
    iters = 200
    M = 500

    # Graph params
    N0 = 150
    N = 100
    k = 8
    p = .1

    # Create graphs
    A0 = nx.to_numpy_array(nx.watts_strogatz_graph(N0, k, p, seed=SEED))
    if WEIGHTED:
        W0 = np.triu(np.random.rand(N0, N0)*3 + .1)
        A0 = A0*(W0 + W0.T)
    L0 = np.diag(np.sum(A0, 0)) - A0
    lambdas0, _ = np.linalg.eigh(L0)

    A = nx.to_numpy_array(nx.watts_strogatz_graph(N, k, p, seed=SEED))
    L = np.diag(np.sum(A, 0)) - A
    lambdas, V = np.linalg.eigh(L)

    t = time.time()
    print("CPUs used:", N_CPUS)
    err_L = np.zeros((len(gammas), len(betas), len(alphas), n_graphs))
    err_lam = np.zeros((len(gammas), len(betas), len(alphas), n_graphs))
    
    pool = Parallel(n_jobs=N_CPUS)
    errs = pool(delayed(est_graph)(i, alphas, betas, gammas, model, N, k, p, M,
                                      iters, lambdas0) for i in range(n_graphs))
    for i, err in enumerate(errs):
        err_L[:,:,:,i], err_lam[:,:,:,i] = err
    print('----- {} mins -----'.format((time.time()-t)/60))

    mean_errA = np.mean(err_L, 3)
    mean_errl = np.mean(err_lam, 3)

    # Print
    idx = np.unravel_index(np.argmin(mean_errA), mean_errA.shape)
    print('Min err L (Alpha: {:.3g}, Beta: {:.3g}, Gamma: {:.3g}): {:.6f}\t Err in Lamb: {:.6f}'
        .format(alphas[idx[2]], betas[idx[1]], gammas[idx[0]], mean_errA[idx], mean_errl[idx]))
    idx = np.unravel_index(np.argmin(mean_errl), mean_errl.shape)
    print('Min err Lambd (Alpha: {:.3g}, Beta: {:.3g}, Gamma: {:.3g}): {:.6f}\t Err in L: {:.6f}'
        .format(alphas[idx[2]], betas[idx[1]], gammas[idx[0]], mean_errl[idx], mean_errA[idx]))
    print()

    # Print error with gamma=0
    idx = np.unravel_index(np.argmin(mean_errA[0,:,:]), mean_errA[0,:,:].shape)
    print('Min err L (Alpha: {:.3g}, Beta: {:.3g}, Gamma: {:.3g}): {:.6f}\t Err in Lamb: {:.6f}'
        .format(alphas[idx[1]], betas[idx[0]], 0, mean_errA[0,:,:][idx], mean_errl[0,:,:][idx]))
    idx = np.unravel_index(np.argmin(mean_errl[0,:,:]), mean_errl[0,:,:].shape)
    print('Min err Lambd (Alpha: {:.3g}, Beta: {:.3g}, Gamma: {:.3g}): {:.6f}\t Err in L: {:.6f}'
        .format(alphas[idx[1]], betas[idx[0]], 0, mean_errl[0,:,:][idx], mean_errA[0,:,:][idx]))

    plot_err(mean_errA, alphas, betas, gammas)
    plot_err(mean_errl, alphas, betas, gammas, label='Lambd2')
    plt.show()

    # data = {
    #     'alphas': alphas,
    #     'betas': betas,
    #     'gammas': gammas,
    #     # 'deltas': deltas,
    #     'iters': iters,
    #     'err_L': err_L,
    #     'err_lam': err_lam
    # }

    # np.save('tmp\params_heat_M1000_i200_err', data)
