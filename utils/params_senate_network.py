
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import time
import sys
from os import cpu_count

sys.path.insert(0, '..')
import src.utils as utils

# CONSTANTS
N_CPUS = cpu_count()
SEED = 28

DATASET_PATH = '../data/senate_data/'

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

# DELTAS = [.86, 9, .05, 1.7]
# C1 = 0.01
# C2 = 20
DELTAS = [1.2, 13, .06, 2.3]
C1 = 0.05     # True: 0.07
C2 = 16.5     # True: 16.37

MODELS = [
    # Ours
    {'name': 'MGL-Tr', 'gs': GS[0], 'bounds': [], 'regs': {'deltas': DELTAS[0]}},
    {'name': 'MGL-Sq', 'gs': GS[1], 'bounds': BOUNDS[0], 'regs': {'deltas': DELTAS[1]}},
    {'name': 'MGL-Heat', 'gs': GS[2], 'bounds': BOUNDS[1], 'regs': {'deltas': DELTAS[2]}},
    {'name': 'MGL-Poly', 'gs': GS[3], 'bounds': BOUNDS[2], 'regs': {'deltas': DELTAS[3]}},

    # Baselines
    {'name': 'GLasso', 'gs': [], 'bounds': [], 'regs': {}},
    {'name': 'MGL-Tr=1', 'gs': GS[0], 'bounds': [], 'regs': {'deltas': 1e-4}},
    {'name': 'SGL', 'gs': [], 'regs': {'c1': C1, 'c2': C2, 'conn_comp': 1}},
    {'name': 'Unconst', 'gs': [], 'bounds': [], 'regs': {'deltas': []}},

    # Combinations
    {'name': 'MGL-Poly+Heat+Tr', 'gs': [GS[0], GS[2], GS[3]],
     'bounds': [BOUNDS[0], BOUNDS[1], BOUNDS[2]], 
     'regs': {'deltas': [DELTAS[0], DELTAS[2], DELTAS[3]]}},
]


def est_params(id, alphas, betas, gammas, model, X_M, L, 
               lambdas, iters):
    L_n = np.linalg.norm(L, 'fro')
    lambs_n = np.linalg.norm(lambdas, 2)

    # C_hat = np.cov(X[:,1:M])
    C_hat = np.cov(X_M)
    # print('Norm C:', np.linalg.norm(C_hat))

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
    model = MODELS[3]
    alphas = [0]
    betas = np.concatenate((np.arange(.1, 1.1, .1), [5]))
    gammas = [500, 1e3, 5e3, 1e4]
    print('Target model:', model['name'])

    # Model params
    n_covs = 50
    iters = 500
    M = 200

    # Read graphs and signals
    X = np.load(DATASET_PATH + 'X115.npy')
    L0 = np.load(DATASET_PATH + 'L114.npy')
    lambdas0, _ = np.linalg.eigh(L0)
    L = np.load(DATASET_PATH + 'L115.npy')
    lambdas, _ = np.linalg.eigh(L)

    X_perms = [np.random.permutation(X.T).T for i in range(n_covs)]

    conn_comp = np.sum(lambdas <= 1e-6)
    if model['name'] == 'SGL':
        model['regs']['conn_comp'] = conn_comp

    print('Connected components:', conn_comp)
    print('Max eigv:', lambdas[-1], 'Min eigv:', lambdas[conn_comp])

    if model['name'] != 'MGL-Tr=1':
        model['cs'], err_cs = utils.compute_cs(model['gs'], lambdas0, lambdas, True)
    else:
        model['cs'] = 1

    t = time.time()
    print("CPUs used:", N_CPUS)
    err_L = np.zeros((len(gammas), len(betas), len(alphas), n_covs))
    err_lam = np.zeros((len(gammas), len(betas), len(alphas), n_covs))
    
    pool = Parallel(n_jobs=N_CPUS)
    errs = pool(delayed(est_params)(i, alphas, betas, gammas, model, X_perms[i][:,:M],
                                    L, lambdas, iters) for i in range(n_covs))

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
    plt.show()
