
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

GS = [
    lambda a, b : cp.sum(a)/b,    # delta: 4e-2
    #lambda a, b : cp.sum(a**2)/b,  # delta: .7
    # lambda a, b : cp.sum(cp.exp(-a))/b,    # delta: 3e-3
    #lambda a, b : cp.sum(cp.sqrt(a))/b,  # delta: 2e-2
    #lambda a, b : cp.sum(cp.exp(.5*a))/b,
    #lambda a, b : cp.sum(.25*a**2-.75*a)/b,
]
BOUNDS = [
    #lambda lamd, lamd_t, b : -2/b*lamd_t.T@lamd,
    # lambda lamd, lamd_t, b : 1/b*cp.exp(-lamd_t).T@lamd,
    #lambda lamd, lamd_t, b : cp.sum(lamd/cp.sqrt(lamd_t))/(2*b),
    #lambda lamd, lamd_t, b : -.5/b*cp.exp(lamd_t).T@lamd,
    # lambda lamd, lamd_t, b: 1/b*(0.75-2*0.25*lamd_t).T@lamd,
]


def create_C(lambdas, M, V):
    lambdas_aux = np.insert(1/np.sqrt(lambdas[1:]),0,0)
    C_inv_sqrt = V@np.diag(lambdas_aux)@V.T
    X = C_inv_sqrt@np.random.randn(lambdas.shape[0], M)
    return X@X.T/M


def est_graph(id, alphas, betas, gammas, deltas, C_hat,
              cs, iters, A, lambdas):
    A_n = np.linalg.norm(A, 'fro')
    lambs_n = np.linalg.norm(lambdas, 2)
    
    regs = {'alpha': 0, 'beta': 0, 'gamma': 0, 'deltas': deltas}

    err_A = np.zeros((len(gammas), len(betas), len(alphas)))
    err_lam = np.zeros((len(gammas), len(betas), len(alphas)))
    for k, alpha in enumerate(alphas):
        regs['alpha'] = alpha
        for j, beta in enumerate(betas):
            regs['beta'] = beta
            for i, gamma in enumerate(gammas):
                regs['gamma'] = gamma
                L_hat, _ = snti.SGL_MM(C_hat, GS, BOUNDS, cs,
                                       regs, max_iters=iters)

                A_hat = np.diag(np.diag(L_hat)) - L_hat
                lamd_hat, _ = np.linalg.eigh(L_hat)

                # err_A[i,j,k] = np.linalg.norm(A-A_hat,'fro')**2/A_n**2
                # err_lam[i,j,k] = np.linalg.norm(lambdas-lamd_hat)**2/lambs_n**2

                A_hat /= np.linalg.norm(A_hat, 'fro')
                lamd_hat /= np.linalg.norm(lamd_hat, 2)
                err_A[i,j,k] = np.linalg.norm(A/A_n-A_hat,'fro')**2
                err_lam[i,j,k] = np.linalg.norm(lambdas/lambs_n-lamd_hat)**2

                print('Cov-{}: Alpha {}, Beta {}, Gamma, {}: ErrA: {:.3f}'.
                      format(id, alpha, beta, gamma, err_A[i,j,k]))
    return err_A, err_lam


def plot_err(err, alphas, betas, gammas, label='A'):
    if len(BOUNDS) == 0:
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
            plt.title('Err {}, Alpha: {}'.format(label, alphas[k]))


if __name__ == "__main__":
    np.random.seed(SEED)

    # Regs
    alphas = [1e-3, 5e-3]
    betas = [0.5]  #np.arange(.25, 3.1, .25)
    gammas = [0, 10, 25, 50, 100, 150]

    # Model params
    n_covs = 10
    iters = 200
    M = 500

    #deltas  = [4e-2, .27, 3e-3, 2e-2, 6.5, 0.05]
    deltas = [0.04]

    # Graph params
    n01 = 15
    n02 = 10
    n1 = 20
    n2 = 10

    # Create graphs
    A0 = nx.to_numpy_array(nx.grid_2d_graph(n01, n02))
    L0 = np.diag(np.sum(A0, 0)) - A0
    lambdas0, _ = np.linalg.eigh(L0)
    A = nx.to_numpy_array(nx.grid_2d_graph(n1, n2))
    L = np.diag(np.sum(A, 0)) - A
    lambdas, V = np.linalg.eigh(L)

    cs = utils.compute_cs(GS, lambdas0, lambdas)
    t = time.time()
    print("CPUs used:", N_CPUS)
    err_A = np.zeros((len(gammas), len(betas), len(alphas), n_covs))
    err_lam = np.zeros((len(gammas), len(betas), len(alphas), n_covs))
    
    pool = Parallel(n_jobs=N_CPUS)
    errs = pool(delayed(est_graph)(i, alphas, betas, gammas, deltas, create_C(lambdas, M, V),
                                      cs, iters, A, lambdas) for i in range(n_covs))
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
        'deltas': deltas,
        'iters': iters,
        'err_A': err_A,
        'err_lam': err_lam
    }

    # np.save('tmp\params_heat_M1000_i200_err', data)
