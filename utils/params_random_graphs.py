
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
# G_TYPE in ['SW', 'SBM']
G_TYPE = 'SW'
WEIGHTED = False
BETTER_DELTAS = False
TRUE_GRAPH = True       # Set the true graph as the reference graph

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
if G_TYPE == 'SW':
    DELTAS = [1e-4, .3, 0.005, .1]
    # DELTAS = [0, .3, 0.005, .1]
else:
    DELTAS = [2.9, 45, 0.06, 11]


MODELS = [
    # Ours
    {'name': 'MGL-Tr', 'gs': GS[0], 'bounds': [], 'regs': {'deltas': DELTAS[0]}},
    {'name': 'MGL-Sq', 'gs': GS[1], 'bounds': BOUNDS[0], 'regs': {'deltas': DELTAS[1]}},
    {'name': 'MGL-Heat', 'gs': GS[2], 'bounds': BOUNDS[1], 'regs': {'deltas': DELTAS[2]}},
    {'name': 'MGL-Poly', 'gs': GS[3], 'bounds': BOUNDS[2], 'regs': {'deltas': DELTAS[3]}},

    # Baselines
    {'name': 'GLasso', 'gs': [], 'bounds': [], 'regs': {}},
    {'name': 'MGL-Tr=1', 'gs': GS[0], 'bounds': [], 'regs': {'deltas': 1e-4}},
    {'name': 'SGL', 'gs': [], 'regs': {'c1': 1, 'c2': 25, 'conn_comp': 1}},  # c1 and c2 obtained from min/max eigenvals 
    {'name': 'Unconst', 'gs': [], 'bounds': [], 'regs': {'deltas': []}},
    {'name': 'Pinv', 'gs': [], 'bounds': [], 'regs': {}}
]


def est_params(id, alphas, betas, gammas, model, graphs, M, 
              iters, lambdas0):
    # Create graph
    if G_TYPE == 'SW':
        A = nx.to_numpy_array(
            nx.watts_strogatz_graph(graphs['N'], graphs['k'], graphs['p']))
    elif G_TYPE == 'SBM':
        A = nx.to_numpy_array(
            # nx.random_partition_graph(graphs['block_sizes'], graphs['p'], graphs['q']))
            nx.random_partition_graph(graphs['block_sizes0'], graphs['p'], graphs['q']))

    if WEIGHTED:
        W = np.triu(np.random.rand(graphs['N'], graphs['N'])*3 + .1)
        A = A*(W + W.T)

    L = np.diag(np.sum(A, 0)) - A
    lambdas, _ = np.linalg.eigh(L)

    L_n = np.linalg.norm(L, 'fro')
    lambs_n = np.linalg.norm(lambdas, 2)

    if TRUE_GRAPH:
        model['regs']['deltas'] = 1e-4
        lambdas0 = lambdas

    X = utils.create_signals(L, M)
    C_hat = X@X.T/M

    if model['name'] == 'MGL-Tr=1':
        model['cs'] = [1]
    else:
        model['cs'], err_cs = utils.compute_cs(model['gs'], lambdas0, lambdas, True)    
        if BETTER_DELTAS:
            model['regs']['deltas'] = err_cs*1.1

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
    assert G_TYPE in ['SW', 'SBM'], 'Unkown graph type.'

    # Regs
    model = MODELS[3]
    alphas = [0] #[0, .001, .005, .01, .05]
    betas =  np.arange(1, 2.1, .1)  #np.concatenate((np.arange(.1, 1.6, .1), [2, 5, 10, 25, 30]))
    gammas =  [500, 1000, 2500, 5000, 1e4]
    print('Target model:', model['name'], 'Graph type:', G_TYPE)

    # Model params
    n_graphs = 10
    iters = 200
    M = 1000

    # Create graphs
    graphs = {'B': 1}
    if G_TYPE == 'SW':
        # SW graph params
        graphs['N0'] = 150
        graphs['N'] = 100
        graphs['k'] = 4
        graphs['p'] = .1
        A0 = nx.to_numpy_array(
             nx.watts_strogatz_graph(graphs['N0'], graphs['k'], graphs['p'], seed=SEED))

        if model['name'] == 'SGL':
            model['regs']['c1'] = 0.01
            model['regs']['c2'] =  graphs['k']*2.5
    elif G_TYPE == 'SBM':
        # SBM graph params
        graphs['B'] = 5
        graphs['block_sizes0'] = [30]*graphs['B']
        graphs['block_sizes'] = [30]*graphs['B']
        graphs['p'] = .3
        graphs['q0'] = 0
        graphs['q'] = 0
        graphs['N0'] = sum(graphs['block_sizes0'])
        graphs['N'] = sum(graphs['block_sizes'])
        A0 = nx.to_numpy_array(
             nx.random_partition_graph(graphs['block_sizes0'], graphs['p'], graphs['q0'], seed=SEED))
        if model['name'] == 'SGL':
            model['regs']['conn_comp'] = graphs['B']
            model['regs']['c1'] = 1
            model['regs']['c2'] = 20

    if WEIGHTED:
        W0 = np.triu(np.random.rand(graphs['N0'], graphs['N0'])*3 + .1)
        A0 = A0*(W0 + W0.T)

    L0 = np.diag(np.sum(A0, 0)) - A0
    lambdas0, _ = np.linalg.eigh(L0)

    t = time.time()
    print("CPUs used:", N_CPUS)
    err_L = np.zeros((len(gammas), len(betas), len(alphas), n_graphs))
    err_lam = np.zeros((len(gammas), len(betas), len(alphas), n_graphs))
    
    pool = Parallel(n_jobs=N_CPUS)
    errs = pool(delayed(est_params)(i, alphas, betas, gammas, model, graphs, M,
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
    plt.show()
