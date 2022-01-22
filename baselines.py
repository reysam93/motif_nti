from pyexpat import model
import cvxpy as cp
import numpy as np

import utils


def step3(d, lambdas, rs):
    # Step 3: solved with cvx
    N = lambdas.shape[0]
    lambdas_hat = cp.Variable(N-rs['conn_comp'])  # Fisrt eigenvalues are 0
    
    # contraints = []
    # for i in range(lambdas_hat.size):
    #     if i == (lambdas_hat.size-1):
    #         contraints.append(lambdas_hat[i] <= rs['c2'])
    #         continue
    #     if i == 0:
    #         contraints.append(rs['c1'] <= lambdas_hat[0])
            
    #     contraints.append(lambdas_hat[i] <= lambdas_hat[i+1])

    # We prefer these contraints to the one's above since they are faster and we 
    # checked that the error is similar
    contraints = [lambdas_hat >= rs['c1'], lambdas_hat <= rs['c2']]

    obj = cp.Minimize(cp.sum(-cp.log(lambdas_hat)) \
         + rs['beta']/2*cp.sum_squares(lambdas_hat-d))
    prob = cp.Problem(obj, contraints)

    try:      
        prob.solve()
    except cp.SolverError:
        print('WARNING: solver error.')
        return lambdas, 'solver_error'

    if prob.status not in ['optimal', 'optimal_inaccurate']:
        print('WARNING: problem status', prob.status)
    else:
        lambdas = np.concatenate(([0]*rs['conn_comp'], lambdas_hat.value))

    return lambdas


def SGL(C, regs, max_iters=100, epsilon=1e-4):
    """
    Implementation of the spectral learning algorithm proposed in 'Structured Graph Learning 
    Via Laplacian Spectral Constraints', from Kumar S, Ying J, Cardoso J, Palomar D.
    """
    N = C.shape[0]
    K = C + regs['alpha']*(2*np.eye(N)-np.ones((N,N)))

    # Naive initialization
    L = np.linalg.pinv(C, rcond=1e-6, hermitian=True)
    L = np.where(L > 0, 0, L)
    L[np.eye(N, dtype=bool)] = -1*np.sum(L,1)

    w = utils.L_inv_op(L)
    lambdas, V = np.linalg.eigh(L)
    prev_L = L
    prev_lam = lambdas

    for t in range(max_iters):
        # Step 1: closed form solution
        z = utils.Lstar_op(V@np.diag(lambdas)@V.T-1/regs['beta']*K)
        gradient = utils.Lstar_op(utils.L_op(w))-z
        w = np.maximum(0, w-gradient/(2*N))
        L = utils.L_op(w)
        if np.all((w == 0)):
            print('WARNING: L_hat is 0. Returning L from previous iteration')
            return prev_L, prev_lam

        # Step 2: eigendecomposition of step 1
        _, V = np.linalg.eigh(L)

        d = np.diag(V.T@L@V)[regs['conn_comp']:]
        lambdas = step3(d, lambdas, regs)

        L_conv = np.linalg.norm(L-prev_L, 'fro')/np.linalg.norm(prev_L, 'fro')
        prev_L = L
        prev_lam = lambdas
        
        if L_conv < epsilon:
            print('CONVERGENCE AT ITERATION:', t)
            break

    return L, lambdas