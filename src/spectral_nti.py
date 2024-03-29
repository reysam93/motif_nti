import cvxpy as cp
import numpy as np

import utils


def step3(d, lambdas, g_funcs, up_bounds, cs, rs):
    # Step 3: solved with cvx
    N = lambdas.shape[0]
    lambdas_hat = cp.Variable(N-1)  # Fisrt eigenvalue is 0
    contraints = []
    for i, g_func in enumerate(g_funcs):
        expr = g_func(lambdas_hat, N)
        if expr.curvature == "AFFINE":
            contraints.append((expr - cs[i])**2 <= rs['deltas'][i]**2)
        elif expr.curvature == "CONCAVE":
            contraints.append((expr - cs[i]) >= -rs['deltas'][i])
        else:
            contraints.append((expr - cs[i]) <= rs['deltas'][i])

    up_bounds_obj = 0
    for up_bound in up_bounds:
        up_bounds_obj += rs['gamma']*up_bound(lambdas_hat, lambdas[1:], N)

    obj = cp.Minimize(cp.sum(-cp.log(lambdas_hat))
                      + rs['beta']/2*cp.sum_squares(lambdas_hat-d) 
                      + up_bounds_obj)

    prob = cp.Problem(obj, contraints)
    try:
        prob.solve()
    except cp.SolverError:
        # print('WARNING: solver error. Returning lambda from previous iteration.')
        return lambdas, 'solver_error'

    if prob.status not in ['optimal', 'optimal_inaccurate']:
        print('WARNING: problem status', prob.status)
    else:
        lambdas = np.concatenate(([0], lambdas_hat.value))

    return lambdas, prob.status


def MGL(C, g_funcs, up_bounds, cs, regs, max_iters=100, epsilon=1e-4,
           verbose=False):
    """
    Motif graph learning algorithm.
    """
    N = C.shape[0]

    # Check inputs
    if not isinstance(g_funcs, list):
        g_funcs = [g_funcs]
    if not isinstance(cs, list) and np.isscalar(cs):
        cs = [cs]
    if 'deltas' in regs.keys():
        if not isinstance(regs['deltas'], list) and np.isscalar(regs['deltas']):
            regs['deltas'] = [regs['deltas']]
    if not isinstance(up_bounds, list):
        up_bounds = [up_bounds]
    
    K = C + regs['alpha']*(2*np.eye(N)-np.ones((N,N)))

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

        d = np.diag(V.T@L@V)[1:]
        lambdas, prob_status = step3(d, lambdas, g_funcs, up_bounds, cs, regs)

        L_conv = np.linalg.norm(L-prev_L, 'fro')/np.linalg.norm(prev_L, 'fro')
        lambd_conv = np.linalg.norm(lambdas-prev_lam, 2)**2/np.linalg.norm(prev_lam, 2)**2
        prev_lam = lambdas
        prev_L = L

        if verbose:
            sim_Ls = np.linalg.norm(L-V@np.diag(lambdas)@V.T,'fro')
            print('{}. status: {} - L_cnv: {:.6f} - Lam_cnv: {:.3f} - L-VLambdaL\': {:.3f}'.
                format(t, prob_status, L_conv, lambd_conv, sim_Ls))
            
            for i, g_func in enumerate(g_funcs):
                c_aux =  g_func(lambdas, N).value
                print('\t- Bound {}:  g(lamd): {:.3f}   g(lamd)-c: {:.3f}'.
                    format(i, c_aux, np.abs(cs[i] - c_aux)))
        
        if L_conv < epsilon:
            print('CONVERGENCE AT ITERATION:', t)
            break

    return L, lambdas


def step1_GMRF_st(C, S, Theta, eta):
    Theta_hat = cp.Variable(Theta.shape, PSD=True)
    constraints = []
    obj_fn = cp.trace(C @ Theta_hat) - cp.log_det(Theta_hat) + eta*cp.sum_squares(Theta_hat @ S - S @ Theta_hat)

    prob = cp.Problem(cp.Minimize(obj_fn), constraints)
    try:
        prob.solve()
    except cp.SolverError:
        # print('WARNING: solver error. Returning lambda from previous iteration.')
        return Theta, 'solver_error'

    if prob.status not in ['optimal', 'optimal_inaccurate']:
        print('WARNING: problem status', prob.status)
    else:
        Theta = Theta_hat.value

    return Theta, prob.status


def step2_GMRF_st(Theta, V, lambdas, L, regs):
    beta = regs['beta']
    alpha = regs['alpha']
    eta = regs['eta']

    L_eig = V @ np.diag(lambdas) @ V.T
    L_hat = cp.Variable(L.shape, PSD=True)
    constraints = [cp.diag(L_hat) >= 0, L_hat[~np.eye(L_hat.shape[0], dtype=bool)] <= 0]

    obj_fn = beta*cp.sum_squares(L_hat - L_eig) + alpha*cp.sum(cp.diag(L_hat))
    obj_fn += eta*cp.sum_squares(Theta @ L_hat - L_hat @ Theta)

    prob = cp.Problem(cp.Minimize(obj_fn), constraints)
    try:
        prob.solve()
    except cp.SolverError:
        # print('WARNING: solver error. Returning lambda from previous iteration.')
        return L, 'solver_error'

    if prob.status not in ['optimal', 'optimal_inaccurate']:
        print('WARNING: problem status', prob.status)
    else:
        L = L_hat.value

    return L, prob.status
    

def step4_GMRF_st(d, lambdas, g_funcs, up_bounds, cs, rs):
    N = lambdas.shape[0]
    lambdas_hat = cp.Variable(N-1)  # Fisrt eigenvalue is 0
    contraints = []
    for i, g_func in enumerate(g_funcs):
        expr = g_func(lambdas_hat, N)
        if expr.curvature == "AFFINE":
            contraints.append((expr - cs[i])**2 <= rs['deltas'][i]**2)
        elif expr.curvature == "CONCAVE":
            contraints.append((expr - cs[i]) >= -rs['deltas'][i])
        else:
            contraints.append((expr - cs[i]) <= rs['deltas'][i])

    up_bounds_obj = 0
    for up_bound in up_bounds:
        up_bounds_obj += rs['gamma']*up_bound(lambdas_hat, lambdas[1:], N)

    obj = cp.Minimize(rs['beta']/2*cp.sum_squares(lambdas_hat-d) + up_bounds_obj)

    prob = cp.Problem(obj, contraints)
    try:
        prob.solve()
    except cp.SolverError:
        # print('WARNING: solver error. Returning lambda from previous iteration.')
        return lambdas, 'solver_error'

    if prob.status not in ['optimal', 'optimal_inaccurate']:
        print('WARNING: problem status', prob.status)
    else:
        lambdas = np.concatenate(([0], lambdas_hat.value))

    return lambdas, prob.status


def MGL_Stationary_GMRF(C, g_funcs, up_bounds, cs, regs, max_iters=100, epsilon=1e-4,
           verbose=False):
    """
    Motif graph learning algorithm assuming that data comes from a GMRF whose precision matrix
    is given by a polynomial of the GSO
    """
    N = C.shape[0]

    # Check inputs
    if not isinstance(g_funcs, list):
        g_funcs = [g_funcs]
    if not isinstance(cs, list) and np.isscalar(cs):
        cs = [cs]
    if 'deltas' in regs.keys():
        if not isinstance(regs['deltas'], list) and np.isscalar(regs['deltas']):
            regs['deltas'] = [regs['deltas']]
    if not isinstance(up_bounds, list):
        up_bounds = [up_bounds]
    
    L = np.linalg.pinv(C, rcond=1e-6, hermitian=True)
    L = np.where(L > 0, 0, L)
    L[np.eye(N, dtype=bool)] = -1*np.sum(L,1)

    prev_Theta = L
    lambdas, V = np.linalg.eigh(L)
    prev_L = L
    prev_lam = lambdas

    for t in range(max_iters):
        # Step 1: estimate precision matrix Theta
        Theta, _ = step1_GMRF_st(C, L, prev_Theta, regs['eta'])

        # Step 2: estimate L
        L, _ = step2_GMRF_st(Theta, V, lambdas, prev_L, regs)

        # Step 3: eigendecomposition of step 2
        _, V = np.linalg.eigh(L)

        # Step 4: estimate lambdas
        d = np.diag(V.T@L@V)[1:]
        lambdas, prob_status = step4_GMRF_st(d, lambdas, g_funcs, up_bounds, cs, regs)

        L_conv = np.linalg.norm(L-prev_L, 'fro')/np.linalg.norm(prev_L, 'fro')
        lambd_conv = np.linalg.norm(lambdas-prev_lam, 2)**2/np.linalg.norm(prev_lam, 2)**2
        prev_lam = lambdas
        prev_L = L

        regs['eta'] *= regs['inc_eta']

        if verbose:
            sim_Ls = np.linalg.norm(L-V@np.diag(lambdas)@V.T,'fro')
            print('{}. status: {} - L_cnv: {:.6f} - Lam_cnv: {:.3f} - L-VLambdaL\': {:.3f}'.
                format(t, prob_status, L_conv, lambd_conv, sim_Ls))
            
            for i, g_func in enumerate(g_funcs):
                c_aux =  g_func(lambdas, N).value
                print('\t- Bound {}:  g(lamd): {:.3f}   g(lamd)-c: {:.3f}'.
                    format(i, c_aux, np.abs(cs[i] - c_aux)))
        
        if t > 0 and L_conv < epsilon:
            print('CONVERGENCE AT ITERATION:', t)
            break

    return L, lambdas



def SGL_show_conv(C, g_funcs, up_bounds, cs, regs, A_true, 
                  max_iters=500, epsilon=1e-6,
                  verbose=True):
    N = C.shape[0]

    # Check inputs
    if not isinstance(g_funcs, list):
        g_funcs = [g_funcs]
    if not isinstance(cs, list) and np.isscalar(cs):
        cs = [cs]
    if not isinstance(regs['deltas'], list) and np.isscalar(regs['deltas']):
        regs['deltas'] = [regs['deltas']]
    if not isinstance(up_bounds, list):
        up_bounds = [up_bounds]

    K = C + regs['alpha']*(2*np.eye(N)-np.ones((N,N)))
    L = np.linalg.pinv(C, rcond=1e-6, hermitian=True)

    L = np.where(L > 0, 0, L)
    L[np.eye(N, dtype=bool)] = -1*np.sum(L,1)

    prev_L = L
    w = utils.L_inv_op(L)
    lambdas, V = np.linalg.eigh(L)
    
    lamb_prev = lambdas
    A_n = np.linalg.norm(A_true, 'fro')**2
    A_true_n = A_true/np.linalg.norm(A_true, 'fro')
    As = np.zeros((N, N, max_iters+1))
    As[:,:,0] = np.diag(np.diag(L)) - L

    errA = np.zeros(max_iters)
    errA2 = np.zeros(max_iters)
    opt_vals = np.zeros(max_iters)
    sims_Ls = np.zeros(max_iters)
    conv_A = np.zeros(max_iters)
    conv_lamb = np.zeros(max_iters)
    conv_c = np.zeros((max_iters,len(g_funcs)))
    conv_c_n = np.zeros((max_iters,len(g_funcs)))
    for t in range(max_iters):
        # Step 1: closed form solution
        z = utils.Lstar_op(V@np.diag(lambdas)@V.T-1/regs['beta']*K)
        gradient = utils.Lstar_op(utils.L_op(w))-z
        w = np.maximum(0, w-gradient/(2*N))
        L = utils.L_op(w)
        
        if np.all((w == 0)):
            print('WARNING: L_hat is 0. Returning L from previous iteration')
            return prev_L, lamb_prev

        # Step 2: eigendecomposition of step 1
        _, V = np.linalg.eigh(L)

        d = np.diag(V.T@L@V)[1:]
        lambdas, prob_status = step3(d, lambdas, g_funcs, up_bounds, cs, regs)
        
        if prob_status in ['infeasible', 'unbounded', 'infeasible_inaccurate']:
            print('WARNING: problem status', prob_status)
            return prev_L, lamb_prev


        sim_Ls = np.linalg.norm(L-V@np.diag(lambdas)@V.T,'fro')
        opt_vals[t] = -np.log(np.prod(lambdas[1:])) + np.trace(K@L) \
                 + regs['beta']/2*sim_Ls

        L_conv = np.linalg.norm(L-prev_L, 'fro')**2/np.linalg.norm(prev_L, 'fro')**2
        lambd_conv = np.linalg.norm(lambdas-lamb_prev, 2)**2/np.linalg.norm(lamb_prev, 2)**2
        prev_L = L
        if verbose:
            sim_Ls = np.linalg.norm(L-V@np.diag(lambdas)@V.T,'fro')
            print('{}. status: {} - L_cnv: {:.6f} - Lam_cnv: {:.3f} - L-VLambdaL\': {:.3f}'.
                format(t, prob_status, L_conv, lambd_conv, sim_Ls))
            
            for i, g_func in enumerate(g_funcs):
                c_aux =  g_func(lambdas, N).value
                print('\t- Bound {}:  g(lamd): {:.3f}   g(lamd)-c: {:.3f}'.
                    format(i, c_aux, np.abs(cs[i] - c_aux)))
                conv_c[t,i] = np.abs(cs[i] - c_aux)
                conv_c_n[t,i] = np.abs(cs[i] - c_aux)/np.abs(cs[i])

        sims_Ls[t] = sim_Ls**2/np.linalg.norm(L, 'fro')**2
        As[:,:,t+1] = np.diag(np.diag(L)) - L
        errA[t] = np.linalg.norm(A_true-As[:,:,t+1])**2/A_n

        errA2[t] = np.linalg.norm(A_true_n-As[:,:,t+1]/np.linalg.norm(As[:,:,t+1],'fro'))**2

        conv_A[t] = np.linalg.norm(As[:,:,t+1]-As[:,:,t], 'fro')/np.linalg.norm(As[:,:,t], 'fro')
        conv_lamb[t] = np.linalg.norm(lambdas-lamb_prev,2)/np.linalg.norm(lamb_prev, 2)


        conv_lamb[t] = np.linalg.norm(lambdas-lamb_prev, 2)**2/np.linalg.norm(lamb_prev, 2)**2
        lamb_prev = lambdas
        

    return As, opt_vals, sims_Ls, errA, errA2, conv_A, conv_lamb, conv_c_n