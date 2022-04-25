import numpy as np


def conjugate_gradient_two(func, g_func, G_func, x0, beta_func, line_searcher, eps=1e-8, n_epochs=100, verbose=False):
    # Basic Conjugate Gradient method
    # xk+1 = xk + alpha * dk
    # where dk+1 = dk + beta_func(gk, gk-1) * gk
    # where beta_func denotes the function to get beta in CG, \in {FR, PRP}
    # x0: initial value of x
    # H0: initial positive-definite value of Hk
    # func, g_func, G_func: function, its gradient and hessian respectively
    # eps: epsilon for the variation of function values & gradients to stop iteration
    
    xk, last_fk = x0, 1000000000
    prev_fks = []

    for epoch in range(n_epochs):
        # calculate gk & fk
        gk = g_func(xk)
        fk = func(xk)
        if np.linalg.norm(fk - last_fk) < eps and np.linalg.norm(gk) < eps:
            break

        # update dk
        if epoch == 0:
            dk = -gk
        else:
            # dk = -gk + beta_{k-1} * d_{k-1}
            last_beta_k = beta_func(gk, last_gk)
            dk = -gk + last_beta_k * last_dk
        last_dk = dk
        last_gk = gk
        last_fk = fk
        
        prev_fks.append(fk)

        # line search for alpha_k
        partial_func = func.get_partial_alpha(xk, dk)
        args = {
            'n_func_calls': 20,
            'prev_fks': prev_fks,
            'gk': gk,
            'dk': dk,
        }
        ak = line_searcher.pipeline(partial_func, args)

        # update xk
        xk = xk + ak * dk
        if verbose:
            print('[{}] fk={:.5f}, |gk|={:.8f}'.format(epoch, fk, np.linalg.norm(gk)))
    return xk, epoch


def fr_func(gk, last_gk):
    # Fletcher-Reeves function for beta_k
    return np.dot(gk, gk) / np.dot(last_gk, last_gk)


def prp_func(gk, last_gk):
    # Polak-Ribiere-Polyak function for beta_k
    return np.dot(gk, (gk - last_gk)) / np.dot(last_gk, last_gk)
