from tabnanny import verbose
import numpy as np


def damp_newton_method(func, g_func, G_func, x0, line_searcher, eps=1e-8, n_epochs=100, use_cholesky_correction=False, verbose=False):
    # Damped Newton's method
    # x0: initial value of x
    # func, g_func, G_func: function, its gradient and hessian respectively
    # eps: epsilon for the variation of function values & gradients to stop iteration
    # use_cholesky_correction: if True, use cholesky correction for newton's method, i.e. perform cholesky decomposition on Gk+vI to avoid not postive-definite Gk
    xk, last_fk = x0, 1000000000
    prev_fks = []

    for epoch in range(n_epochs):
        # calculate gk & fk
        gk = g_func(xk)
        fk = func(xk)
        if np.linalg.norm(fk - last_fk) < eps and np.linalg.norm(gk) < eps:
            break
        last_fk = fk
        prev_fks.append(fk)

        # dk = -Gk^(-1)*gk
        Bk = G_func(xk)
        if use_cholesky_correction:
            Bk = cholesky_correction(Bk)
        dk = -np.dot(np.linalg.inv(Bk), gk)

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


def cholesky_correction(Gk, beta=1e-3, sigma=5):
    # Use cholesky decomposition to decide the minimum correction factor tau
    # Returns the corrected Bk = Gk + tau * I for Gk
    I = np.eye(len(Gk))
    min_diag = np.diag(Gk).min()
    if min_diag > 0:
        tau = 0
    else:
        tau = -min_diag + beta
    
    for ix in range(20):
        cur_Bk = Gk + tau * I
        try:
            L = np.linalg.cholesky(cur_Bk)
        except np.linalg.LinAlgError:
            # Cholesky failed, maybe not postive-definite, enlarge tau then
            tau = max(sigma * tau, beta)
            continue
        break
    return cur_Bk

