from cmath import nan
import numpy as np

from line_serach import fib_searcher, back_forth


def damp_newton_method(func, g_func, G_func, x0, eps=1e-8, n_epochs=1000):
    # Damped Newton's method
    # x0: initial value of x
    # func, g_func, G_func: function, its gradient and hessian respectively
    # eps: epsilon for the variation of function values & gradients to stop iteration
    fib_search_inst = fib_searcher()
    xk, last_fk = x0, 1000000000

    for epoch in range(n_epochs):
        # calculate gk & fk
        gk = g_func(xk)
        if np.linalg.norm(gk) < eps:
            break
        fk = func(xk)
        if np.linalg.norm(fk - last_fk) < eps:
            break
        last_fk = fk

        # dk = -Gk^(-1)*gk
        dk = -np.dot(np.linalg.inv(G_func(x0)), gk)

        # line search for alpha_k
        partial_func = func.get_partial_alpha(xk, dk)
        init_l, init_r = back_forth(partial_func)
        ak = fib_search_inst.fib_search(partial_func, init_l, init_r, 50)

        # update xk
        xk = xk + ak * dk



