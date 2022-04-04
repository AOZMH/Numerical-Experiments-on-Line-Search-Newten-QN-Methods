import numpy as np

from line_search import fib_searcher, back_forth


def quasi_newton_method(func, g_func, G_func, x0, H0, eps=1e-8, n_epochs=1000, Hk_update_func=None):
    # The framework of quasi-newton method
    # x0: initial value of x
    # H0: initial positive-definite value of Hk
    # func, g_func, G_func: function, its gradient and hessian respectively
    # eps: epsilon for the variation of function values & gradients to stop iteration
    # Hk_update_func: update method for a specific quasi-newton method

    fib_search_inst = fib_searcher()
    xk, last_fk, Hk, last_gk = x0, 1000000000, H0, None

    for epoch in range(n_epochs):
        # calculate gk & fk
        gk = g_func(xk)
        if np.linalg.norm(gk) < eps:
            break
        fk = func(xk)
        if np.linalg.norm(fk - last_fk) < eps:
            break
        
        # update Hk
        if epoch > 0:
            sk = fk - last_fk   # sk = fk+1 - fk
            yk = gk - last_gk   # gk = gk+1 - gk
            Hk = Hk_update_func(Hk, sk, yk)
        last_fk = fk
        last_gk = gk

        # dk = -Hk*gk
        dk = -np.dot(Hk, gk)

        # line search for alpha_k
        partial_func = func.get_partial_alpha(xk, dk)
        init_l, init_r = back_forth(partial_func)
        ak = fib_search_inst.fib_search(partial_func, init_l, init_r, 50)

        # update xk
        xk = xk + ak * dk
        print('[{}] fk={:.5f}, |gk|={:.5f}'.format(epoch, fk, np.dot(gk, gk)))


def sr1_update_func(Hk, sk, yk):
    # SR1 updater
    s_Hy = sk - np.dot(Hk, yk)
    upper = np.matmul(s_Hy.reshape(-1, 1), s_Hy.reshape(1, -1))
    lower = np.dot(s_Hy, yk)
    assert(np.isscalar(lower) and upper.shape == Hk.shape)
    return Hk + (upper / lower).numpy()


def dfp_update_func(Hk, sk, yk):
    # DFP updater
    hy = np.dot(Hk, yk)
    upper1 = np.matmul(hy.reshape(-1, 1), hy.reshape(1, -1))
    lower1 = np.dot(yk, hy)
    upper2 = np.matmul(sk.reshape(-1, 1), sk.reshape(1, -1))
    lower2 = np.dot(yk, sk)
    assert(np.isscalar(lower1) and np.isscalar(lower2) and upper1.shape == Hk.shape and upper2.shape == Hk.shape)
    return Hk - upper1 / lower1 + upper2 / lower2


def bfgs_update_func(Hk, sk, yk):
    # BFGS updater
    sy_matrix = np.matmul(sk.reshape(-1, 1), yk.reshape(1, -1))
    ys_matrix = np.matmul(yk.reshape(-1, 1), sk.reshape(1, -1))
    ys_dot = np.dot(sk, yk)
    ss_matrix = np.matmul(sk.reshape(-1, 1), sk.reshape(1, -1))
    assert(np.isscalar(ys_dot) and ss_matrix.shape == Hk.shape and sy_matrix.shape == Hk.shape and ys_matrix.shape == Hk.shape)
    I = np.eye(len(Hk))
    mat1 = I - sy_matrix / ys_dot
    mat2 = I - ys_matrix / ys_dot
    return np.matmul(np.matmul(mat1, Hk), mat2) + ss_matrix / ys_dot
