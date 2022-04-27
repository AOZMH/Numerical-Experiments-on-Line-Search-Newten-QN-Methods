from operator import gt
import numpy as np
np.seterr(all='raise')


def conjugate_gradient_two(func, g_func, x0, beta_func, line_searcher, eps=1e-8, n_epochs=100, verbose=False):
    # Basic Conjugate Gradient method
    # xk+1 = xk + alpha * dk
    # where dk+1 = dk + beta_func(gk, gk-1) * dk
    # where beta_func denotes the function to get beta in CG, \in {FR, PRP}
    # x0: initial value of x
    # func, g_func: function and its gradient
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


def fr_func(gk, last_gk, eps=1e-6):
    # Fletcher-Reeves function for beta_k
    last_gk_norm2 = np.dot(last_gk, last_gk)
    if last_gk_norm2 < eps:
        return 0
    else:
        return np.dot(gk, gk) / last_gk_norm2


def prp_func(gk, last_gk, eps=1e-6):
    # Polak-Ribiere-Polyak function for beta_k
    last_gk_norm2 = np.dot(last_gk, last_gk)
    if last_gk_norm2 < eps:
        return 0
    else:
        return np.dot(gk, (gk - last_gk)) / last_gk_norm2


def powell_beta(gk, last_gk, last_dk, eps=1e-6):
    # The 2nd item for Powell's restart method
    mom = np.dot(last_dk, (gk - last_gk))
    if mom < eps:
        return 0
    else:
        return np.dot(gk, (gk - last_gk)) / mom


def powell_gamma(gk, d_t, g_t, g_t_plus_1, eps=1e-6):
    # The 3rd item for Powell's restart method
    mom = np.dot(d_t, (g_t_plus_1 - g_t))
    if mom < eps:
        return 0
    else:
        return np.dot(gk, (g_t_plus_1 - g_t)) / mom


def conjugate_gradient_powell_three(func, g_func, x0, line_searcher, eps=1e-8, n_epochs=100, verbose=False):
    # Powell's Restart Conjugate Gradient method
    # xk+1 = xk + alpha * dk
    # where dk+1 = dk + beta_k * dk + gamma_k * dt
    # x0: initial value of x
    # func, g_func: function and its gradient
    # eps: epsilon for the variation of function values & gradients to stop iteration
    
    xk, last_fk = x0, 1000000000
    g_t, g_t_plus_1, d_t, t = None, None, None, 0
    prev_fks = []

    for epoch in range(n_epochs):
        # calculate gk & fk
        gk = g_func(xk)
        fk = func(xk)
        if np.linalg.norm(fk - last_fk) < eps and np.linalg.norm(gk) < eps:
            break
        
        # Find dk
        if epoch == 0:  # k=0, also set t=k=0
            dk = -gk
            g_t = gk
            d_t = dk
        else:   # need to consider restarts
            # Restart condition 1: if g{k-1}*gk / ||gk||^2 > 0.2, update t = k - 1
            try:
                restart_cond1 = (abs(np.dot(last_gk, gk)) / np.dot(gk, gk) >= 0.2)
            except:
                restart_cond1 = True
            # Restart condition 2: if (k-t) >= n, update t = k - 1
            restart_cond2 = (epoch - t >= len(x0))
            if restart_cond1 or restart_cond2:
                g_t = last_gk
                d_t = last_dk
                t = epoch - 1
            
            # Calculate beta for the 2nd item (i.e. the PRP item)
            beta_k = powell_beta(gk, last_gk, last_dk)
            # Calculate gamma for the 3rd item
            if t + 1 == epoch:
                assert(np.linalg.norm(g_t - last_gk) < eps), np.linalg.norm(g_t - last_gk)
                assert(np.linalg.norm(d_t - last_dk) < eps), np.linalg.norm(d_t - last_dk)
                g_t_plus_1 = gk     # NOTE update gt+1 here to be compatible with k=1!
                gamma_k = 0
            else:
                gamma_k = powell_gamma(gk, d_t, g_t, g_t_plus_1)
            
            # Get new direction dk
            dk = -gk + beta_k * last_dk + gamma_k * d_t
            
            # When not sufficiently downhill, restart & redefine dk
            if t + 1 < epoch:
                gk_norm2 = np.dot(gk, gk)
                gkdk = np.dot(gk, dk)
                if gkdk > -0.8 * gk_norm2 or gkdk < -1.2 * gk_norm2:
                    # restart
                    g_t = last_gk
                    d_t = last_dk
                    g_t_plus_1 = gk
                    t = epoch - 1
                    # redefine dk
                    dk = -gk + beta_k * last_dk

        # Update last values
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

