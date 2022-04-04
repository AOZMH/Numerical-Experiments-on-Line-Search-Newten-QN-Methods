import time
import numpy as np
import torch

from ackley_func import ackley_func
from line_search import fib_searcher
from newton import damp_newton_method, cholesky_newton_method
from quasi_newton import quasi_newton_method, sr1_update_func, dfp_update_func, bfgs_update_func


def newton_test(x0, optimizer, line_searcher, trial_name):
    # test newton methods
    func = ackley_func()
    t0 = time.time()
    x_star, epochs = optimizer(func, func.g_ackley_func, func.G_ackley_func, x0, line_searcher, eps=1e-8)
    
    # output statistics
    elapsed_time = time.time() - t0
    eval_info = func.get_eval_infos()   # NOTE this must be called before the final evaluation of x*
    f_star = func(x_star)
    g_star = func.g_ackley_func(x_star)
    g_norm = np.linalg.norm(g_star)

    print('{}\tfx* = {:.4f}\t|gx*| = {:.8f}\ttime = {:.6f}\tepochs = {}\t{}'.format(trial_name, f_star, g_norm, elapsed_time, epochs, eval_info))


def q_newton_test(x0, updater, line_searcher, trial_name):
    # test newton methods
    func = ackley_func()
    H0 = np.eye(len(x0))
    t0 = time.time()
    x_star, epochs = quasi_newton_method(func, func.g_ackley_func, func.G_ackley_func, x0, H0, line_searcher, eps=1e-8, Hk_update_func=updater)

    # output statistics
    elapsed_time = time.time() - t0
    eval_info = func.get_eval_infos()   # NOTE this must be called before the final evaluation of x*
    f_star = func(x_star)
    g_star = func.g_ackley_func(x_star)
    g_norm = np.linalg.norm(g_star)

    print('{}\tfx* = {:.4f}\t|gx*| = {:.8f}\ttime = {:.6f}\tepochs = {}\t{}'.format(trial_name, f_star, g_norm, elapsed_time, epochs, eval_info))


def main():
    fib_search_inst = fib_searcher()
    x_scales = [8, 16, 32, 64, 128]
    #x_scales = [32]
    for x_scale in x_scales:
        print('================= Scale of X is {} ================'.format(x_scale))
        #x0 = torch.randn(x_scale) * 32.768  # Ackley's function's range
        x0 = torch.randn(x_scale) * 2
        newton_test(x0, damp_newton_method, fib_search_inst, 'Damp Newton')
        newton_test(x0, cholesky_newton_method, fib_search_inst, 'Cholesky')
        q_newton_test(x0, sr1_update_func, fib_search_inst, 'Quasi SR1')
        q_newton_test(x0, dfp_update_func, fib_search_inst, 'Quasi DFP')
        q_newton_test(x0, bfgs_update_func, fib_search_inst, 'Quasi BFGS')


if __name__ == '__main__':
    main()
