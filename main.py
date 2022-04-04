import numpy as np
import torch

from ackley_func import ackley_func
from newton import damp_newton_method, cholesky_newton_method
from quasi_newton import quasi_newton_method, sr1_update_func, dfp_update_func, bfgs_update_func


def newton_test(x0, optimizer):
    # test newton methods
    func = ackley_func()
    optimizer(func, func.g_ackley_func, func.G_ackley_func, x0, eps=1e-4)


def q_newton_test(x0, updater):
    # test newton methods
    func = ackley_func()
    H0 = np.eye(len(x0))
    quasi_newton_method(func, func.g_ackley_func, func.G_ackley_func, x0, H0, eps=1e-8, Hk_update_func=updater)


def main():
    x_scales = [8, 16, 32, 64, 128]
    for x_scale in x_scales:
        x0 = torch.randn(x_scale) * 32.768  # Ackley's function's range
        x0 = torch.randn(x_scale)
        #newton_test(x0, damp_newton_method)
        #newton_test(x0, cholesky_newton_method)
        q_newton_test(x0, sr1_update_func)


if __name__ == '__main__':
    main()
