# Zakharov's function, with its gradient & hessian function
import numpy as np
import torch

from base_function import base_func


class rastrigin_func(base_func):

    def __init__(self):
        super(rastrigin_func, self).__init__()
    
    def __call__(self, x):
        # Zakharov's function
        # Input: x, an n-dim float vector
        # Output: function value
        self.func_calls += 1

        v1 = torch.sum(x ** 2)
        v2 = 10 * torch.sum(torch.cos(2 * np.pi * x))
        return 10 * len(x) + v1 - v2


    
