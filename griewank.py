# Zakharov's function, with its gradient & hessian function
import numpy as np
import torch

from base_function import base_func


class griewank_func(base_func):

    def __init__(self):
        super(griewank_func, self).__init__()
    
    def __call__(self, x):
        # Griewank's function
        # Input: x, an n-dim float vector
        # Output: function value
        self.func_calls += 1

        sub = torch.arange(1, len(x) + 1) ** 0.5
        v1 = torch.sum(x ** 2)
        v2 = torch.prod(torch.cos(x / sub))
        return v1 / 4000 - v2 + 1

