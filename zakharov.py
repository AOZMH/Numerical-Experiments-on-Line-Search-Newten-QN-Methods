# Zakharov's function, with its gradient & hessian function
import numpy as np
import torch

from base_function import base_func


class zakharov_func(base_func):

    def __init__(self):
        super(zakharov_func, self).__init__()
    
    def __call__(self, x):
        # Zakharov's function
        # Input: x, an n-dim float vector
        # Output: function value
        self.func_calls += 1

        v1 = torch.sum(x ** 2)
        v2 = torch.sum(torch.arange(1, len(x) + 1) * x) / 2
        return v1 + (v2 ** 2) + (v2 ** 4)

    
