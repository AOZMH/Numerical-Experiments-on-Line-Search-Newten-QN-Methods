# Zakharov's function, with its gradient & hessian function
import numpy as np
import torch

from base_function import base_func


class easom_func(base_func):

    def __init__(self):
        super(easom_func, self).__init__()
    
    def __call__(self, x):
        # Easom's function
        # Input: x, an n-dim float vector
        # Output: function value
        self.func_calls += 1

        v1 = torch.prod(torch.cos(x) ** 2)
        v2 = torch.exp(-torch.sum((x - np.pi) ** 2))
        return ((-1) ** (len(x) + 1)) * v1 * v2


    
