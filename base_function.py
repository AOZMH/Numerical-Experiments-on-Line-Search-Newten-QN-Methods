# Base function class, define the naive implementation of its gradient & hessian function
import torch


class base_func:

    def __init__(self):
        self.reset()
    
    def reset(self, func_calls=0, g_calls=0, G_calls=0):
        self.func_calls = func_calls
        self.g_calls = g_calls
        self.G_calls = G_calls
    
    def get_eval_infos(self):
        return 'feval = {}\tgeval = {}\tGeval = {}'.format(self.func_calls, self.g_calls, self.G_calls)

    def g_func(self, x):
        # Gradient of as function
        # Input: x, an n-dim float vector
        # Output: g(x), an n-dim float vector
        self.g_calls += 1

        tmp_x = x.clone().detach().requires_grad_()
        #tmp_x.grad.zero_()
        func_val = self(tmp_x)
        func_val.backward()
        return tmp_x.grad.clone().detach()

    def G_func(self, x):
        # Hessian of Griewank's function
        # Input: x, an n-dim float vector
        # Output: G(x), an n*n float matrix
        self.G_calls += 1

        return torch.autograd.functional.hessian(self.__call__, x)
    
    def get_partial_alpha(self, xk, dk):
        # Get 1-dim function of a for line search
        return lambda alpha : self(xk + dk * alpha)
    
