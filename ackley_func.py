# Ackley's function, with its gradient & hessian function
import numpy as np
import torch

from base_function import base_func


class ackley_func(base_func):

    def __init__(self):
        super(ackley_func, self).__init__()
    
    def __call__(self, x):
        # Ackley's function
        # Input: x, an n-dim float vector
        # Output: function value
        self.func_calls += 1

        v1 = -20 * torch.exp(-0.2 * (torch.mean(x ** 2) ** 0.5))
        v2 = torch.exp(torch.mean(torch.cos(2 * np.pi * x)))
        return v1 - v2 + 20 + np.e
    
    def g_func(self, x):
        # Gradient of ackley's function
        # Input: x, an n-dim float vector
        # Output: g(x), an n-dim float vector
        self.g_calls += 1

        mean_sqrt = torch.mean(x ** 2) ** 0.5
        v1 = torch.exp(-0.2 * mean_sqrt) * 4 / (len(x) * mean_sqrt) * x
        mean_cos = torch.exp(torch.mean(torch.cos(2 * np.pi * x)))
        v2 = mean_cos * 2 * np.pi / len(x) * torch.sin(2 * np.pi * x)
        return v1 + v2


def ackley_test(x):
    a,b,c = 20.0, 0.2, 2*np.pi 
    f  = -a*np.exp(-b*np.sqrt(np.mean(x**2)))
    f -= np.exp(np.mean(np.cos(c*x)))
    f += a + np.exp(1)
    return f


def main():
    ackley_func_inst = ackley_func()
    for ix in range(20):
        a = torch.randn(32, requires_grad=True)
        res = ackley_func_inst(a)
        
        delta = ackley_test(a.detach().numpy()) - res
        assert(abs(delta) < 2e-6), delta
        
        res.backward()
        delta_g = torch.norm(a.grad - ackley_func_inst.g_func(a))       
        assert(abs(delta) < 2e-6), delta


if __name__ == '__main__':
    main()
