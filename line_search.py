

def back_forth(partial_func):
    # Back-and-forth method for initial range
    # partial_func := func(alpha) = f_raw(alpha * dk)
    alpha, step_size, mag_scale = 2, 1, 2
    if partial_func(alpha) > partial_func(alpha + step_size):
        while partial_func(alpha) > partial_func(alpha + step_size):
            alpha += step_size
            step_size *= mag_scale
            #print(step_size, alpha)
        return [alpha - step_size / mag_scale, alpha + step_size]
    elif partial_func(alpha) > partial_func(alpha - step_size):
        while partial_func(alpha) > partial_func(alpha - step_size):
            alpha -= step_size
            step_size *= mag_scale
        return [alpha - step_size, alpha + step_size / mag_scale]
    else:
        return [-10, 10]


def back_forth_test():
    from ackley_func import ackley_func
    import torch
    func = ackley_func()
    xk = torch.randn(128) / 10
    dk = -func.g_ackley_func(xk)
    partial_func = func.get_partial_alpha(xk, dk)
    #print(xk)
    res = back_forth(partial_func)
    print(res)


class fib_searcher:
    # Fibbonacci search for line search
    def __init__(self, max_fib_len=100):
        self.fib_seq = self.get_fib_seq(max_fib_len)

    def get_fib_seq(self, len_seq):
        # Generate fibonacci sequence
        cur_list = [1, 1]
        for ix in range(len_seq - 2):
            cur_list.append(cur_list[-2] + cur_list[-1])
        return cur_list
    
    def fib_search(self, partial_func, init_l, init_r, n_func_calls, verbose=False):
        # Fibonacci search on initial range [init_l, init_r]
        assert(n_func_calls <= len(self.fib_seq))
        a, b = init_l, init_r
        state = 'start'
        for ix in range(n_func_calls - 2):
            x1 = a + self.fib_seq[n_func_calls - ix - 3] / self.fib_seq[n_func_calls - ix - 1] * (b - a)
            x2 = a + self.fib_seq[n_func_calls - ix - 2] / self.fib_seq[n_func_calls - ix - 1] * (b - a)
            if state == 'right':
                assert(abs(x1 - prev_x2) < 1e-4)
                f_x1, f_x2 = f_x2, partial_func(x2)
            elif state == 'left':
                assert(abs(x2 - prev_x1) < 1e-4)
                f_x1, f_x2 = partial_func(x1), f_x1
            else:
                f_x1, f_x2 = partial_func(x1), partial_func(x2)
            if verbose:
                print('[{}] a={:.5f}, x1={:.5f}, x2={:.5f}, b={:.5f}, f(x1)={:.5f}, f(x2)={:.5f}'.format(ix+1, a, x1, x2, b, f_x1, f_x2))
            
            prev_x1, prev_x2 = x1, x2
            if f_x1 >= f_x2:  # if f(x1) >= f(x2), search in (x1,b)
                a = x1
                x1 = x2
                state = 'right'
                #x2 = a + self.fib_seq[n_func_calls - ix - 3] / self.fib_seq[n_func_calls - ix - 2] * (b - a)
            else:  # if f(x1) < f(x2), search in (a,x2)
                b = x2
                x2 = x1
                state = 'left'
                #x1 = a + self.fib_seq[n_func_calls - ix - 4] / self.fib_seq[n_func_calls - ix - 2] * (b - a)
        
        return (a + b) / 2
    
    def fib_pipeline(self, partial_func, n_func_calls):
        # Full pipeline
        init_l, init_r = back_forth(partial_func)
        search_res = self.fib_search(partial_func, init_l, init_r, n_func_calls)
        return search_res


def fib_test():
    from ackley_func import ackley_func
    import torch
    func = ackley_func()
    xk = torch.randn(128) / 2
    dk = -func.g_ackley_func(xk)
    #dk = -xk
    partial_func = func.get_partial_alpha(xk, dk)
    fib_search_inst = fib_searcher()
    #print(xk)
    init_l, init_r = back_forth(partial_func)
    #init_l, init_r = -100, 100
    search_res = fib_search_inst.fib_search(partial_func, init_l, init_r, 50, verbose=True)
    print(search_res)


def gd_test():
    from ackley_func import ackley_func
    import torch
    func = ackley_func()
    xk = torch.randn(128) / 2
    print('[0] {:.6f}'.format(func(xk)))
    fib_search_inst = fib_searcher()
    
    for ix in range(100):
        dk = -func.g_ackley_func(xk)
        partial_func = func.get_partial_alpha(xk, dk)
        init_l, init_r = back_forth(partial_func)
        search_res = fib_search_inst.fib_search(partial_func, init_l, init_r, 50)
        xk = xk + search_res * dk
        print('[{}] {:.6f}'.format(ix, func(xk)))


if __name__ == '__main__':
    #back_forth_test()
    #fib_test()
    gd_test()
