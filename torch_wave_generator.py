import scipy.stats as st
# import numpy as np
import torch
from math import pi
from copy import deepcopy
import itertools

def add_noise(arr, distribution='gaussian', symmetric=True, noise_amplitude=1, method='additive', bias=0):
    noise_len = len(arr)
    match distribution:
        case 'gaussian':
            dist = st.norm(0,.5)
        case 'beta':
            dist = st.beta(0,1)
        case 'levy':
            dist = st.levy(0,1)
        case 'uniform':
            dist = st.uniform(0,1)
        case 'pareto':
            dist = st.power(0,1)
        case 't':
            dist = st.t(4,0,1)
    noise_tensor = torch.from_numpy(dist.rvs(size=noise_len))
    if bias:
        noise_tensor = torch.add(noise_tensor, bias)
    if symmetric and distribution in ['beta','pareto']:
        random_indices = torch.randint(0,1).bool()
        noise_tensor[random_indices] = -noise_tensor[random_indices]
    if method == 'additive':
        arr = torch.add(arr, noise_tensor)
    elif method == 'multiplicative':
        arr = torch.mul(arr, noise_tensor)
    else:
        raise ValueError(f'{method} not recognized as noise method')
    return arr

class RecursiveWaveGen():
    def __init__(self, size=100):
        ## method in ('recursive', '')
        self.size = size
        self.operation_dict = {
            '_n_periods' : 5,
        }
        self.key_dict = {
            '_n_periods' : 0,
            '_phase_angle' : 5,
            '_sin' : 8,
            '_cos' : 8,
            '_amp' : 12,
            '_bias' : 13,
            '_repeat' : 20,
            '_gaussian_noise' : 30,
        }
        return
    
    def _phase_angle(self, x, phase_angle):
        return x + phase_angle

    def phase_angle(self, phase_angle):
        self.operation_dict['_phase_angle'] = phase_angle,
        return

    def _n_periods(self, _, n_periods):
        return torch.linspace(0, n_periods * 2 * pi, size=self.size)

    def n_periods(self, n_periods):
        self.operation_dict['_phase_angle'] = n_periods,
        return self
    
    def _bias(self, x, bias):
        return x + bias

    def bias(self, bias):
        self.operation_dict['_bias'] = bias, 
        return self

    def opt_sample(self, xs, op_dict):
        assert '_n_periods' in self.operation_dict
        total_len = torch.prod([len(op) for op in op_dict.values() if isinstance(op, list)])
        x_list = list()
        n_periodss = op_dict['n_periods']
        for n_periods in n_periodss:
            x = torch.linspace(0, n_periods * 2 * pi, size=self.size).unsqueeze(1)
            x = torch.repeat(x, total_len // len(n_periodss), dim=1)
            x_list.append(x)
        xs = torch.cat(x_list, dim=1)
        assert xs.shape[1] == total_len
        assert xs.shape[0] == self.size
        del op_dict['n_periods']
        total_partitions = 1
        for func_name, args in sorted(op_dict.items(), key=lambda x : self.key_dict[x]):
            if isinstance(args, list):
                n_args = len(args)
                for i in range(total_partitions):
                    for j, arg in enumerate(args):
                        # this needs to be fixed for proper combinatorics
                        start = i * xs.shape[0] // total_partitions
                        xs[:,start::n_args + j] = getattr(self, func_name)(xs[:,start::n_args + j], *arg)
                total_partitions *= n_args
            else:
                xs = getattr(self, func_name)(xs, *arg)
        return xs

    def recursive_sample(self, xs, op_dict):
        for func_name, args in sorted(op_dict.items(), key=lambda x : self.key_dict[x]):
            if isinstance(args, (list)):
                xs = [getattr(self, func_name)(x, *arg) for x, arg in itertools.product(xs, args)]
            else:
                xs = [getattr(self, func_name)(x, *args) for x in xs]
        return xs
    
    def sample(self, n_samples=1):
        assert '_n_periods' in self.operation_dict
        for func_name, args in sorted(self.operation_dict.items(), key=lambda x : self.key_dict[x]):
            assert isinstance(args, tuple), type(args)
            x = getattr(self, func_name)(x, *args)
        return x
    
    def _repeat(self, x, n_samples):
        x = x.repeat(n_samples, 1, 1)
        assert x.shape[1] == n_samples, x.shape
        return x
    
    def repeat(self, n_samples):
        self.operation_dict['_repeat'] = n_samples,
        return self

    def _cos(self, x, _):
        return torch.cos(x)

    def cos(self):
        self.operation_dict['_cos'] = None,
        return self
    
    def _sin(self, x, _):
        return torch.sin(x)

    def sin(self):
        self.operation_dict['_sin'] = None,
        return self

    def _amp(self, x, amp):
        return x * amp

    def amp(self, amp):
        self.operation_dict['_amp'] = amp,
        return self
    
    def _fliph(self, x, p):
        indices = torch.randint(low=0, high=x.shape[0], size=int(x.shape[0] * p))
        x[indices,:] = x[indices,:][:,::-1]
        return x

    def fliph(self, p):
        assert 0 < p < 1
        self.operation_dict['_fliph'] = p,
        return self

    def _flipv(self, x, p):
        indices = torch.randint(low=0, high=x.shape[0], size=int(x.shape[0] * p))
        x[indices,:] = x[indices,:][:,::-1]
        return self
    
    def flipv(self, p):
        assert 0 < p < 1
        self.operation_dict['_flipv'] = p,
        return self
    
    def _gaussian_noise(self, x, mean, std):
        xn = torch.normal(mean=mean, std=std, size=x.size())
        return x + xn
    
    def gaussian_noise(self, mean, std):
        self.operation_dict['_gaussian_noise'] = (mean, std)
        return self