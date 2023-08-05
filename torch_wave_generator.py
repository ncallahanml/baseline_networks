import scipy.stats as st
# import numpy as np
import torch
from math import pi
from copy import deepcopy

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
        }
        return
    
    def _phase_angle(self, x, phase_angle):
        return x + phase_angle

    def phase_angle(self, phase_angle=0):
        self.operation_dict['_phase_angle'] = (phase_angle,)
        return

    def _n_periods(self, _, n_periods):
        return torch.linspace(0, n_periods * 2 * pi, size=self.size)

    def n_periods(self, n_periods=5):
        self.operation_dict['_phase_angle'] = (n_periods,)
        return self
    
    def _bias(self, x, bias):
        return x + bias

    def bias(self, bias=0):
        self.operation_dict['_bias'] = (bias, )
        return self

    def recursive_sample(self, x, op_dict):
        raise NotImplemented
        for func_name, args in sorted(self.operation_dict.items(), key=self.key_func):
            xs = [getattr(self, func_name)(x, *arg) for arg in args]
        return x
    
    def sample(self):
        assert '_n_periods' in self.operation_dict
        for func_name, args in sorted(self.operation_dict.items(), key=self.key_func):
            x = getattr(self, func_name)(x, *args)
        return x

    def _cos(self, x, _):
        return torch.cos(x)

    def cos(self):
        self.operation_dict['_cos'] = None
        return self
    
    def _sin(self, x, _):
        return torch.sin(x)

    def sin(self):
        self.operation_dict['_sin'] = None
        return self

    def _amp(self, x, amp):
        return x * amp

    def amp(self, amp=1):
        self.operation_dict['_amp'] = amp
        return self
    
    def _fliph(self, x, p):
        indices = torch.randint(low=0, high=x.shape[0], size=int(x.shape[0] * p))
        x[indices,:] = x[indices,:][:,::-1]
        return x

    def fliph(self, p=0):
        assert 0 < p < 1
        self.operation_dict['_fliph'] = p
        return self

    def _flipv(self, x, p):
        indices = torch.randint(low=0, high=x.shape[0], size=int(x.shape[0] * p))
        x[indices,:] = x[indices,:][:,::-1]
        return self
    
    def flipv(self, p=0):
        assert 0 < p < 1
        self.operation_dict['_flipv'] = p
        return self