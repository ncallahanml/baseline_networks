import scipy.stats as st
import numpy as np
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
    noise_arr = dist.rvs(size=noise_len)
    if bias:
        noise_arr = np.add(noise_arr, bias)
    if symmetric and distribution in ['beta','pareto']:
        random_indices = np.random.randint(0,1).astype(bool)
        noise_arr[random_indices] = -noise_arr[random_indices]
    if method == 'additive':
        arr = np.add(arr, noise_arr)
    elif method == 'multiplicative':
        arr = np.multiply(arr, noise_arr)
    else:
        raise ValueError(f'{method} not recognized as noise method')
    return arr

def return_copy(func):
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        return deepcopy(output)
    return wrapper

def recursive_copy(func):
    def wrapper(*args, **kwargs):
        instances = list()
        filt_iter = lambda x : isinstance(x[1], (list, np.ndarray))
        prev = None
        for key, vals in filter(filt_iter, kwargs.items()):
            for val in vals:
                func_kwargs = deepcopy(kwargs) # dictionary method for this?
                func_kwargs[key] = val
                output = func(*args, prev=prev, **func_kwargs)
                prev = deepcopy(output)
                instances.append(prev)
        return instances
    return wrapper

class WaveGen():
    def __init__(self, size=100, sample_level='total', prev=None):
        self.size = size
        self.sample_level = sample_level
        self.prev = prev
        
        self.x = None
        self.wave = None
        
        self._bias = 0
        self._amp = 1
        
        self.indices = None
        return
    
    def linear_phase(self, phase_angle=0, n_periods=5):
        self.x = np.linspace(phase_angle, n_periods * np.pi * 2 + phase_angle, self.size)
        return self
    
    def geometric_phase(self, phase_angle=0, n_periods=0):
        # Test if this is ever useful
        self.x = np.geomspace(phase_angle, n_periods * np.pi * 2 + phase_angle, self.size)
        return self
   
    def cos(self):
        if self.x is None:
            self = self.linear_phase()
        self.wave = np.cos(self.x)
        return self
    
    def _ensure_sin(self):
        if self.x is None:
            self = self.linear_phase()
        if self.wave is None:
            self = self.sin()
        return self
    
    def sin(self):
        # won't be meaningful until adjustments to x are made
        if self.x is None:
            self = self.linear_phase()
        self.wave = np.sin(self.x)
        return self
        
    def bias(self, bias=0):
        self._bias = bias
        return self
    
    def amp(self, amp=1):
        self._amp = amp
        return self
    
    def _assemble(self):
        self = self._ensure_sin()
        self.wave *= self._amp
        self.wave += self._bias
        return self
    
    def pad(self, left_pad=0, right_pad=0):    
        self.left_pad = left_pad
        self.right_pad = right_pad
        return self
    
    def noise_patch(self, start=0, stop=-1, **noise_kwargs):
        raise NotImplementedError
        return self
    
    def gaussian_noise(self, indices=None, loc=0, std=1):
        if self.indices is None:
            self.indices = np.linspace(0, self.size-1, self.size).astype(np.int32) if indices is None else indices
        elif indices is not None:
            print('Overriding indices')
            self.indices = indices
        self.noise = lambda n_samples : np.random.normal(loc, std, size=(n_samples, len(indices)))
        return self
    
    def t_noise(self, indices=None, loc=0, std=1, dof=1):
        if self.indices is None:
            self.indices = np.linspace(0, self.size-1, self.size).astype(np.int32) if indices is None else indices
        elif indices is not None:
            print('Overriding indices')
            self.indices = indices
        self.noise = lambda n_samples : np.random.standard_t(dof, size=(n_samples, len(self.indices))) * std + loc
        return self
    
    def sample(self, n_samples=1000):
        self = self._assemble()
        if self.indices is None: 
            print('Entered')
            self.indices = np.linspace(0, self.size-1, self.size).astype(np.int32)
        wave = np.expand_dims(self.wave, axis=0) # broadcast
        background = np.zeros((n_samples, wave.shape[1]), dtype=np.float32)
        background[:,self.indices] = self.noise(n_samples) 
        samples = wave + background
        return samples