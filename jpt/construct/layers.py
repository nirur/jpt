from functools import partial
from .frame import Layer
import numpy as np
import jax.numpy as jnp
from jax import nn, devices, jit

#inst_jit_0 = partial(jit, static_argnums=0, device=devices()[0])
#inst_jit_1 = partial(jit, static_argnums=0, device=devices()[1])

class Linear(Layer):
    def build(self, out):
        C = self.shape[-1]
        self.add(
            self.mw(C, out),
            self.mw(out),
        )
        self.shape[-1] = out
    
    def __call__(self, i, mat, bias):
        return (i @ mat) + bias

class Lambda(Layer):
    def build(self, fn):
        self.fn = fn
    
    def __call__(self, i):
        return self.fn(i)

class PosEnc(Layer):
    def build(self, enc):
        _, T, C = self.shape
        self.add(
            self.mw(C, enc),
            self.mw(T, enc),
        )
        self.shape[-1] = enc
    
    def __call__(self, i, embed, pos):
        out = (i @ embed) + pos
        return out

class FFW(Layer):
    def build(self, hdm):
        C = self.shape[-1]
        self.add(
            self.mw(C, hdm),
            self.mw(hdm),
            self.mw(hdm, C),
            self.mw(C),
        )
    
    #@inst_jit_0
    def __call__(self, i, m1, b1, m2, b2):
        i = (i @ m1) + b1
        i = nn.leaky_relu(i)
        i = (i @ m2) + b2
        return i

class MHAttn(Layer):
    def build(self, H):
        self.H = H
        B, T, C = self.shape
        self.T = T
        D = C//H
        self.add(
            self.mw(H, C, D),
            self.mw(H, C, D),
            self.mw(H, C, C),
            self.mw(C),
        )
        
        inf = float('inf')
        self.mask = np.zeros((T,T))
        for i in range(T):
            self.mask[i, i+1:] = -inf
    
    #@inst_jit_1
    def __call__(self, i, k, q, v, b):
        j = i
        i = jnp.expand_dims(i, 1)
        i = i.repeat(self.H, axis=1)
        keys = (i @ k).mT
        queries = i @ q
        act = nn.softmax((queries @ keys)/self.T**0.5 + self.mask)
        vls = i @ v
        deltas = (act @ vls).sum(axis=1)
        return j + deltas + b
