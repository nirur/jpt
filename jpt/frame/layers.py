from functools import partial
from .base import Layer
import numpy as np
import jax.numpy as jnp
from jax import nn, devices, jit

class Linear(Layer):
    def build(self, out):
        C = self.shape[-1]
        self.shape[-1] = out
        return (
            self.mw(C, out),
            self.mw(out),
        )
    
    def __call__(self, i, mat, bias):
        return (i @ mat) + bias

class Lambda(Layer):
    def build(self, fn):
        self.fn = fn
        return ()
    
    def __call__(self, i):
        return self.fn(i)

class RMSNorm(Layer):
    def build(self, eps=1e-7):
        self.eps = eps
        self.shape = tuple(self.shape)
        shp = self.shape[1:-1]
        return (
            self.mw(*shp, initfn=lambda _,sh: jnp.ones(sh)),
            self.mw(*shp, initfn=lambda _,sh: jnp.zeros(sh)),
        )
    
    def __call__(self, i, gamma, beta):
        #print(self.shape)
        ln = self.shape[-1]
        g = self.nax(gamma, ln)
        b = self.nax(beta, ln)
        
        mean = i.mean(axis=-1)
        mean = self.nax(mean, ln)
        i -= mean
        
        std = ((i**2).sum(axis=-1) + self.eps)**0.5
        std = self.nax(std, ln)
        i /= std
        
        return i * g + b
    
    def nax(self, mat, ln):
        mat = jnp.expand_dims(mat, -1)
        return jnp.repeat(mat, ln, axis=-1)

class PosEnc(Layer):
    def build(self, enc):
        _, T, C = self.shape
        self.shape[-1] = enc
        return (
            self.mw(C, enc),
            self.mw(T, enc),
        )
    
    def __call__(self, i, embed, pos):
        out = (i @ embed) + pos
        return out

class FFW(Layer):
    def build(self, hdm):
        C = self.shape[-1]
        return (
            self.mw(C, hdm),
            self.mw(hdm),
            self.mw(hdm, C),
            self.mw(C),
        )
    
    def __call__(self, i, m1, b1, m2, b2):
        i = (i @ m1) + b1
        i = nn.relu(i)
        i = (i @ m2) + b2
        return i

class MHAttn(Layer):
    def build(self, H):
        self.H = H
        B, T, C = self.shape
        self.T = T
        D = C//H
        
        inf = float('inf')
        self.mask = np.zeros((T,T))
        for i in range(T):
            self.mask[i, i+1:] = -inf
        
        return (
            self.mw(H, C, D) * 0.02,
            self.mw(H, C, D) * 0.02,
            self.mw(H, C, C) * 0.02,
            self.mw(C) * 0.02,
        )
    
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

class TransformerBlock(Layer):
    def build(self, H, hdm, eps=1e-7):
        self.mh = MHAttn(H)
        self.ln1 = RMSNorm(eps)
        self.ffw = FFW(hdm)
        self.ln2 = RMSNorm(eps)
        return (
            self.mh.init(self.shape),
            self.ln1.init(self.shape),
            self.ffw.init(self.shape),
            self.ln2.init(self.shape),
        )
    
    def __call__(self, i, mh, ln1, ffw, ln2):
        j = self.mh(i, *mh)
        j = self.ln1(j, *ln1)
        i += j
        j = self.ffw(i, *ffw)
        j = self.ln2(j, *ln2)
        i += j
        return i
