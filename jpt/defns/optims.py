from .frame import Optim
from functools import partial
from jax import numpy as jnp
from jax import tree

class SGD(Optim):
    def build(self, zw, lr=1e-3, momentum=0.1, nesterov=False):
        self.lr = lr
        self.m = momentum
        self.n = nesterov
        return (
            self.cpy(zw), # velocity
        )
    
    def __call__(self, w, g, v):
        v = tree.map(
            lambda v, g: self.m*v - self.lr*g,
            v, g,
        )
        if not self.n:
            w = tree.map(
                lambda w, v: w+v,
                w, v,
            )
        else:
            w = tree.map(
                lambda w, v, g: w + self.m*v - self.lr*g,
                w, v, g,
            )
        return w, (v,)

class AdaDelta(Optim):
    def build(self, zw, rho=0.95, eps=1e-7):
        self.rho = rho
        self.eps = eps
        return (
            self.cpy(zw), # Delta accumulation
            self.cpy(zw), # Grad accumulation
        )
    
    def __call__(self, w, g, dx, dg):
        dg = tree.map(
            lambda dg, g: dg*self.rho + g**2*(1-self.rho),
            dg, g,
        )
        x = tree.map(
            lambda g, dg, dx: g * (dx+self.eps)**0.5 * (dg+self.eps)**-0.5,
            g, dg, dx,
        )
        w = tree.map(
            lambda w,x: w-x,
            w, x,
        )
        dx = tree.map(
            lambda dx, x: dx*self.rho + x**2*(1-self.rho),
            dx, x,
        )
        return w, (dx,dg)

