from .frame import Optim
from functools import partial
from jax import numpy as jnp
from jax import jit
from jax.tree import map as tmp

#mp = jit(tmp, static_argnums=0, donate_argnums=1)

class SGD(Optim):
    def build(self, zw, lr=1e-3, momentum=0.1, nesterov=False):
        self.lr = lr
        self.m = momentum
        self.n = nesterov
        return (
            self.cpy(zw), # velocity
        )
    
    def __call__(self, w, g, v):
        v = tmp(
            lambda v, g: self.m*v - self.lr*g,
            v, g,
        )
        if not self.n:
            w = tmp(
                lambda w, v: w+v,
                w, v,
            )
        else:
            w = tmp(
                lambda w, v, g: w + self.m*v - self.lr*g,
                w, v, g,
            )
        return w, (v,)

class AdaDelta(Optim):
    def build(self, zw, rho=0.95, eps=1e-5, lr=0.8):
        self.rho = rho
        self.eps = eps
        self.lr = lr
        return (
            self.cpy(zw), # Delta accumulation
            self.cpy(zw), # Grad accumulation
        )
    
    def __call__(self, w, g, sum_x, sum_g):
        sum_g = tmp(
            lambda sum_g, g: sum_g*self.rho + g**2*(1-self.rho),
            sum_g, g,
        )
        x = tmp(
            lambda g, sum_g, sum_x: g * ((sum_x+self.eps)/(sum_g+self.eps))**0.5,
            g, sum_g, sum_x,
        )
        w = tmp(
            lambda w,x: w-x*self.lr,
            w, x,
        )
        sum_x = tmp(
            lambda sum_x, x: sum_x*self.rho + x**2*(1-self.rho),
            sum_x, x,
        )
        print('optim built')
        return w, (sum_x,sum_g)

class Adam(Optim):
    def build(self, zw, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.lr = lr
        return (
            self.cpy(zw), # First moment
            self.cpy(zw), # Second moment
            0, # time
        )
    
    def __call__(self, w, g, sum_g, sum_g2, t):
        t += 1
        sum_g = tmp(
            lambda sum_g, g: (sum_g*self.b1 + g*(1-self.b1))/(1-self.b1**t),
            sum_g, g,
        )
        sum_g2 = tmp(
            lambda sum_g2, g: (sum_g2*self.b2 + g**2*(1-self.b2))/(1-self.b2**t),
            sum_g2, g,
        )
        w = tmp(
            lambda w, sum_g, sum_g2: w - self.lr*sum_g/(sum_g2**0.5+self.eps),
            w, sum_g, sum_g2
        )
        print('optim built')
        return w, (sum_g,sum_g2,t)

