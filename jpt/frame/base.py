from functools import partial
from .. import const
import time
import numpy as np
import pickle
import jax
from jax import grad, jit, value_and_grad
from jax import random, tree, numpy as jnp
#from jax.tree_util import register_pytree_node_class as pytree_cls

key = random.PRNGKey(0) # for random nums

clipsize = 1e5

dev = tuple(jax.devices())
mv = lambda arr, to: jax.device_put(arr, device=to)
ndevs = len(dev)

@jit
def div(arr, denom):
    return tree.map(lambda pm: pm/denom, arr)

@jit
def add(*arrs):
    return tree.map(lambda *pms: sum(pms), *arrs)

def join(arr, norm):
    for i in range(1,ndevs):
        arr[i] = mv(arr[i], dev[0])
    return div(add(*arr), norm)

def devsplit(devs, arr):
    return [
        mv(arr, dv)
        for dv in devs
    ]

@partial(jit, static_argnums=1)
def pred(params, model, i):
    for p,l in zip(params, model):
        i = l(i, *p)
        i = jnp.clip(i, -clipsize, clipsize)
    return i
    
@partial(jit, static_argnums=(1,2))
@value_and_grad
def loss(params, model, loss_fn, x, y):
    return loss_fn(pred(params, model, x), y)

@partial(jit, static_argnums=2, donate_argnums=(0,))
def upd(weights, grad, optim, oparams):
    return optim(weights, grad, *oparams)

class Model:
    def __init__(self, shape, stack, optim, lossfn, fp=None):
        self.calls = tuple(stack)
        self.lossfn = lossfn
        self.fp = fp
        self.shape = list(shape) # External (layer) use
        
        self.weights = () # Immutable to aid with jitting
        for call in stack:
            self.weights += (call.init(shape),)
        
        self.set_optim(optim)
    
    def set_optim(self, optim):
        self.optim = optim
        self.o_weights = optim.init(self.weights)
    
    def count_weights(self):
        return self._count_weights(self.weights)
    
    def _count_weights(self, pytree):
        if type(pytree)!=tuple:
            p = 1
            for d in pytree.shape:
                p *= d
            return p
        s = 0
        for w in pytree:
            s += self._count_weights(w)
        return s
    
    def pred(self, x): # External use
        return pred(self.weights, self.calls, x)
    
    def loss(self, x, y): # External use
        return self._loss(self.weights, x, y)
    
    def _loss(self, params, x, y):
        return loss(params, self.calls, self.lossfn, x, y)
    
    def train(self, train, val, epochs=80, per=100, accum=16):
        train = iter(train)
        val = iter(val)
        zeros = tree.map(lambda x: 0*x, self.weights)
        
        w_dev = devsplit(dev, self.weights)
        ow_dev = devsplit(dev, self.o_weights)
        grads = devsplit(dev, zeros)
        
        itr = 0
        for ep in range(epochs):
            t_epoch = time.time()
            print("\nepoch", ep+1, "of", epochs)
            
            losses = [0]*ndevs
            for pr in range(per):
                itr += 1
                ind = (itr%11)//7
                l,g = self._loss(w_dev[ind], *next(train))
                losses[ind] += l
                grads[ind] = add(grads[ind], g)
                if not itr%accum:
                    self.weights, self.o_weights = upd(
                        self.weights,
                        join(grads, accum),
                        self.optim,
                        self.o_weights
                    )
                    w_dev = devsplit(dev, self.weights)
                    ow_dev = devsplit(dev, self.o_weights)
                    grads = devsplit(dev, zeros)
            
            self.save_weights()
            l_mean = join(losses, per)
            lval = self._loss(self.weights, *next(val))[0]
            print(f"{l_mean:>.4}, {lval:>.4}")
            print('epoch time:', time.time()-t_epoch)
        self.save_weights()
    
    def save_weights(self):
        with open(self.fp, 'wb') as f:
            pickle.dump(self.weights, f)
    
    def load_weights(self):
        with open(self.fp, 'rb') as f:
            self.weights = pickle.load(f)

class Layer:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    
    def init(self, shape):
        self.shape = shape
        #self.params = len(wgts)
        return self.build(*self.args, **self.kwargs)

    
    def mw(self, *shape, initfn=random.normal):
        return initfn(key, shape)

class Optim:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    
    def init(self, weights):
        weights = tree.map(lambda w: 0, weights)
        return self.build(weights, *self.args, **self.kwargs)
    
    def cpy(self, weights):
        return tree.map(lambda x: jnp.copy(x), weights)

