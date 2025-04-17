from functools import partial
import time
import numpy as np
import jax
from jax import grad, jit, value_and_grad
from jax import random, tree, numpy as jnp
#from jax.tree_util import register_pytree_node_class as pytree_cls

key = random.PRNGKey(0) # for random nums

clipsize = 1e5

dev = tuple(jax.devices())
mv = lambda arr, to: jax.device_put(arr, device=to)
ndevs = len(dev)

@partial(jit)
def combine(arr, norm):
    return tree.map(lambda *pms: sum(pms)/norm, *arr)

def join(arr, norm):
    for i in range(1,ndevs):
        arr[i] = mv(arr[i], dev[0])
    return combine(arr, norm)

def devsplit(devs, arr):
    return [
        mv(arr, dv)
        for dv in devs
    ]

@partial(jit, static_argnums=1)
def pred(params, model, i):
    ind = 0
    for l in model:
        i = l(i, *params[ind:ind+l.params])
        i = jnp.clip(i, -clipsize, clipsize)
        ind += l.params
    return i
    
@partial(jit, static_argnums=(1,2))
@value_and_grad
def loss(params, model, loss_fn, x, y):
    return loss_fn(pred(params, model, x), y)

@partial(jit, static_argnums=2, donate_argnums=(0,3))
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
            self.weights += call.init(shape)
        
        self.set_optim(optim)
    
    def set_optim(self, optim):
        self.optim = optim
        self.o_weights = optim.init(self.weights)
    
    def count_weights(self):
        s = 0
        for w in self.weights:
            p = 1
            for d in w.shape:
                p *= d
            s += p
        return s
    
    def pred(self, x): # External use
        return pred(self.weights, self.calls, x)
    
    def loss(self, x, y): # External use
        return self._loss(self.weights, x, y)
    
    def _loss(self, params, x, y):
        return loss(params, self.calls, self.lossfn, x, y)
    
    def train(self, train, val, epochs=80, per_epoch=100):
        train = iter(train)
        val = iter(val)
        itr = 0
        for ep in range(epochs):
            t_epoch = time.time()
            print("\nepoch", ep+1, "of", epochs)
            w_dev = devsplit(dev, self.weights)
            ow_dev = devsplit(dev, self.o_weights)
            losses = [0]*ndevs
            for per in range(per_epoch):
                itr += 1
                ind = (itr%7)//4
                l,g = self._loss(w_dev[ind], *next(train))
                w_dev[ind], ow_dev[ind] = upd(w_dev[ind], g, self.optim, ow_dev[ind])
                losses[ind] += l
                #time.sleep(0.08)
            self.weights = join(w_dev, ndevs)
            self.o_weights = join(ow_dev, ndevs)
            l_mean = join(losses, per_epoch)
            lval = self._loss(self.weights, *next(val))[0]
            self.save_weights()
            print(f"{l_mean:>.4}, {lval:>.4}")
            print('epoch time:', time.time()-t_epoch)
        self.save_weights()
    
    def save_weights(self):
        jnp.savez(self.fp, *self.weights)
    
    def load_weights(self):
        self.weights = tuple(jnp.load(self.fp, allow_pickle=True).values())

class Layer:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    
    def init(self, shape):
        self.shape = shape
        wgts = self.build(*self.args, **self.kwargs)
        self.params = len(wgts)
        return wgts
    
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

