from functools import partial
import time
import numpy as np
import jax
from jax import grad, jit, value_and_grad
from jax import random, tree, numpy as jnp
#from jax.tree_util import register_pytree_node_class as pytree_cls

key = random.PRNGKey(0) # for random nums

dev = jax.devices()
#dev = dev[:1]

@partial(jit, static_argnums=1)
def _pred(params, model, i):
    ind = 0
    for l in model:
        i = l(i, *params[ind:ind+l.params])
        i = jnp.clip(i, -1e6, 1e6)
        ind += l.params
    return i
    
@partial(jit, static_argnums=(1,2))
@value_and_grad
def _loss(params, model, loss_fn, x, y):
    return loss_fn(_pred(params, model, x), y)

@jit
def _upd(weights, grad, lr):
    return tree.map(lambda w,g: w-lr*g, weights, grad)

class Model:
    def __init__(self, shape, lossfn, optim, fp=None):
        self.calls = ()
        # Since the weights are immutable, things can be jitted
        self.weights = ()
        self.shape = list(shape) # This is only used externally
        self.lr = lr
        self.lossfn = lossfn
        self.fp = fp
    
    def pred(self, x):
        return _pred(self.weights, self.calls, x)
    
    def loss(self, x, y):
        return self.loss_(self.weights, x, y)
    
    def loss_(self, params, x, y):
        return _loss(params, self.calls, self.lossfn, x, y)
    
    def train(self, train, val, epochs=80, per_epoch=100, save_freq=50):
        train = iter(train)
        val = iter(val)
        ndevs = len(dev)
        itr = 0
        for ep in range(epochs):
            t_epoch = time.time()
            print("\nepoch", ep+1, "of", epochs)
            
            w_devsplit = [
                jax.device_put(self.weights, dv)
                for dv in dev
            ]
            losses = [0]*ndevs
            for per in range(per_epoch):
                x, y = next(train)
                itr += 1
                ind = (itr%7)//4
                l,g = self.loss_(w_devsplit[ind], x, y)
                w_devsplit[ind] = _upd(w_devsplit[ind], g, self.lr)
                losses[ind] += l
            
            for i in range(1, ndevs):
                losses[i] = jax.device_put(losses[i], dev[0])
                w_devsplit[i] = jax.device_put(w_devsplit[i], dev[0])
            self.weights = tree.map(lambda *pms: sum(pms)/ndevs, *w_devsplit)
            l_mean = sum(losses)/per_epoch
            
            xv, yv = next(val)
            lval = self.loss_(self.weights, xv, yv)[0]
            if not ep%save_freq:
                self.save_weights()
            print(f"{l_mean:>.4}, {lval:>.4}")
            print('epoch time:', time.time()-t_epoch)
        self.save_weights()
    
    def save_weights(self):
        jnp.savez(self.fp, *self.weights)
    
    def load_weights(self):
        self.weights = tuple(jnp.load(self.fp, allow_pickle=True).values())

class Layer:
    def __init__(self, model, *args, **kwargs):
        self.model = model
        self.shape = model.shape
        self.params = 0
        self.build(*args, **kwargs)
        self.model.calls += (self,)
    
    def mw(self, *shape):
        w = random.normal(key, shape) * 0.05
        return w
    
    def add(self, *wgts):
        self.model.weights += wgts
        self.params += len(wgts)

class Optimizer:
    def __init__(self, *args, **kwargs):
        self.init(*args, **kwargs)
        self.model.

    def __call__(self, wgts, grads):
