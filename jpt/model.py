from .frame import base
from .frame.base import Model
from .frame.layers import *
from .frame.optims import *
from . import const
from jax import nn, numpy as jnp, jit

@jit
def cce(x, y):
    return - (y * jnp.log(x)).sum(-1).mean()

def gen_model(in_shape):
    enc_dim = 72
    stack = [
        PosEnc(enc_dim),
    ] + [
        TransformerBlock(8, 4*enc_dim)
        for i in range(8)
    ] + [
        RMSNorm(),
        Linear(in_shape),
        Lambda(nn.softmax),
    ]
    
    mdl = Model(
        shape=[
            const.batch,
            const.rlens,
            in_shape,
        ],
        stack=stack,
        optim=
        #SGD(
        #    lr=3e-3,
        #    momentum=0.2,
        #    nesterov=True,
        #),
        #AdaDelta(
        #    rho=0.95,
        #    eps=1e-7,
        #    lr=1,
        #),
        Adam(lr=3e-3),
        lossfn=cce,
        fp='saved/models/00.npz',
    )
    print('built')
    return mdl
