from .frame import Model
from .layers import *
from .optims import *
from .. import const
from jax import nn, numpy as jnp, jit

@jit
def cce(x, y):
    return - (y * jnp.log(x)).sum(-1).mean()

enc_dim = 192
stack = [PosEnc(enc_dim)]
for i in range(12):
    stack += [
        MHAttn(16),
        LayerNorm(),
        FFW(4*enc_dim),
        LayerNorm(),
    ]
stack += [
    Linear(const.span),
    Lambda(nn.softmax),
]

model = Model(
    shape=[
        const.batch,
        const.rlens,
        const.span,
    ],
    stack=stack,
    optim=
    #SGD(
    #    lr=3e-3,
    #    momentum=0.2,
    #    nesterov=True,
    #),
    AdaDelta(
        rho=0.95,
        eps=1e-7,
        lr=1.0,
    ),
    #Adam(lr=1e-3),
    lossfn=cce,
    fp = 'saved/models/00.npz',
)
print('built')
