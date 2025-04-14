from .frame import Model
from .layers import *
from .optims import SGD, AdaDelta
from .. import const
from jax import nn, numpy as np

def cce(x, y):
    ce = - (np.log(x) * y)
    return ce.sum(-1).mean()

enc_dim = 256
stack = [PosEnc(enc_dim)]
for i in range(8):
    stack.append(MHAttn(8))
    stack.append(FFW(2*enc_dim))
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
    optim=AdaDelta(
        rho=0.8,
        eps=1e-7,
    ),
    lossfn=cce,
    fp = 'saved/models/00.npz',
)
print('built')
