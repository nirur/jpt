from .frame import Model
from .layers import *
from .. import data
from jax import nn, numpy as np

def cce(x, y):
    ce = - (np.log(x) * y)
    return ce.sum(-1).mean()

enc_dim = 128
model = Model(
    [
        data.batch,
        data.rlens,
        data.tokens.span,
    ],
    lossfn=cce,
    lr=1e-3,
    fp = 'saved/models/00.npz',
)
PosEnc(model, enc_dim)
for i in range(8):
    MHAttn(model, 8)
    FFW(model, 4*enc_dim)
Linear(model, data.tokens.span)
Lambda(model, nn.softmax)
print('built')
