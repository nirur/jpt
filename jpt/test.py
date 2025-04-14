from . import data
from .defns.construct import model
import numpy as np
from jax import numpy as jnp

model.load_weights()
enc = data.enc
enc.translate = lambda x: x.encode('utf-8')

out = []
text = ' '*data.rlens
text = enc.encode(text)
shaped = data.splice(text)
for i in range(2000):
    add = model.pred(shaped)[:, -1:]
    out.append(list(add[0, 0]))
    shaped = np.concatenate((shaped[:, 1:], add), axis=1)
print('predicted')
print(enc.decode(out))
