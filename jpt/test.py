from . import const, data
from .model import gen_model
import numpy as np
from jax import numpy as jnp

mdl = gen_model(data.tokens.enc.span)
mdl.load_weights()
enc = data.enc

out = []
text = ' '*const.rlens
text = enc.encode(text)
#print(enc.decode(text))
shaped = data.splice(text)
for i in range(500):
    if i%100==0: print(i, 'of', 500)
    add = mdl.pred(shaped)[:, -1:]
    val = enc.sample(add[0])[0]
    snapped = np.zeros((add.shape[-1],))
    snapped[val] = 1
    add = np.reshape(
        snapped,
        add.shape,
    )
    out.append(add[0, 0])
    shaped = np.concatenate((shaped[:, 1:], add), axis=1)
print('predicted')
print(enc.decode(out))
