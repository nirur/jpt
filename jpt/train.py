import os
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".99"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
from . import data
from .data import configs, tokens
from .model import gen_model
import jax
#jax.config.update("jax_debug_nans", True)

ltrain, lval = data.fetch(configs.conf[0])
#ltrain, lval = data.fetch_shakespeare()

mdl = gen_model(tokens.enc.span)
print('params:', mdl.count_weights())
mdl.train(
    ltrain, lval,
    epochs=50,
    per=500,
    accum=8,
)
mdl.train(
    ltrain, lval,
    epochs=50,
    per=500,
    accum=16,
)
