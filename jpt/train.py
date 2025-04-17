import os
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".99"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
from . import data
from .data import configs
from .defns.construct import model
import jax
#jax.config.update("jax_debug_nans", True)

ltrain, lval = data.fetch(configs.conf[0])

print('params:', model.count_weights())
model.train(
    ltrain, lval,
    epochs=50,
    per_epoch=100,
)
