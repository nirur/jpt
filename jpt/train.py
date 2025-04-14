from . import data
from .defns.construct import model
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.99'

ltrain, lval = data.fetch(data.configs[0])

model.train(
    ltrain, lval,
    epochs=50,
    per_epoch=100,
)
