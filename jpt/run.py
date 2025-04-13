from . import data
from .construct import model

ltrain, lval = data.fetch(data.configs[0])

model.train(
    ltrain, lval,
    epochs=50,
    per_epoch=100,
)
