from .. import const
from . import tokens
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv
import datasets as ds
import pickle
from jax import jit, lax, numpy as jnp

val_prop = 10 # 1/val_prop = proportion of data that goes to validation

enc = tokens.enc

def fetch(cfg):
    dst = ds.load_dataset(**cfg)
    d = [dst.shard(val_prop, i) for i in range(val_prop)]
    return loader(*d[:-1]), loader(d[-1])

def loader(*dsts):
    while True:
        for d in dsts:
            for i in d:
                v = splice(enc.encode(i))
                for j in range(0, len(v)-const.batch, const.batch):
                    yield v[j:j+const.batch], v[j+1:j+const.batch+1]

def splice(t):
    v = swv(t, (const.rlens,), axis=0)
    v = v.swapaxes(1, 2)
    return v

configs = [
    {
        "path": "openwebtext",
        "streaming": True,
        "split": "train",
    },
    {
        "path": "roneneldan/TinyStories",
    },
    {
        "path": "HuggingFaceFW/fineweb",
        "name": "CC-MAIN-2024-10",
        "split": "train",
        "streaming": True,
    },
    {
        "path": "HuggingFaceFW/fineweb",
        "name": "sample-10BT",
        "split": "train",
        "streaming": True,
    },
]
