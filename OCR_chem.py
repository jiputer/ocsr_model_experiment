#building my own neural net layers

#%%
import torch
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import matplotlib.pyplot as plt


# %%
# parsing the input with the labels
# make other samples of the input to train with (how should we do this)

def parse():
    pass