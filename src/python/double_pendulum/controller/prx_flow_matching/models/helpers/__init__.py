from .losses import *
from .nn_helpers import *

def apply_conditioning(x, conditions):
    for t, val in conditions.items():
        if x.ndim == 2:
            x[t] = val.clone()
        else:
            x[:, t] = val.clone()
    return x