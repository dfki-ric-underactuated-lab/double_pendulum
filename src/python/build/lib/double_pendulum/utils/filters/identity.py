import numpy as np


class identity_filter():
    def __init__(self):
        pass

    def __call__(self, x, u=None):
        return np.copy(x)
