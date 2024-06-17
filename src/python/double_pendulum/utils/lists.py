import numpy as np


def obj_to_list(obj):
    if type(obj) == np.ndarray:
        l = obj.tolist()
    else:
        l = list(obj)
    return l
