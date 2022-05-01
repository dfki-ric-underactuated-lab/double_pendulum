import numpy as np


def get_par_list(x0, min_rel, max_rel, n):
    if x0 != 0:
        if n % 2 == 0:
            n = n+1
        li = np.linspace(min_rel, max_rel, n)
    else:
        li = np.linspace(0, max_rel, n)
    par_list = li*x0
    return par_list
