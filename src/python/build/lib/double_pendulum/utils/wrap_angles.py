import numpy as np


def wrap_angles(x):
    y = np.copy(x)
    y[0] = x[0] % (2*np.pi)
    y[1] = x[1] % (2*np.pi)
    return y


def wrap_angles_top(x):
    y = np.copy(x)
    y[0] = x[0] % (2*np.pi)
    y[1] = (x[1] + np.pi) % (2*np.pi) - np.pi
    return y


def wrap_angles_diff(x):
    y = np.copy(x)
    y[0] = x[0] % (2*np.pi)
    y[1] = x[1] % (2*np.pi)
    while np.abs(y[0]) > np.pi:
        y[0] -= 2*np.pi
    while np.abs(y[1]) > np.pi:
        y[1] -= 2*np.pi
    return y
