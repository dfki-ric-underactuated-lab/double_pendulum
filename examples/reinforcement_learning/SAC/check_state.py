import os
import numpy as np

# load_path = "lqr_data/acrobot/lqr/roa"
load_path = "lqr_data/pendubot/lqr/roa"

rho = np.loadtxt(os.path.join(load_path, "rho"))
vol = np.loadtxt(os.path.join(load_path, "vol"))
S = np.loadtxt(os.path.join(load_path, "Smatrix"))

print("rho: ", rho)
print("volume: ", vol)
print("S", S)


def check_if_state_in_roa(S, rho, x):
    xdiff = x - np.array([np.pi, 0.0, 0.0, 0.0])
    rad = np.einsum("i,ij,j", xdiff, S, xdiff)
    return rad < rho, rad


x1 = [0, 0, 0, 0]
x2 = [np.pi, 0, 0, 0]
x3 = [np.pi - 0.1, 0.1, 0, 0]
x4 = [np.pi, 0, 1.0, -1.0]
x5 = [np.pi, 0, 0.1, -0.1]

print("x={x1}: ", check_if_state_in_roa(S, rho, x1))
print("x={x2}: ", check_if_state_in_roa(S, rho, x2))
print("x={x3}: ", check_if_state_in_roa(S, rho, x3))
print("x={x4}: ", check_if_state_in_roa(S, rho, x4))
print("x={x5}: ", check_if_state_in_roa(S, rho, x5))