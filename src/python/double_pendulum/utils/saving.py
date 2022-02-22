import numpy as np


def save_trajectory(filename, T, X, U):
    TT = np.asarray(T)
    XX = np.asarray(X)
    UU = np.asarray(U)
    data = np.asarray([TT,
                       XX.T[0], XX.T[1], XX.T[2], XX.T[3],
                       UU.T[0], UU.T[1]]).T
    np.savetxt(filename, data, delimiter=",",
               header="time, pos1, pos2, vel1, vel2, tau1, tau2")
