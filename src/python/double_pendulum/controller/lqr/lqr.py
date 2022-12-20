from __future__ import division, print_function

import numpy as np
import scipy.linalg
from numpy.linalg import multi_dot, inv


def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.
    dx/dt = A x + B u
    cost = integral x.T*Q*x + u.T*R*u
    """
    # copied from https://www.mwm.im/lqr-controllers-with-python/
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
    # print("X", X)

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R)*(B.T*X))
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
    return K, X, eigVals


def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    # copied from https://www.mwm.im/lqr-controllers-with-python/
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
    return K, X, eigVals


def iterative_ricatti(A, B, Q, R, n):
    """iteratively solve the dynamic ricatti equation.
    intended for finite horizon lqr
    """
    Pk = [Q]
    #k = multi_dot([scipy.linalg.inv(R + multi_dot([B.T, Pk[0], B])),
    #               multi_dot([B.T, Pk[0], A])])
    #Kk = [np.copy(k)]
    for _ in range(n):
        p = (multi_dot([A.T, Pk[0], A]) -
             multi_dot([multi_dot([A.T, Pk[0], B]),
                        inv(R + multi_dot([B.T, Pk[0], B])),
                        multi_dot([B.T, Pk[0], A])])
             + Q)
        #print("p", np.shape(p), p)
        Pk.insert(0, np.copy(p))
    Kk = []
    for i in range(1, n+1, 1):
        k = multi_dot([inv(R + multi_dot([B.T, Pk[i], B])),
                       multi_dot([B.T, Pk[i], A])])
        #print("k", np.shape(k), k)
        #Kk.insert(0, np.copy(k))
        Kk.append(k)
    Pk = np.asarray(Pk)
    Kk = np.asarray(Kk)
    return Kk, Pk

def solve_differential_ricatti(A, B, Q, R, n, dt):
    """
    Implementation based on
    'The numerical solution of the matrix Riccati differential equation'
    E.Davison, E.Maki
    https://ieeexplore.ieee.org/document/1100210?arnumber=1100210

    note:
    paper       here    usually
    \mathcal{a} al
    C           C
    K           S
    exp(at)     Ct
    R^-1 B^T p  K
    """

    si = int(np.shape(A)[0])

    al = np.block([[A, multi_dot([-B, inv(R), B.T])],
                   [-Q, -A.T]])

    # Taylor
    # eq. (11) and (12)
    al2 = np.dot(al, al)
    #al3 = np.dot(al2, al)
    #al4 = np.dot(al3, al)
    #al5 = np.dot(al4, al)
    #C = np.eye(2*si) + dt*al + dt**2/2.*al2 + dt**3/6.*al3 + dt**4/24.*al4 + dt**5/120*al5
    C_part1 = inv(np.eye(2*si) - dt/2.*al + dt**2/12.*al2)
    C_part2 = np.eye(2*si) + dt/2.*al + dt**2/12.*al2
    C = np.dot(C_part1, C_part2)

    Ct = [C]
    for i in range(1, n):
        ct = np.dot(Ct[-1], C)
        Ct.append(ct)

    # equation (9)
    s0 = -np.dot(inv(Ct[-1][-si:, -si:]), Ct[-1][-si:,:si])
    S = [s0]
    for i in range(1, n):
        # equation (8)
        s = np.dot(Ct[i][-si:,:si] + np.dot(Ct[i][-si:, -si:], s0),
                   inv(Ct[i][:si,:si] + np.dot(Ct[i][:si, -si:], s0)))
        S.append(s)

    K = []
    for i in range(n):
        # equation (5) rhs
        k = multi_dot([inv(R), B.T, S[i]])
        K.append(k)

    K = np.asarray(K)
    S = np.asarray(S)
    # print(K)
    # print(S)

    return K, S
