
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

    si = int(np.shape(A)[0])

    al = np.block([[A, multi_dot([-B, inv(R), B.T])],
                   [-Q, -A.T]])

    #C_part1 = inv(np.eye(2*si) - dt/2.*al + dt**2/12.*np.dot(al, al))
    #C_part2 = np.eye(2*si) + dt/2.*al + dt**2/12.*np.dot(al, al)
    #C = np.dot(C_part1, C_part2)

    # Taylor
    al2 = np.dot(al, al)
    al3 = np.dot(al2, al)
    al4 = np.dot(al3, al)
    al5 = np.dot(al4, al)
    C = np.eye(2*si) + dt*al + dt**2/2.*al2 + dt**3/6.*al3 + dt**4/24.*al4 + dt**5/120*al5

    Ct = [C]
    for i in range(1, n):
        ct = np.dot(Ct[-1], C)
        Ct.append(ct)

    s0 = -np.dot(inv(Ct[-1][-si:, -si:]), Ct[-1][-si:,:si])
    S = [s0]
    for i in range(1, n):
        s = np.dot(Ct[i][-si:,:si] + np.dot(Ct[i][-si:, -si:], s0),
                   inv(Ct[i][:si,:si] + np.dot(Ct[i][:si, -si:], s0)))
        S.append(s)

    K = []
    for i in range(n):
        k = multi_dot([inv(R), B.T, S[i]])
        K.append(k)

    K = np.asarray(K)
    S = np.asarray(S)
    # print(K)
    # print(S)

    return K, S


# the functions below were taken from
# https://github.com/markwmuller/controlpy/blob/master/controlpy/analysis.py


def uncontrollable_modes(A, B, returnEigenValues=False, tolerance=1e-9):
    '''Returns all the uncontrollable modes of the pair A,B.

    tolerance defines the minimum distance we should be from the imaginary axis
     to be considered stable.

    Does the PBH test for controllability for the system:
     dx = A*x + B*u

    Returns a list of the uncontrollable modes, and (optionally)
    the corresponding eigenvalues.

    See Callier & Desoer "Linear System Theory", P. 253

    NOTE!: This can't work if we have repeated eigen-values! TODO FIXME!
    '''

    assert A.shape[0] == A.shape[1], "Matrix A is not square"
    assert A.shape[0] == B.shape[0], "Matrices A and B do not align"

    nStates = A.shape[0]
    nInputs = B.shape[1]

    eVal, eVec = np.linalg.eig(np.matrix(A))  # todo, matrix cast is ugly.

    uncontrollableModes = []
    uncontrollableEigenValues = []

    for e, v in zip(eVal, eVec.T):
        M = np.matrix(np.zeros([nStates, (nStates+nInputs)]), dtype=complex)
        M[:, :nStates] = e*np.identity(nStates) - A
        M[:, nStates:] = B

        s = np.linalg.svd(M, compute_uv=False)
        if min(s) <= tolerance:
            uncontrollableModes.append(v.T[:, 0])
            uncontrollableEigenValues.append(e)

    if returnEigenValues:
        return uncontrollableModes, uncontrollableEigenValues
    else:
        return uncontrollableModes


def is_controllable(A, B, tolerance=1e-9):
    '''Compute whether the pair (A,B) is controllable.
    tolerance defines the minimum distance we should be from the imaginary axis
     to be considered stable.

    Returns True if controllable, False otherwise.
    '''

    if uncontrollable_modes(A, B, tolerance=tolerance):
        return False
    else:
        return True


def is_stabilizable(A, B):
    '''Compute whether the pair (A,B) is stabilisable.
    Returns True if stabilisable, False otherwise.
    '''

    return is_stabilisable(A, B)


def is_stabilisable(A, B):
    '''Compute whether the pair (A,B) is stabilisable.
    Returns True if stabilisable, False otherwise.
    '''

    modes, eigVals = uncontrollable_modes(A, B, returnEigenValues=True)
    if not modes:
        return True  # controllable => stabilisable

    if max(np.real(eigVals)) >= 0:
        return False
    else:
        return True
