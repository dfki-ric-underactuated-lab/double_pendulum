import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from matplotlib import patches


def directSphere(d, r_i=0, r_o=1):
    """
    Implementation: Krauth, Werner. Statistical Mechanics: Algorithms and
    Computations. Oxford Master Series in Physics 13. Oxford: Oxford University
    Press, 2006. page 42
    """
    # vector of univariate gaussians:
    rand = np.random.normal(size=d)
    # get its euclidean distance:
    dist = np.linalg.norm(rand, ord=2)
    # divide by norm
    normed = rand/dist

    # sample the radius uniformly from 0 to 1
    rad = np.random.uniform(r_i, r_o**d)**(1/d)
    # the r**d part was not there in the original implementation.
    # I added it in order to be able to change the radius of the sphere
    # multiply with vect and return
    return normed*rad


def quadForm(M, x):
    """
    Helper function to compute quadratic forms such as x^TMx
    """
    return np.dot(x, np.dot(M, x))


def sampleFromEllipsoid(S, rho, rInner=0, rOuter=1):
    lamb, eigV = np.linalg.eigh(S/rho)
    d = len(S)
    xy = directSphere(d, r_i=rInner, r_o=rOuter)  # sample from outer shells
    # transform sphere to ellipsoid
    # (refer to e.g. boyd lectures on linear algebra)
    T = np.linalg.inv(np.dot(np.diag(np.sqrt(lamb)), eigV.T))
    return np.dot(T, xy.T).T


def volEllipsoid(rho, M):
    """
    Calculate the Volume of a Hyperellipsoid Volume of the Hyperllipsoid
    according to
    https://math.stackexchange.com/questions/332391/volume-of-hyperellipsoid/332434
    Intuition: https://textbooks.math.gatech.edu/ila/determinants-volumes.html
    Volume of n-Ball https://en.wikipedia.org/wiki/Volume_of_an_n-ball
    """

    # For a given hyperellipsoid, find the transformation that when applied to
    # the n Ball yields the hyperellipsoid
    lamb, eigV = np.linalg.eigh(M/rho)
    A = np.dot(np.diag(np.sqrt(lamb)), eigV.T)  # transform ellipsoid to sphere
    detA = np.linalg.det(A)

    # Volume of n Ball (d dimensions)
    d = M.shape[0]  # dimension
    volC = (np.pi**(d/2)) / (gamma((d/2)+1))

    # Volume of Ellipse
    volE = volC/detA
    return volE


"""
Visualization functions used for RoA estimation
"""


def getEllipseParamsFromQuad(s0Idx, s1Idx, rho, S):
    """
    Returns ellipses in the plane defined by the states matching the indices
    s0Idx and s1Idx for funnel plotting.
    """

    ellipse_mat = np.array([[S[s0Idx][s0Idx], S[s0Idx][s1Idx]],
                            [S[s1Idx][s0Idx], S[s1Idx][s1Idx]]])*(1/rho)

    # eigenvalue decomposition to get the axes
    w, v = np.linalg.eigh(ellipse_mat)

    try:
        # let the smaller eigenvalue define the width (major axis*2!)
        width = 2/float(np.sqrt(w[0]))
        height = 2/float(np.sqrt(w[1]))
        # the angle of the ellipse is defined by the eigenvector assigned to
        # the smallest eigenvalue (because this defines the major axis (width
        # of the ellipse))
        angle = np.rad2deg(np.arctan2(v[:, 0][1], v[:, 0][0]))

    except:
        print("paramters do not represent an ellipse.")

    return width, height, angle


def getEllipsePatch(x0, x1, s0Idx, s1Idx, rho, S):
    """
    just return the patches object. I.e. for more involved plots...
    x0 and x1 -> centerpoint
    """
    w, h, a = getEllipseParamsFromQuad(s0Idx, s1Idx, rho, S)
    return patches.Ellipse((x0, x1), w, h, a, alpha=1, ec="red", facecolor="none")


def getEllipsePatches(x0, x1, s0Idx, s1Idx, rhoHist, S):
    p = []
    for rhoVal in rhoHist:
        p.append(getEllipsePatch(x0, x1, s0Idx, s1Idx, rhoVal, S))

    return p


def plotEllipse(x0, x1, s0Idx, s1Idx, rho, S, save_to=None, show=True):
    p = getEllipsePatch(x0, x1, s0Idx, s1Idx, rho, S)

    fig, ax = plt.subplots()
    ax.add_patch(p)
    l = np.max([p.width, p.height])
    ax.set_xlim(x0-l/2, x0+l/2)
    ax.set_ylim(x1-l/2, x1+l/2)
    ax.grid(True)
    if not (save_to is None):
        plt.savefig(save_to)
    if show:
        plt.show()
    plt.close()
