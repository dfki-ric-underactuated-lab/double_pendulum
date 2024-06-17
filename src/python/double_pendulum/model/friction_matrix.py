import math
import numpy as np


def yb_friction_matrix(dq_vec):
    """
    matrix to be multiplied with damping/friction coefficients
    resulting in equation of motion contribution

    Parameters
    ----------
    dq_vec : array-like
        shape=(2,),
        velocities of the double pendulum,
        order=[dq1, dq2],
        units=[m/s],

    Returns
    -------
    numpy array
        shape=(2,4)
        friction matrix
    """

    y_11 = math.atan(100*dq_vec[0])
    y_12 = dq_vec[0]

    y_23 = math.atan(100*dq_vec[1])
    y_24 = dq_vec[1]

    yb_fric = np.array([[y_11, y_12, 0, 0],
                        [0, 0, y_23, y_24]])
    return yb_fric
