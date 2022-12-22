from functools import partial
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise


def iden(x):
    return x


class unscented_kalman_filter_rt():
    def __init__(self, dim_x=4,
                 x0=np.array([0., 0., 0., 0.]),
                 dt=0.01,
                 measurement_noise=[0.001, 0.001, 0.1, 0.1],
                 process_noise=[0., 0., 0., 0.],
                 fx=None):

        self.dim_x = dim_x
        self.dt = dt
        #self.fx = partial(fx, t=0.)
        self.fx = fx

        self.last_x = np.asarray(x0)

        points = MerweScaledSigmaPoints(
                4, alpha=.1, beta=2., kappa=-1)

        self.f = UnscentedKalmanFilter(
                dim_x=dim_x,
                dim_z=dim_x,
                dt=dt,
                fx=None,
                hx=iden,
                points=points)
        self.f.x = np.copy(x0)
        #self.f.P *= 0.2
        #z_std = [0.001, 0.001, 0.2, 0.2]
        self.f.R = np.diag(measurement_noise)
        self.f.Q = Q_discrete_white_noise(dim=dim_x, dt=dt, var=process_noise)

    def __call__(self, x, u):
        self.f.predict(dt=self.dt, UT=None, fx=self.fx, t=0., tau=u)
        self.f.update(np.asarray(x))
        self.last_x = np.copy(x)
        return self.f.x
