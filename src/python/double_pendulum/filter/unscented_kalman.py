import os
import yaml
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise

from double_pendulum.filter.abstract_filter import AbstractFilter


def iden(x):
    return x


class unscentedkalman_filter(AbstractFilter):
    def __init__(
        self,
        x0=np.array([0.0, 0.0, 0.0, 0.0]),
        dt=0.01,
        measurement_noise=[0.001, 0.001, 0.1, 0.1],
        process_noise=[0.0, 0.0, 0.0, 0.0],
        filt_velocity_cut=-1.0,
        fx=None,
    ):
        super().__init__()

        self.dim_x = 4
        self.x0 = x0
        self.dt = dt
        self.measurement_noise = measurement_noise
        self.process_noise = process_noise
        self.filt_velocity_cut = filt_velocity_cut
        self.fx = fx

        self.init_()

    def init_(self):
        points = MerweScaledSigmaPoints(4, alpha=0.1, beta=2.0, kappa=-1)

        self.f = UnscentedKalmanFilter(
            dim_x=self.dim_x,
            dim_z=self.dim_x,
            dt=self.dt,
            fx=self.fx,
            hx=iden,
            points=points,
        )
        self.f.x = np.copy(self.x0)
        # self.f.P *= 0.2
        # z_std = [0.001, 0.001, 0.2, 0.2]
        self.f.R = np.diag(self.measurement_noise)
        self.f.Q = Q_discrete_white_noise(
            dim=self.dim_x, dt=self.dt, var=self.process_noise
        )

    def save_(self, save_dir):
        par_dict = {
            "x0": list(self.x0),
            "dt": self.dt,
            "measurement_noise": list(self.measurement_noise),
            "process_noise": list(self.process_noise),
            "filt_velocity_cut": self.filt_velocity_cut,
        }
        with open(
            os.path.join(save_dir, "filter_unscentedkalman_parameters.yml"), "w"
        ) as f:
            yaml.dump(par_dict, f)

    def get_filtered_state_(self, x, u, t=None):
        x_filt = np.copy(x)

        # velocity cut
        if self.filt_velocity_cut > 0.0:
            x_filt[2] = np.where(
                np.abs(x_filt[2]) < self.filt_velocity_cut, 0, x_filt[2]
            )
            x_filt[3] = np.where(
                np.abs(x_filt[3]) < self.filt_velocity_cut, 0, x_filt[3]
            )

        self.f.predict(dt=self.dt, UT=None, fx=self.fx, t=0.0, tau=u)
        self.f.update(np.asarray(x_filt))
        return self.f.x
