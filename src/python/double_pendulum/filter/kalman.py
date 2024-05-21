import os
import yaml
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from double_pendulum.filter.abstract_filter import AbstractFilter


class kalman_filter(AbstractFilter):
    def __init__(
        self,
        A,
        B,
        x0=np.array([0.0, 0.0, 0.0, 0.0]),
        dt=0.01,
        process_noise=[0.0, 0.0, 0.0, 0.0],
        measurement_noise=[0.001, 0.001, 0.1, 0.1],
        covariance_matrix=np.diag((1.0, 1.0, 1.0, 1.0)),
        filt_velocity_cut=-1.0,
    ):
        super().__init__()

        self.dim_x = 4
        self.dim_u = 2
        # discrete transition matrices
        self.A = np.eye(4) + A * dt
        self.B = B * dt
        self.x0 = x0
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.covariance_matrix = covariance_matrix
        self.filt_velocity_cut = filt_velocity_cut

        self.init_()

    def init_(self):
        self.x_hist = [self.x0]

        # First construct the object with the required dimensionality.
        self.f = KalmanFilter(dim_x=self.dim_x, dim_z=self.dim_x, dim_u=self.dim_u)

        self.f.F = np.asarray(self.A)
        self.f.B = np.asarray(self.B)

        # Assign the initial value for the state (position and velocity).
        self.f.x = np.asarray(self.x0)  # position, velocity

        # Measurement function:
        self.f.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # Covariance matrix
        # self.f.P = 1000 * np.identity(np.size(self.f.x))
        # self.f.P = np.diag((0., 0., 1000., 1000.))
        self.f.P = self.covariance_matrix

        # Measurement noise
        self.f.R = np.diag(self.measurement_noise)

        # Process noise
        self.f.Q = Q_discrete_white_noise(
            dim=self.dim_x, dt=self.dt, var=self.process_noise
        )

    def save_(self, save_dir):
        par_dict = {
            "x0": list(self.x0),
            "dt": self.dt,
            "measurement_noise": list(self.measurement_noise),
            "process_noise": list(self.process_noise),
            "covariance_matrix": list(self.covariance_matrix),
            "A": list(self.A),
            "B": list(self.B),
            "filt_velocity_cut": self.filt_velocity_cut,
        }
        with open(os.path.join(save_dir, "filter_kalman_parameters.yml"), "w") as f:
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

        # Perform one KF step
        self.f.u = np.asarray(u)
        self.f.predict()  # f.predict(f.u, f.B, f.F, f.Q)
        self.f.update([np.asarray(x_filt)])  # f.update(z, f.R, f.H)

        # Output state
        x_filt = self.f.x
        # covariance = self.f.P

        return x_filt
