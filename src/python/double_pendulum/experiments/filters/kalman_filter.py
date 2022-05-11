import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


def kalman_filter(data_measured, control_input, dt):
    """
    Kalman Filter
    Filter that tracks position and velocity using a sensor that only reads position.
    State vector: [q1 q2 q1_dot q2_dot]
    Reference: https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html
    """

    # First construct the object with the required dimensionality.
    f = KalmanFilter(dim_x=4, dim_z=2)

    # Assign the initial value for the state (position and velocity).
    f.x = np.array([0., 0., 0., 0.])  # position, velocity

    # State transition matrix
    f.F = np.array([[1., 0., dt, 0.],
                    [0., 1., 0., dt],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]])

    # Control transition matrix  # TODO
    f.B = np.array([[2., 0.],
                   [0., 2.],
                   [0., 0.],
                   [0., 0.]])

    # Measurement function:
    f.H = np.array([[1., 0., 0., 0.],
                    [0., 1., 0., 0.]])

    # Covariance matrix
    f.P = 1000 * np.identity(np.size(f.x))

    # Measurement noise
    f.R = np.array([[5., 0.],
                    [0., 5.]])

    # Process noise
    f.Q = Q_discrete_white_noise(dim=4, dt=dt, var=0.13)

    # Perform one KF step
    f.u = control_input
    z = data_measured
    f.predict()  # f.predict(f.u, f.B, f.F, f.Q)
    f.update(z)  # f.update(z, f.R, f.H)

    # Output state
    data_filtered = f.x
    covariance = f.P

    return data_filtered, covariance


class kalman_filter_rt():
    """
    kalman filter for realtime data processing
    """
    def __init__(self, dim_x=4, dim_u=2,
                 measurement_noise=None):

        self.dim_x = dim_x
        self.dim_u = dim_u
        self.measurement_noise = measurement_noise

        self.x0 = np.array([0., 0., 0., 0.])
        self.dt = 0.01
        self.f = None

    def set_parameters(self, x0=np.array([0., 0., 0., 0.]), dt=0.01):
        self.x0 = np.asarray(x0)
        self.dt = dt

    def init(self):
        # First construct the object with the required dimensionality.
        self.f = KalmanFilter(
                dim_x=self.dim_x,
                dim_z=self.dim_x,
                dim_u=self.dim_u)

        # Assign the initial value for the state (position and velocity).
        self.f.x = np.asarray(self.x0)  # position, velocity

        # not sure if these are necessary
        # # Measurement function:
        # f.H = np.array([[1., 0., 0., 0.],
        #                 [0., 1., 0., 0.]])

        # # Covariance matrix
        # f.P = 1000 * np.identity(np.size(f.x))

        # Measurement noise
        if self.measurement_noise is None:
            self.measurement_noise = np.zeros(self.dim_x, self.dim_x)
        self.f.R = np.array(self.measurement_noise)

        # Process noise
        self.f.Q = Q_discrete_white_noise(dim=self.dim_x, dt=self.dt, var=0.13)  # TODO: variance value

    def __call__(self, A, B, measured_x, u):

        self.f.F = np.asarray(A)
        self.f.B = np.asarray(B)

        # Perform one KF step
        self.f.u = np.asarray(u)
        self.f.predict()  # f.predict(f.u, f.B, f.F, f.Q)
        self.f.update(np.asarray(measured_x))  # f.update(z, f.R, f.H)

        # Output state
        est_x = self.f.x
        #covariance = self.f.P

        return est_x


def main():
    dt = 0.1
    control_input = np.array([1., 1.])  # TODO
    data_measured = np.array([3., 3.])
    x, P = kalman_filter(data_measured, control_input, dt)
    print("Kalman Filter \ndt:", dt, "\nu:", control_input, "\nz:", data_measured, "\nx:", x, "\nP:", P)


if __name__ == "__main__":
    main()



