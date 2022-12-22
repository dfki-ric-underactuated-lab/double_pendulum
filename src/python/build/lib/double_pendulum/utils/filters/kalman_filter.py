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
    def __init__(self, A, B, dim_x=4, dim_u=2,
                 x0=np.array([0., 0., 0., 0.]),
                 dt=0.01,
                 process_noise=[0., 0., 0., 0.],
                 measurement_noise=[0.001, 0.001, 0.1, 0.1],
                 covariance_matrix=np.diag((1., 1., 1., 1.))):

        self.dim_x = dim_x
        self.dim_u = dim_u
        self.measurement_noise = measurement_noise

        self.x0 = np.asarray(x0)
        self.dt = dt

        # discrete transition matrices
        self.A = np.eye(4) + A*dt
        self.B = B*dt

        self.x_data = [self.x0]
        self.u_data = []

        # First construct the object with the required dimensionality.
        self.f = KalmanFilter(
                dim_x=self.dim_x,
                dim_z=self.dim_x,
                dim_u=self.dim_u)

        self.f.F = np.asarray(self.A)
        self.f.B = np.asarray(self.B)

        # Assign the initial value for the state (position and velocity).
        self.f.x = np.asarray(self.x0)  # position, velocity

        # Measurement function:
        self.f.H = np.array([[1., 0., 0., 0.],
                             [0., 1., 0., 0.],
                             [0., 0., 1., 0.],
                             [0., 0., 0., 1.]])

        # Covariance matrix
        #self.f.P = 1000 * np.identity(np.size(self.f.x))
        #self.f.P = np.diag((0., 0., 1000., 1000.))
        self.f.P = covariance_matrix

        # Measurement noise
        self.f.R = np.diag(self.measurement_noise)

        # Process noise
        self.f.Q = Q_discrete_white_noise(dim=self.dim_x, dt=self.dt, var=process_noise)

    def __call__(self, x, u):

        # A, B = self.plant.linear_matrices(
        #                 self.x_data[-1],
        #                 u)

        # Perform one KF step
        self.f.u = np.asarray(u)
        self.f.predict()  # f.predict(f.u, f.B, f.F, f.Q)
        self.f.update([np.asarray(x)])  # f.update(z, f.R, f.H)
        self.x_data.append(np.copy(x))

        # Output state
        x_est = self.f.x
        # covariance = self.f.P

        return x_est


def main():
    dt = 0.1
    control_input = np.array([1., 1.])  # TODO
    data_measured = np.array([3., 3.])
    x, P = kalman_filter(data_measured, control_input, dt)
    print("Kalman Filter \ndt:", dt, "\nu:", control_input, "\nz:", data_measured, "\nx:", x, "\nP:", P)


if __name__ == "__main__":
    main()



