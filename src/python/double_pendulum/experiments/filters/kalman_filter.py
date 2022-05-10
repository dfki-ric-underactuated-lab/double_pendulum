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


def main():
    dt = 0.1
    control_input = np.array([1., 1.])  # TODO
    data_measured = np.array([3., 3.])
    x, P = kalman_filter(data_measured, control_input, dt)
    print("Kalman Filter \ndt:", dt, "\nu:", control_input, "\nz:", data_measured, "\nx:", x, "\nP:", P)


if __name__ == "__main__":
    main()



