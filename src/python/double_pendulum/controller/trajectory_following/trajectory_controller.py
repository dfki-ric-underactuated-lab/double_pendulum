import os
import yaml
import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.utils.pcw_polynomial import InterpolateVector
from double_pendulum.utils.csv_trajectory import load_trajectory, trajectory_properties, load_Kk_values


class TrajectoryController(AbstractController):
    """TrajectoryController
    Controllers which executes a feedforward torque trajectory
    and optionally stabilizes the trajectory with linear feedback.

    Parameters
    ----------
    csv_path : string or path object
        path to csv file where the trajectory is stored.
        csv file should use standarf formatting used in this repo.
        If T, X, or U are provided they are preferred.
    torque_limit : array_like, optional
        shape=(2,), dtype=float, default=[0.0, 1.0]
        torque limit of the motors
        [tl1, tl2], units=[Nm, Nm]
    kK_stabilization : bool
        Whether to stabilize the trajectory with linear feedback gains
        (K matrix and additive k vector)
        (Default value=False)
    """
    def __init__(self,
                 csv_path,
                 torque_limit=[0.0, 1.0],
                 kK_stabilization=False):

        super().__init__()

        self.torque_limit = torque_limit
        self.kK_stabilization = kK_stabilization

        self.T, self.X, self.U = load_trajectory(csv_path, with_tau=True)
        self.dt, self.max_t, _, _ = trajectory_properties(self.T, self.X)
        if self.kK_stabilization:
            self.K1, self.K2, self.k1, self.k2 = load_Kk_values(csv_path)

    def get_control_output_(self, x, t):
        """
        The function to compute the control input for the double pendulum's
        actuator(s).

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        t : float, optional
            time, unit=[s]
            (Default value=None)

        Returns
        -------
        array_like
            shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
        """
        n = int(np.around(min(t, self.max_t) / self.dt))

        u1 = self.U[n][0]
        u2 = self.U[n][1]

        if self.kK_stabilization:
            x_des = self.X[n]

            K1 = self.K1[n]
            K2 = self.K2[n]

            # k1 = self.k1[n]
            # k2 = self.k2[n]
            k1 = 0.0
            k2 = 0.0

            u1 = u1 + k1 - np.dot(K1, x_des - x)
            u2 = u2 + k2 - np.dot(K2, x_des - x)
        u1 = np.clip(u1, -self.torque_limit[0], self.torque_limit[0])
        u2 = np.clip(u2, -self.torque_limit[1], self.torque_limit[1])

        u = np.asarray([u1, u2])
        return u

    def get_init_trajectory(self):
        """
        Get the initial (reference) trajectory used by the controller.

        Returns
        -------
        numpy_array
            time points, unit=[s]
            shape=(N,)
        numpy_array
            shape=(N, 4)
            states, units=[rad, rad, rad/s, rad/s]
            order=[angle1, angle2, velocity1, velocity2]
        numpy_array
            shape=(N, 2)
            actuations/motor torques
            order=[u1, u2],
            units=[Nm]
        """
        return self.T, self.X, self.U


class TrajectoryInterpController(AbstractController):
    """TrajectoryController
    Controllers which executes a feedforward torque trajectory
    and optionally stabilizes the trajectory with linear feedback.
    The trajectory can be interpolated with polynomials.

    Parameters
    ----------
    csv_path : string or path object
        path to csv file where the trajectory is stored.
        csv file should use standarf formatting used in this repo.
        If T, X, or U are provided they are preferred.
    torque_limit : array_like, optional
        shape=(2,), dtype=float, default=[0.0, 1.0]
        torque limit of the motors
        [tl1, tl2], units=[Nm, Nm]
    kK_stabilization : bool
        Whether to stabilize the trajectory with linear feedback gains
        (K matrix and additive k vector)
        (Default value=False)
    num_break : int
        number of break points used for interpolation
        (Default value = 40)
    """
    def __init__(self,
                 csv_path,
                 torque_limit=[0.0, 1.0],
                 kK_stabilization=False,
                 num_break=40):

        super().__init__()

        self.torque_limit = torque_limit
        self.kK_stabilization = kK_stabilization

        self.T, self.X, self.U = load_trajectory(csv_path, with_tau=True)
        self.dt, self.max_t, _, _ = trajectory_properties(self.T, self.X)
        if self.kK_stabilization:
            self.K1, self.K2, self.k1, self.k2 = load_Kk_values(csv_path)

        self.U_interp = InterpolateVector(
                T=self.T,
                X=self.U,
                num_break=num_break,
                poly_degree=3)

        if self.kK_stabilization:
            self.X_interp = InterpolateVector(
                    T=self.T,
                    X=self.X,
                    num_break=num_break,
                    poly_degree=3)
            self.K1_interp = InterpolateVector(
                    T=self.T,
                    X=self.K1,
                    num_break=num_break,
                    poly_degree=3)
            self.K2_interp = InterpolateVector(
                    T=self.T,
                    X=self.K2,
                    num_break=num_break,
                    poly_degree=3)
            k = np.swapaxes([self.k1, self.k2], 0, 1)
            self.k_interp = InterpolateVector(
                    T=self.T,
                    X=k,
                    num_break=num_break,
                    poly_degree=3)

    def get_control_output_(self, x, t):
        """
        The function to compute the control input for the double pendulum's
        actuator(s).

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        t : float, optional
            time, unit=[s]
            (Default value=None)

        Returns
        -------
        array_like
            shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
        """
        tt = min(t, self.max_t)

        uu = self.U_interp.get_value(tt)
        u1 = uu[0]
        u2 = uu[1]

        if self.kK_stabilization:
            x_des = self.X_interp.get_value(tt)

            K1 = self.K1_interp.get_value(tt)
            K2 = self.K2_interp.get_value(tt)

            # k = self.k_interp.get_value(tt)
            # k1 = k[0]
            # k2 = k[1]
            k1 = 0.0
            k2 = 0.0

            u1 = u1 + k1 - np.dot(K1, x_des - x)
            u2 = u2 + k2 - np.dot(K2, x_des - x)
        #print(x-x_des, K1[0], K1[1], K1[2], K1[3], uu[0], u1)
        u1 = np.clip(u1, -self.torque_limit[0], self.torque_limit[0])
        u2 = np.clip(u2, -self.torque_limit[1], self.torque_limit[1])

        u = np.asarray([u1, u2])
        return u

    def get_init_trajectory(self):
        """
        Get the initial (reference) trajectory used by the controller.

        Returns
        -------
        numpy_array
            time points, unit=[s]
            shape=(N,)
        numpy_array
            shape=(N, 4)
            states, units=[rad, rad, rad/s, rad/s]
            order=[angle1, angle2, velocity1, velocity2]
        numpy_array
            shape=(N, 2)
            actuations/motor torques
            order=[u1, u2],
            units=[Nm]
        """
        return self.T, self.X, self.U

    def save_(self, save_dir):
        """
        Save controller parameters

        Parameters
        ----------
        save_dir : string or path object
            directory where the parameters will be saved
        """

        par_dict = {
                "torque_limit1" : self.torque_limit[0],
                "torque_limit2" : self.torque_limit[1],
                "kK_stabilization" : self.kK_stabilization,
                "dt" : self.dt,
                "max_t" : self.max_t,
        }

        with open(os.path.join(save_dir, "controller_trajfollowing_parameters.yml"), 'w') as f:
            yaml.dump(par_dict, f)
