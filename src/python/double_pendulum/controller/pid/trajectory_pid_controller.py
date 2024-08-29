import os
import yaml
import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.utils.pcw_polynomial import InterpolateVector
from double_pendulum.utils.csv_trajectory import load_trajectory, trajectory_properties


class TrajPIDController(AbstractController):
    """TrajPIDController
    PID controller for following a trajectory.

    Parameters
    ----------
    T : array_like
        shape=(N,)
        time points of reference trajectory, unit=[s]
        (Default value=None)
    X : array_like
        shape=(N, 4)
        reference trajectory states
        order=[angle1, angle2, velocity1, velocity2]
        units=[rad, rad, rad/s, rad/s]
        (Default value=None)
    U : array_like
        shape=(N, 2)
        reference trajectory actuations/motor torques
        order=[u1, u2],
        units=[Nm]
        (Default value=None)
    csv_path : string or path object
        path to csv file where the trajectory is stored.
        csv file should use standarf formatting used in this repo.
        If T, X, or U are provided they are preferred.
        (Default value=None)
    use_feed_forward_torque : bool
        whether to use feed forward torque for the control output
        (Default value=True)
    torque_limit : array_like, optional
        shape=(2,), dtype=float, default=[0.0, 1.0]
        torque limit of the motors
        [tl1, tl2], units=[Nm, Nm]
    num_break : int
        number of break points used for interpolation
        (Default value = 40)
    """

    def __init__(
        self,
        T=None,
        X=None,
        U=None,
        csv_path=None,
        use_feed_forward_torque=True,
        torque_limit=[0.0, 1.0],
        num_break=40,
    ):
        super().__init__()

        self.use_ff = use_feed_forward_torque
        self.torque_limit = torque_limit

        # load trajectory
        if csv_path is not None:
            self.T, self.X, self.U = load_trajectory(
                csv_path=csv_path, with_tau=self.use_ff
            )
        elif T is not None and X is not None:
            self.T = T
            self.X = X
            if U is not None:
                self.U = U
            else:
                self.U = np.zeros((len(self.T), 2))
        else:
            print("Please Parse a trajectory to the TrajPIDController")
            exit()

        self.dt, self.max_t, _, _ = trajectory_properties(self.T, self.X)

        # interpolate trajectory
        self.P_interp = InterpolateVector(
            T=self.T, X=self.X.T[:2].T, num_break=num_break, poly_degree=3
        )

        if self.use_ff:
            self.U_interp = InterpolateVector(
                T=self.T, X=self.U, num_break=num_break, poly_degree=3
            )

        # default weights
        self.Kp = 10.0
        self.Ki = 0.0
        self.Kd = 0.1

        # init pars
        self.errors1 = []
        self.errors2 = []

    def set_parameters(self, Kp, Ki, Kd):
        """
        Set controller gains.

        Parameters
        ----------
        Kp : float
            Gain for position error
        Ki : float
            Gain for integrated error
        Kd : float
            Gain for differentiated error
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def init_(self):
        """
        Initialize the controller.
        """
        self.errors1 = []
        self.errors2 = []

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

        p = self.P_interp.get_value(tt)
        e1 = p[0] - x[0]
        e2 = p[1] - x[1]
        e1 = (e1 + np.pi) % (2 * np.pi) - np.pi
        e2 = (e2 + np.pi) % (2 * np.pi) - np.pi
        self.errors1.append(e1)
        self.errors2.append(e2)

        P1 = self.Kp * e1
        P2 = self.Kp * e2

        I1 = self.Ki * np.sum(np.asarray(self.errors1)) * self.dt
        I2 = self.Ki * np.sum(np.asarray(self.errors2)) * self.dt

        if len(self.errors1) > 2:
            D1 = self.Kd * (self.errors1[-1] - self.errors1[-2]) / self.dt
            D2 = self.Kd * (self.errors2[-1] - self.errors2[-2]) / self.dt
        else:
            D1 = 0.0
            D2 = 0.0

        if self.use_ff:
            uu = self.U_interp.get_value(tt)
            u1 = uu[0] + P1 + I1 + D1
            u2 = uu[1] + P2 + I2 + D2
        else:
            u1 = P1 + I1 + D1
            u2 = P2 + I2 + D2

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
        Save the energy trajectory to file.

        Parameters
        ----------
        path : string or path object
            directory where the parameters will be saved
        """

        par_dict = {
            "dt": self.dt,
            "torque_limit1": self.torque_limit[0],
            "torque_limit2": self.torque_limit[1],
            "Kp": self.Kp,
            "Ki": self.Ki,
            "Kd": self.Kd,
            "goal_x1": self.goal[0],
            "goal_x2": self.goal[1],
            "goal_x3": self.goal[2],
            "goal_x4": self.goal[3],
            "use_feed_forward_torque": self.use_ff,
        }

        with open(
            os.path.join(save_dir, "controller_traj_pid_parameters.yml"), "w"
        ) as f:
            yaml.dump(par_dict, f)

        np.savetxt(
            os.path.join(save_dir, "controller_traj_pid_errors.csv"),
            [self.errors1, self.errors2],
        )
