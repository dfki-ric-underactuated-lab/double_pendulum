import os
import yaml
import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController


class PointPIDController(AbstractController):
    """PointPIDController
    PID controller with a fix state as goal.

    Parameters
    ----------
    torque_limit : array_like, optional
        shape=(2,), dtype=float, default=[1.0, 1.0]
        torque limit of the motors
        [tl1, tl2], units=[Nm, Nm]
    dt : float
        timestep , unit=[s]
        (Default value=0.01)
    """

    def __init__(self, torque_limit=[1.0, 1.0], dt=0.01, modulo_angles=True):

        super().__init__()

        self.torque_limit = torque_limit
        self.dt = dt
        self.modulo_angles = modulo_angles

        # default weights
        self.Kp = 1.0
        self.Ki = 0.0
        self.Kd = 0.1
        self.goal = np.array([np.pi, 0.0, 0.0, 0.0])

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

    def set_goal(self, x):
        """set_goal.
        Set goal for the controller.

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        """
        self.goal = x

    def init_(self):
        """
        Initialize the controller.
        """
        self.errors1 = []
        self.errors2 = []

    def get_control_output_(self, x, t=None):
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
        e1 = self.goal[0] - x[0]
        e2 = self.goal[1] - x[1]
        if self.modulo_angles:
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

        u1 = P1 + I1 + D1
        u2 = P2 + I2 + D2

        u1 = np.clip(u1, -self.torque_limit[0], self.torque_limit[0])
        u2 = np.clip(u2, -self.torque_limit[1], self.torque_limit[1])
        u = np.asarray([u1, u2])
        return u

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
        }

        with open(os.path.join(save_dir, "controller_pid_parameters.yml"), "w") as f:
            yaml.dump(par_dict, f)

        np.savetxt(
            os.path.join(save_dir, "controller_pid_errors.csv"),
            [self.errors1, self.errors2],
        )
