"""
Gamepad Controller
==============
"""

import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.controller.gamepad.gamepad import GamePad
from double_pendulum.controller.pid.point_pid_controller import PointPIDController


class GamepadPIDController(AbstractController):
    """
    Controller actuates the pendulum with a gamepad

    Parameters
    ----------
    gamepad_name: string
        string refering to the gamepad type
        Currently supported:
            - "Logitech Logitech RumblePad 2 USB"
        (Default value="Logitech Logitech RumblePad 2 USB")
    torque_limit : float, default=np.inf
        the torque_limit of the pendulum actuator
    """

    def __init__(
        self,
        torque_limit=[5.0, 5.0],
        pid_dt=0.01,
        pid_modulo_angles=True,
        pid_pos_contribution_limit=[np.inf, np.inf],
        pid_gains=[1.0, 0.0, 1.0],
        gamepad_name="Logitech Logitech RumblePad 2 USB",
        max_vel=5.0,
    ):

        super().__init__()

        self.pid_controller = PointPIDController(
            torque_limit=torque_limit,
            dt=pid_dt,
            modulo_angles=pid_modulo_angles,
            pos_contribution_limit=pid_pos_contribution_limit,
        )
        self.pid_controller.set_parameters(pid_gains[0], pid_gains[1], pid_gains[2])
        self.pid_controller.init()

        self.pid_controller_damp = PointPIDController(
            torque_limit=torque_limit,
            dt=pid_dt,
            modulo_angles=pid_modulo_angles,
            pos_contribution_limit=[0.0, 0.0],
        )
        self.pid_controller_damp.set_parameters(0.0, 0.0, 1.0)
        self.pid_controller_damp.init()

        self.torque_limit = torque_limit
        self.max_vel = max_vel
        self.GP = GamePad(gamepad_name)

    def get_control_output_(self, x, t=None):
        """
        The function to compute the control input for the double pendulum's
        actuator(s) based on the gamepad input.

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

        inp = self.GP.read()
        th1 = np.pi / 2.0 + np.arctan2(inp[1], inp[0])
        th2 = np.pi / 2.0 + np.arctan2(inp[3], inp[2]) - x[0]

        # if np.abs(inp[0]) > 0.1:
        #     th1 = np.pi / 2.0 + np.arctan2(inp[1], inp[0])
        # else:
        #     if inp[1] > 0.8:
        #         th1 = np.pi
        #     else:
        #         th1 = 0.0
        # if np.abs(inp[2]) > 0.1:
        #     th2 = np.pi / 2.0 + np.arctan2(inp[3], inp[2]) - x[0]
        # else:
        #     if inp[3] > 0.8:
        #         th2 = np.pi - x[0]
        #     else:
        #         th2 = -x[0]
        self.pid_controller.set_goal([th1, th2])
        u = self.pid_controller.get_control_output_(x, t)
        u_damp = self.pid_controller_damp.get_control_output_(x, t)

        if np.abs(inp[0]) < 0.1 and np.abs(inp[1]) < 0.1:
            u[0] = u_damp[0]
        if np.abs(inp[2]) < 0.1 and np.abs(inp[3]) < 0.1:
            u[1] = u_damp[1]

        if np.max(np.abs(x[2:]) > self.max_vel):
            u = u_damp

        # print(inp, th1, th2, u)
        # u1 = inp[0] * self.torque_limit[0]
        # u2 = inp[1] * self.torque_limit[1]
        # u = [u1, u2]

        return u
