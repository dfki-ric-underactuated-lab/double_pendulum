"""
Gamepad Controller
==============
"""

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.controller.gamepad.gamepad import GamePad


class GamepadController(AbstractController):
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
    def __init__(self, torque_limit=[5.0, 5.0],
                 gamepad_name="Logitech Logitech RumblePad 2 USB"):

        super().__init__()

        self.torque_limit = torque_limit
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
        u1 = inp[0]*self.torque_limit[0]
        u2 = inp[1]*self.torque_limit[1]
        u = [u1, u2]

        return u
