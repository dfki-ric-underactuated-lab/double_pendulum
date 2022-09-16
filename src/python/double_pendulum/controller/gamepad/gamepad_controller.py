"""
Gamepad Controller
==============
"""

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.controller.gamepad.gamepad import GamePad


class GamepadController(AbstractController):
    """
    Controller actuates the pendulum with a gamepad
    """
    def __init__(self, torque_limit=[5.0, 5.0],
                 gamepad_name="Logitech Logitech RumblePad 2 USB"):
        """
        Controller actuates the pendulum with a gamepad

        Parameters
        ----------
        torque_limit : float, default=np.inf
            the torque_limit of the pendulum actuator
        """

        super().__init__()

        self.torque_limit = torque_limit
        self.GP = GamePad(gamepad_name)

    def get_control_output_(self, x, t=None):

        inp = self.GP.read()
        u1 = inp[0]*self.torque_limit[0]
        u2 = inp[1]*self.torque_limit[1]
        u = [u1, u2]

        return u
