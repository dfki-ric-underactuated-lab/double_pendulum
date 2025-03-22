import jax.numpy as jnp
import numpy as np
from double_pendulum.controller.abstract_controller import AbstractController

from double_pendulum.controller.vimppi.mppi import Config, create_mppi


class MPPIController(AbstractController):
    """Controller Template"""

    def __init__(self, config: Config):
        super().__init__()

        self._cfg = config
        self._mppi = create_mppi(self._cfg)
        # initialize previous control sequence with zeros
        self._x_prev = None
        self._t_prev = None
        self._u_prev = None

    def set_parameters(self):
        """
        Set controller parameters. Optional.
        Can be overwritten by actual controller.
        """
        pass

    def set_goal(self, x):
        """
        Set the desired state for the controller. Optional.
        Can be overwritten by actual controller.

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        """
        self.goal = x

        self._cfg.update_goal(x)
        self._mppi = create_mppi(self._cfg)

    def init_(self):
        """
        Initialize the controller. Optional.
        Can be overwritten by actual controller.
        Initialize function which will always be called before using the
        controller.
        """
        pass

    def reset_(self):
        """
        Reset the baseline control, when a disturbance is detected. Optional.
        """
        self._u_prev = self._cfg.baseline_control_generator(self._x_prev, self.goal)
        # print(self._u_prev)

    def detect_disturbance(self, x: np.ndarray, dt: float = 0) -> bool:
        if self._x_prev is None or self._t_prev == 0:
            return False
        return self._cfg.check_disturbance(dt, self._x_prev, x, self._u_prev[0])

    def get_control_output_(self, x, t=None):
        """
        The function to compute the control input for the double pendulum's
        actuator(s).
        Supposed to be overwritten by actual controllers. The API of this
        method should not be changed. Unused inputs/outputs can be set to None.

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
        dt = t - self._t_prev if self._t_prev is not None else 0
        # Initialize at the first controller call
        if self._u_prev is None or self._x_prev is None:
            self._x_prev = x
            self.reset()
        # check for disturbance
        elif self.detect_disturbance(x, dt):
            # print("Disturbance detected")
            self.reset()

        # get control output from MPPI
        action, _u_prev = self._mppi(x, self._u_prev, self._cfg.mppi_dt)
        if np.any(np.isnan(action)):
            return np.zeros(2)
        self._u_prev = _u_prev

        # Save previous state for disturbance detection
        self._x_prev = x
        self._t_prev = t
        # bring back to numpy
        return np.array(action)
