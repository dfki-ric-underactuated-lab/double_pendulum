import numpy as np

from double_pendulum.model.plant import DoublePendulumPlant
from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.analysis.leaderboard import check_if_up
from double_pendulum.controller.pid.point_pid_controller import PointPIDController


class GlobalPolicyTestingController(AbstractController):
    """
    Controller to combine two controllers and switch between them on conditions

    Parameters
    ----------
    controller : Controller object
    """

    def __init__(
        self,
        controller,
        goal=np.array([np.pi, 0.0, 0.0, 0.0]),
        knockdown_after=2,
        knockdown_length=1,
        method="height",
        height=0.9,
        eps=[1e-2, 1e-2, 1e-2, 1e-2],
        mpar=None,
        dt=0.001,
    ):
        super().__init__()

        self.controllers = controller
        self.knockdown_after = knockdown_after
        self.knockdown_length = knockdown_length
        self.method = method
        self.height = height
        self.eps = eps
        self.mpar = mpar

        plant = DoublePendulumPlant(model_pars=self.mpar)
        super().set_gravity_compensation(plant)
        self.use_gravity_compensation = False

        self.pid_controller = PointPIDController(torque_limit=mpar.tl, dt=dt)
        self.pid_controller.set_parameters(10.0, 0.1, 1.0)

        self.up_timer = 0.0
        self.last_t = 0.0

        self.perturbation_mode = False

        # negative_amplitudes = np.random.uniform(-5, -3, (100, 2))
        # positive_amplitudes = np.random.uniform(3, 5, (100, 2))
        # mask = np.random.rand(100, 2) > 0.5
        # self.random_amplitudes = np.where(
        #     mask, positive_amplitudes, negative_amplitudes
        # )

        self.random_states = np.random.uniform(0.0, 2 * np.pi, (1000, 4))
        self.random_states.T[2:] -= np.pi
        self.random_states.T[2:] *= 4.0

        self.random_timer = 0
        self.random_counter = 0

    def init_(self):
        """
        initialize both controllers
        """
        self.controllers.init_()

    def set_parameters(self, controller_pars):
        """
        Set parametrers for both controllers.

        Parameters
        ----------
        controller_pars : list
            parameters for controller to be parsed to set_parameters
        """
        self.controllers.set_parameters(*controller_pars)

    def set_start(self, x):
        """
        Set start state for the controller.

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        """
        self.controllers.set_start(x)

    def set_goal(self, x):
        """
        Set the desired state for the controllers.

        Parameters
        ----------
        x : array like
            the desired goal state of the controllers
        """
        self.controllers.set_goal(x)

    def save_(self, save_dir):
        """
        Save controllers' parameters.

        Parameters
        ----------
        save_dir : string or path object
            directory where the parameters will be saved
        """
        self.controllers.save_(save_dir)

    def reset_(self):
        """
        Reset controllers.
        """
        self.controllers.reset_()

    def get_control_output_(self, x, t):
        """
        The function to compute the control input for the double pendulum's
        actuator(s).
        Will check the switch condition, potetntiolly switch the active
        controller and use the active controller.

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        t : float, optional
            time, unit=[s]

        Returns
        -------
        array_like
            shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
        """

        # check if in goal region
        up = check_if_up(
            x, method=self.method, mpar=self.mpar, eps=self.eps, height=self.height
        )
        if up:
            self.up_timer += t - self.last_t
        else:
            self.up_timer = 0.0

        if self.up_timer >= self.knockdown_after and not self.perturbation_mode:
            self.perturbation_mode = True
            self.random_timer = 0.0
            self.random_counter += 1
            self.pid_controller.set_goal(self.random_states[self.random_counter])
            self.use_gravity_compensation = True

        if self.random_timer > self.knockdown_length and self.perturbation_mode:
            self.perturbation_mode = False
            self.up_timer = 0.0
            self.use_gravity_compensation = False

        if self.perturbation_mode:
            # u = self.random_amplitudes[self.random_counter]
            u = self.pid_controller.get_control_output_(x, t)
            self.random_timer += t - self.last_t
        else:
            u = self.controllers.get_control_output_(x, t)

        self.last_t = t

        return u

    def get_forecast(self):
        """
        Get a forecast trajectory as planned by the controller.
        Uses active controller
        """
        return self.controllers.get_forecast()

    def get_init_trajectory(self):
        """
        Get the initial (reference) trajectory as planned by the controller.
        Uses active controller
        """
        return self.controllers.get_init_trajectory()
