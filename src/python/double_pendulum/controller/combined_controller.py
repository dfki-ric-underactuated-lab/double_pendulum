import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController


class CombinedController(AbstractController):
    """
    Controller to combine two controllers and switch between them on conditions

    Parameters
    ----------
    controller1 : Controller object
        First Controller
    controller2 : Controller object
        Second Controller
    condition1 : function of (x, t)
        condition to switch to controller 1
        must be a functin of the state x and the time t
    condition2 : function of (x, t)
        condition to switch to controller 2
        must be a functin of the state x and the time t
    compute_both : bool
        Flag whether to compute the control output for both controllers at each
        timestep or only for the active one
    """

    def __init__(
        self,
        controller1,
        controller2,
        condition1,
        condition2,
        compute_both=False,
        verbose=False,
    ):
        super().__init__()

        self.controllers = [controller1, controller2]
        self.active = 0

        self.conditions = [condition1, condition2]

        self.compute_both = compute_both
        self.verbose = verbose

    def init_(self):
        """
        initialize both controllers
        """
        self.controllers[0].init_()
        self.controllers[1].init_()
        self.active = 0

    def set_parameters(self, controller1_pars, controller2_pars):
        """
        Set parametrers for both controllers.

        Parameters
        ----------
        controller1_pars : list
            parameters for controller 1 to be parsed to set_parameters
        controller2_pars : list
            parameters for controller 1 to be parsed to set_parameters
        """
        self.controllers[0].set_parameters(*controller1_pars)
        self.controllers[1].set_parameters(*controller2_pars)

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
        self.controllers[0].set_start(x)
        self.controllers[1].set_start(x)

    def set_goal(self, x):
        """
        Set the desired state for the controllers.

        Parameters
        ----------
        x : array like
            the desired goal state of the controllers
        """
        self.controllers[0].set_goal(x)
        self.controllers[1].set_goal(x)

    def save_(self, save_dir):
        """
        Save controllers' parameters.

        Parameters
        ----------
        save_dir : string or path object
            directory where the parameters will be saved
        """
        self.controllers[0].save_(save_dir)
        self.controllers[1].save_(save_dir)

    def reset_(self):
        """
        Reset controllers.
        """
        self.controllers[0].reset_()
        self.controllers[1].reset_()

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
        inactive = 1 - self.active

        if self.conditions[inactive](t, x):
            self.active = 1 - self.active
            if self.verbose:
                print("Switching to Controller ", self.active + 1)

        if self.compute_both:
            _ = self.controllers[inactive].get_control_output_(x, t)

        return self.controllers[self.active].get_control_output_(x, t)

    def get_forecast(self):
        """
        Get a forecast trajectory as planned by the controller.
        Uses active controller
        """
        return self.controllers[self.active].get_forecast()

    def get_init_trajectory(self):
        """
        Get the initial (reference) trajectory as planned by the controller.
        Uses active controller
        """
        return self.controllers[self.active].get_init_trajectory()


class SimultaneousControllers(AbstractController):
    """
    Controller to combine multiple controllers and add all their outputs torques.

    Parameters
    ----------
    controllers : list
        list containint Controller objects
    forecast_con : int
        integer indicating which controller will be used for the forecast
        (Default value=0)
    """

    def __init__(self, controllers, forecast_con=0):
        super().__init__()

        self.controllers = controllers
        self.fc_ind = forecast_con

    def init_(self):
        """
        Initialize all controllers.
        """
        for c in self.controllers:
            c.init_()

    def set_parameters(self, controller_pars):
        """
        Set parameters for all controllers.

        Parameters
        ----------
        controller_pars : list
            list of lists containing the controller parameters which will be
            parsed to set_parameters
        """
        for i, c in enumerate(self.controllers):
            c.set_parameters(*(controller_pars[i]))

    def set_start(self, x):
        """
        Set start state for the controllers.

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        """
        for c in self.controllers:
            c.set_start(x)

    def set_goal(self, x):
        """
        Set the desired state for the controllers.

        Parameters
        ----------
        x : array like
            the desired goal state of the controllers
        """
        for c in self.controllers:
            c.set_goal(x)

    def get_control_output_(self, x, t):
        """
        The function to compute the control input for the double pendulum's
        actuator(s).
        Will sum the torques of all controller.

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
        u_cons = []
        for c in self.controllers:
            u_cons.append(c.get_control_output_(x, t))

        u = np.sum(u_cons)
        return u

    def get_forecast(self):
        """
        Get a forecast trajectory as planned by the controller.
        Uses controller indicated by self.fc_ind.
        """
        return self.controllers[self.fc_ind].get_forecast()

    def get_init_trajectory(self):
        """
        Get the intital (reference) trajectory as planned by the controller.
        Uses controller indicated by self.fc_ind.
        """
        return self.controllers[self.fc_ind].get_init_trajectory()
