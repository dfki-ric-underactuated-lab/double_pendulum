# here, the abstract controller class is defined, to which all controller
# classes have to adhere
from abc import ABC, abstractmethod


class AbstractController(ABC):
    """
    Abstract controller class. All controller should inherit from
    this abstract class.
    """
    @abstractmethod
    def get_control_output(self, x, t=None):
        """
        The function to compute the control input for the double pendulum's
        actuator(s).
        Supposed to be overwritten by actual controllers. The API of this
        method should be adapted. Unused inputs/outputs can be set to None.

        Parameters
        ----------
        x : array-like
            state array for double pendulum system with np.shape(x)=4
        t : float, optional
            time in seconds
        Returns
        -------
        u : array-like
            the torque supposed to be applied by the actuators [Nm]
            np.shape(u)=2.
            For acrobot/pendubot system set the unactuated motor command to 0
        """

        u = [0.0, 0.0]
        return u

    def set_parameters(self):
        """
        Set controller parameters. Optional.
        """
        pass

    def init(self):
        """
        Initialize the controller. Optional.
        """
        pass

    def set_start(self, x):
        """
        Set the desired state for the controller. Optional.

        Parameters
        ----------
        x0 : array like
            the start state of the double pendulum
        """
        self.x0 = x

    def set_goal(self, x):
        """
        Set the desired state for the controller. Optional.

        Parameters
        ----------
        x : array like
            the desired goal state of the controller
        """

        self.goal = x

    def get_forecast(self):
        return [], [], []

    def get_init_trajectory(self):
        return [], [], []
