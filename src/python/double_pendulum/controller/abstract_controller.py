# here, the abstract controller class is defined, to which all controller
# classes have to adhere
import os
import yaml
from abc import ABC, abstractmethod
import numpy as np

from double_pendulum.model.friction_matrix import yb_friction_matrix
from double_pendulum.filter.identity import identity_filter


class AbstractController(ABC):
    """
    Abstract controller class. All controller should inherit from
    this abstract class.
    """

    def __init__(self):
        self.set_friction_compensation()
        self.set_gravity_compensation()
        self.x_hist = []
        self.u_hist = [[0.0, 0.0]]
        self.u_fric_hist = []
        self.u_grav_hist = []
        self.filter = identity_filter(filt_velocity_cut=0.0)
        self.use_gravity_compensation = False

    @abstractmethod
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

        u = [0.0, 0.0]
        return u

    def get_control_output(self, x, t=None):
        """
        The method which is called in the Simulator and real experiment loop
        to get the control signal from the controller.
        This method does:
            - filter the state
            - comput torques by calling get_control_output_
            - compute friction compensation torque (if activated)
            - comput gravity compensation torque (if activated)

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
        self.x_hist.append(x)
        y = self.filter.get_filtered_state(x, self.u_hist[-1])

        u = np.asarray(self.get_control_output_(y, t))

        u_fric = self.get_friction_torque(y)
        self.u_fric_hist.append(u_fric)
        u += u_fric

        u_grav = self.get_gravity_torque(y)
        self.u_grav_hist.append(u_grav)
        u += u_grav

        self.u_hist.append(u)
        return u

    def set_parameters(self):
        """
        Set controller parameters. Optional.
        Can be overwritten by actual controller.
        """
        pass

    def init_(self):
        """
        Initialize the controller. Optional.
        Can be overwritten by actual controller.
        Initialize function which will always be called before using the
        controller.
        """
        pass

    def init(self):
        """
        Initialize function which will always be called before using the
        controller.
        In addition to the controller specific init_ method this method
        initialized the filter and internal logs.
        """
        self.filter.init()
        self.x_hist = []
        self.u_hist = [[0.0, 0.0]]
        self.u_fric_hist = []
        self.u_grav_hist = []

        self.init_()

    def reset_(self):
        """
        Reset the Controller. Optional
        Can be overwritten by actual controller.
        Function to reset parameters inside the controller.
        """
        pass

    def reset(self):
        """
        Reset the Controller
        Resets:
            - Filter
            - Friction compensation parameters
            - Gravity Compensation parameters
            - calls the controller specific reset_() function
        """
        # self.set_filter_args()
        self.filter.init()
        self.set_friction_compensation()
        self.set_gravity_compensation()
        self.reset_()

    def save(self, save_dir):
        """
        Save controller parameters

        Parameters
        ----------
        save_dir : string or path object
            directory where the parameters will be saved
        """
        self.save_(save_dir)
        # self.filter.save(save_dir)

        if self.grav_plant is not None:
            g = self.grav_plant.g
        else:
            g = 0.0

        par_dict = {
            "coulomb_fric1": float(self.friction_terms[0]),
            "coulomb_fric2": float(self.friction_terms[2]),
            "damping1": float(self.friction_terms[1]),
            "damping2": float(self.friction_terms[3]),
            "gravity_compensation_g": g,
        }
        with open(
            os.path.join(save_dir, "controller_abstract_parameters.yml"), "w"
        ) as f:
            yaml.dump(par_dict, f)

    def save_(self, save_dir):
        """
        Save controller parameters. Optional
        Can be overwritten by actual controller.

        Parameters
        ----------
        save_dir : string or path object
            directory where the parameters will be saved
        """
        pass

    def set_start(self, x):
        """
        Set the start state for the controller. Optional.
        Can be overwritten by actual controller.

        Parameters
        ----------
        x0 : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        """
        self.x0 = x

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

    def set_filter(self, filter):
        self.filter = filter

    def get_forecast(self):
        """
        Get a forecast trajectory as planned by the controller. Optional.
        Can be overwritten by actual controller.

        Returns
        -------
        list
            placeholder
        list
            placeholder
        list
            placeholder
        """
        return [], [], []

    def get_init_trajectory(self):
        """
        Get an initial (reference) trajectory used by the controller. Optional.
        Can be overwritten by actual controller.

        Returns
        -------
        list
            placeholder
        list
            placeholder
        list
            placeholder
        """
        return [], [], []

    def set_friction_compensation(self, damping=[0.0, 0.0], coulomb_fric=[0.0, 0.0]):
        """
        Set friction terms used for the friction compensation.

        Parameters
        ----------
        damping : array_like, optional
            shape=(2,), dtype=float, default=[0.5, 0.5]
            damping coefficients of the double pendulum actuators
            [b1, b2], units=[kg*m/s]
        coulomb_fric : array_like, optional
            shape=(2,), dtype=float, default=[0.0, 0.0]
            coulomb friction coefficients for the double pendulum actuators
            [cf1, cf2], units=[Nm]
        """
        self.friction_terms = np.array(
            [coulomb_fric[0], damping[0], coulomb_fric[1], damping[1]]
        )

    def get_friction_torque(self, x):
        """
        Get the torque needed to compensate for friction.

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        Returns
        -------
        numpy_array
            shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
        """
        friction_regressor_mat = yb_friction_matrix([x[2], x[3]])
        tau_fric = np.dot(friction_regressor_mat, self.friction_terms)
        return tau_fric

    def set_gravity_compensation(self, plant=None):
        """
        Provide plant for gravity compensation.

        Parameters
        ----------
        plant : SymbolicDoublePendulum or DoublePendulumPlant object
            A plant object containing the kinematics and dynamics of the
            double pendulum
        """
        self.grav_plant = plant

    def get_gravity_torque(self, x):
        """
        Get the torque needed to compensate for gravity.

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        Returns
        -------
        numpy_array
            shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
        """
        # if self.grav_plant is not None:
        if self.use_gravity_compensation:
            g = self.grav_plant.gravity_vector(x)
            tau_grav = -np.dot(self.grav_plant.B, g)
        else:
            tau_grav = [0.0, 0.0]
        return np.asarray(tau_grav)
