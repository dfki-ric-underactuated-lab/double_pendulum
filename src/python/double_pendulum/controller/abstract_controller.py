# here, the abstract controller class is defined, to which all controller
# classes have to adhere
import os
import yaml
from abc import ABC, abstractmethod
import numpy as np

from double_pendulum.model.friction_matrix import yb_friction_matrix
from double_pendulum.utils.filters.identity import identity_filter
from double_pendulum.utils.filters.low_pass import lowpass_filter_rt, butter_filter_rt
from double_pendulum.utils.filters.kalman_filter import kalman_filter_rt
from double_pendulum.utils.filters.unscented_kalman_filter import unscented_kalman_filter_rt


class AbstractController(ABC):
    """
    Abstract controller class. All controller should inherit from
    this abstract class.
    """

    def __init__(self):
        self.set_filter_args()
        self.set_friction_compensation()
        self.set_gravity_compensation()
        self.x_hist = []
        self.x_filt_hist = []
        self.u_hist = [[0.0, 0.0]]
        self.u_fric_hist = []
        self.u_grav_hist = []

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
        y = self.filter_measurement(x, self.u_hist[-1])
        self.x_filt_hist.append(y)

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
        self.init_filter()
        self.x_hist = []
        self.u_hist = [[0.0, 0.0]]
        self.u_fric_hist = []
        self.u_grav_hist = []
        self.xfilt_hist = []

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
        self.set_filter_args()
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

        if self.grav_plant is not None:
            g = self.grav_plant.g
        else:
            g = 0.0

        par_dict = {
            "filt": self.filt,
            "filt_x0": self.filt_x0,
            "filt_dt": self.filt_dt,
            "filt_velocity_cut": self.filt_velocity_cut,
            "filt_kwargs": self.filt_kwargs,
            "coulomb_fric1": float(self.friction_terms[0]),
            "coulomb_fric2": float(self.friction_terms[2]),
            "damping1": float(self.friction_terms[1]),
            "damping2": float(self.friction_terms[3]),
            "gravity_compensation_g": g,
        }
        with open(os.path.join(save_dir, "controller_abstract_parameters.yml"), "w") as f:
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

    def set_filter_args(
        self,
        filt=None,
        x0=[np.pi, 0.0, 0.0, 0.0],
        dt=0.001,
        plant=None,
        simulator=None,
        velocity_cut=-1.0,
        filter_kwargs={},
    ):
        """
        Set filter arguments for the measurement filter.

        Parameters
        ----------
        filt : string
            string determining the velocity noise filter
            "None": No filter
            "lowpass": lowpass filter
            "kalman": kalman filter
            "unscented_kalman": unscented kalman filter
            (Default value = None)
        x0 : array_like, shape=(4,), dtype=float,
            reference state if a linearization is needed (Kalman filter),
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
            (Default value=[np.pi, 0., 0., 0.])
        dt : float
            timestep, unit=[s]
            (Default value=0.001)
        plant : SymbolicDoublePendulum or DoublePendulumPlant object
            A plant object containing the kinematics and dynamics of the
            double pendulum
            (Default value=None)
        simulator : Simulator object
            simulator object necessary for the unscented Kalman filter
            (Default value=None)
        velocity_cut : float
            measurements smaller than this value will be set to 0.
            (they are assumed to be noise)
            For meas_noise_cut<=0.0, the measurement is not cut
            (Default value = -1.)
        filter_kwargs : dict
            dictionary containing parameters for the velocity filter
            (Default value = {})
        """
        self.filt = filt
        self.filt_x0 = x0
        self.filt_dt = dt
        self.filt_plant = plant
        self.filt_simulator = simulator
        self.filt_velocity_cut = velocity_cut
        self.filt_kwargs = filter_kwargs

    def init_filter(self):
        """
        Initialize the measurement filter
        """
        if self.filt == "butter":
            dof = 2

            self.filter = butter_filter_rt(dof=dof, cutoff=self.filt_kwargs["butter_cutoff"], x0=self.filt_x0,
                                           dt=self.filt_kwargs['dt'])

        elif self.filt == "lowpass":
            # dof = self.filt_plant.dof
            dof = 2

            self.filter = lowpass_filter_rt(dim_x=2 * dof, alpha=self.filt_kwargs["lowpass_alpha"], x0=self.filt_x0)

        elif self.filt == "kalman":
            dof = self.filt_plant.dof

            A, B = self.filt_plant.linear_matrices(self.filt_kwargs["kalman_xlin"], self.filt_kwargs["kalman_ulin"])

            self.filter = kalman_filter_rt(
                A=A,
                B=B,
                dim_x=2 * dof,
                dim_u=self.filt_plant.n_actuators,
                x0=self.filt_x0,
                dt=self.filt_dt,
                process_noise=self.filt_kwargs["kalman_process_noise_sigmas"],
                measurement_noise=self.filt_kwargs["kalman_meas_noise_sigmas"],
            )
        elif self.filt == "unscented_kalman":
            dof = self.filt_plant.dof
            if self.filt_kwargs["ukalman_integrator"] == "euler":
                fx = self.filt_simulator.euler_integrator
            elif self.filt_kwargs["ukalman_integrator"] == "runge_kutta":
                fx = self.filt_simulator.runge_integrator
            self.filter = unscented_kalman_filter_rt(
                dim_x=2 * dof,
                x0=self.filt_x0,
                dt=self.filt_dt,
                process_noise=self.filt_kwargs["ukalman_process_noise_sigmas"],
                measurement_noise=self.filt_kwargs["ukalman_meas_noise_sigmas"],
                fx=fx,
            )
        else:
            self.filter = identity_filter()

    def filter_measurement(self, x, last_u):
        """
        filter_measurement

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        last_u : array_like, shape=(2,), dtype=float
            desired actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]

        Returns
        -------
        numpy_array
            shape=(4,), dtype=float,
            filters state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        """
        x_filt = np.copy(x)

        # velocity cut
        if self.filt_velocity_cut > 0.0:
            x_filt[2] = np.where(np.abs(x_filt[2]) < self.filt_velocity_cut, 0, x_filt[2])
            x_filt[3] = np.where(np.abs(x_filt[3]) < self.filt_velocity_cut, 0, x_filt[3])

        # filter
        x_filt = self.filter(x_filt, last_u)

        return x_filt

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
        self.friction_terms = np.array([coulomb_fric[0], damping[0], coulomb_fric[1], damping[1]])

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
        if self.grav_plant is not None:
            g = self.grav_plant.gravity_vector(x)
            tau_grav = -np.dot(self.grav_plant.B, g)
        else:
            tau_grav = [0.0, 0.0]
        return np.asarray(tau_grav)
