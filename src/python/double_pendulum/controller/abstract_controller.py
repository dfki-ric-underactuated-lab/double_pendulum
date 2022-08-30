# here, the abstract controller class is defined, to which all controller
# classes have to adhere
from abc import ABC, abstractmethod
import numpy as np

from double_pendulum.model.friction_matrix import yb_friction_matrix
from double_pendulum.utils.filters.identity import identity_filter
from double_pendulum.utils.filters.low_pass import lowpass_filter_rt
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
        self.u_hist = [[0., 0.]]
        self.u_fric_hist = []

    @abstractmethod
    def get_control_output_(self, x, t=None):
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

    def get_control_output(self, x, t=None):
        self.x_hist.append(x)
        y = self.filter_measurement(x, self.u_hist[-1])
        self.x_filt_hist.append(y)

        u = np.asarray(self.get_control_output_(y, t))

        u_fric = self.get_friction_torque(y)
        self.u_fric_hist.append(u_fric)
        u += u_fric

        u_grav = self.get_gravity_torque(y)
        u += u_grav

        self.u_hist.append(u)
        return u

    def set_parameters(self):
        """
        Set controller parameters. Optional.
        """
        pass

    def init_(self):
        """
        Initialize the controller. Optional.
        """

    def init(self):
        self.init_filter()
        self.x_hist = []
        self.u_hist = [[0., 0.]]
        self.xfilt_hist = []

        self.init_()

    def reset(self):
        self.set_filter_args()
        self.set_friction_compensation()
        self.set_gravity_compensation()

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

    def set_filter_args(self, filt=None, x0=[np.pi, 0., 0., 0.], dt=0.001, plant=None,
                        simulator=None, velocity_cut=-1., filter_kwargs={}):
        self.filt = filt
        self.filt_x0 = x0
        self.filt_dt = dt
        self.filt_plant = plant
        self.filt_simulator = simulator
        self.filt_velocity_cut = velocity_cut
        self.filt_kwargs = filter_kwargs

    def init_filter(self):

        if self.filt == "lowpass":
            #dof = self.filt_plant.dof
            dof = 2

            self.filter = lowpass_filter_rt(
                    dim_x=2*dof,
                    alpha=self.filt_kwargs["lowpass_alpha"],
                    x0=self.filt_x0)

        elif self.filt == "kalman":
            dof = self.filt_plant.dof

            A, B = self.filt_plant.linear_matrices(
                    self.filt_kwargs["kalman_xlin"],
                    self.filt_kwargs["kalman_ulin"])

            self.filter = kalman_filter_rt(
                    A=A,
                    B=B,
                    dim_x=2*dof,
                    dim_u=self.filt_plant.n_actuators,
                    x0=self.filt_x0,
                    dt=self.filt_dt,
                    process_noise=self.filt_kwargs["kalman_process_noise_sigmas"],
                    measurement_noise=self.filt_kwargs["kalman_meas_noise_sigmas"])
        elif self.filt == "unscented_kalman":
            dof = self.filt_plant.dof
            if self.filt_kwargs["ukalman_integrator"] == "euler":
                fx = self.filt_simulator.euler_integrator
            elif self.filt_kwargs["ukalman_integrator"] == "runge_kutta":
                fx = self.filt_simulator.runge_integrator
            self.filter = unscented_kalman_filter_rt(
                    dim_x=2*dof,
                    x0=self.filt_x0,
                    dt=self.filt_dt,
                    process_noise=self.filt_kwargs["ukalman_process_noise_sigmas"],
                    measurement_noise=self.filt_kwargs["ukalman_meas_noise_sigmas"],
                    fx=fx)
        else:
            self.filter = identity_filter()

    def filter_measurement(self, x, last_u):
        x_filt = np.copy(x)

        # velocity cut
        if self.filt_velocity_cut > 0.:
            x_filt[2] = np.where(np.abs(x_filt[2]) < self.filt_velocity_cut, 0, x_filt[2])
            x_filt[3] = np.where(np.abs(x_filt[3]) < self.filt_velocity_cut, 0, x_filt[3])

        # filter
        x_filt = self.filter(x_filt, last_u)

        return x_filt

    def set_friction_compensation(self, damping=[0., 0.], coulomb_fric=[0., 0.]):
        self.friction_terms = np.array([coulomb_fric[0],
                                        damping[0],
                                        coulomb_fric[1],
                                        damping[1]])

    def get_friction_torque(self, x):
        friction_regressor_mat = yb_friction_matrix([x[2], x[3]])
        tau_fric = np.dot(friction_regressor_mat, self.friction_terms)
        return tau_fric

    def set_gravity_compensation(self, plant=None):
        self.grav_plant = plant

    def get_gravity_torque(self, x):
        if self.grav_plant is not None:
            g = self.grav_plant.gravity_vector(x)
            tau_grav = -np.dot(self.grav_plant.B, g)
        else:
            tau_grav = [0., 0.]
        return np.asarray(tau_grav)
