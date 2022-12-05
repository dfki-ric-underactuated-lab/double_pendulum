import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.controller.gravity_compensation.gravity_compensation_controller import GravityCompensationController
from double_pendulum.controller.pid.point_pid_controller import PointPIDController


class PIDGravityCompensationController(AbstractController):
    """PIDGravityCompensationController
    Controller to compensate the gravitational force and
    apply a PID controller on top.

    Parameters
    ----------
    mass : array_like, optional
        shape=(2,), dtype=float, default=[0.5, 0.6]
        masses of the double pendulum,
        [m1, m2], units=[kg]
    length : array_like, optional
        shape=(2,), dtype=float, default=[0.3, 0.2]
        link lengths of the double pendulum,
        [l1, l2], units=[m]
    com : array_like, optional
        shape=(2,), dtype=float, default=[0.3, 0.2]
        center of mass lengths of the double pendulum links
        [r1, r2], units=[m]
    damping : array_like, optional
        shape=(2,), dtype=float, default=[0.1, 0.1]
        damping coefficients of the double pendulum actuators
        [b1, b2], units=[kg*m/s]
    gravity : float, optional
        default=9.81
        gravity acceleration (pointing downwards),
        units=[m/s²]
    coulomb_fric : array_like, optional
        shape=(2,), dtype=float, default=[0.0, 0.0]
        coulomb friction coefficients for the double pendulum actuators
        [cf1, cf2], units=[Nm]
    inertia : array_like, optional
        shape=(2,), dtype=float, default=[None, None]
        inertia of the double pendulum links
        [I1, I2], units=[kg*m²]
        if entry is None defaults to point mass m*l² inertia for the entry
    torque_limit : array_like, optional
        shape=(2,), dtype=float, default=[0.0, 1.0]
        torque limit of the motors
        [tl1, tl2], units=[Nm, Nm]
    model_pars : model_parameters object, optional
        object of the model_parameters class, default=None
        Can be used to set all model parameters above
        If provided, the model_pars parameters overwrite
        the other provided parameters
        (Default value=None)
    dt : float
        timestep, unit=[s]
        (Default value=0.01)
    """
    def __init__(self,
                 mass=[0.5, 0.6],
                 length=[0.3, 0.2],
                 com=[0.3, 0.2],
                 damping=[0.1, 0.1],
                 coulomb_fric=[0.0, 0.0],
                 gravity=9.81,
                 inertia=[None, None],
                 torque_limit=[0.0, 1.0],
                 model_pars=None,
                 dt=0.01):

        super().__init__()

        self.torque_limit = torque_limit
        if model_pars is not None:
            self.torque_limit = model_pars.tl

        self.grav_con = GravityCompensationController(
                mass=mass,
                length=length,
                com=com,
                damping=damping,
                coulomb_fric=coulomb_fric,
                gravity=gravity,
                inertia=inertia,
                torque_limit=torque_limit,
                model_pars=model_pars)

        self.pid_con = PointPIDController(
                torque_limit=self.torque_limit,
                dt=dt)


    def set_goal(self, x):
        """
        Set the desired state for the controller.

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        """
        self.pid_con.set_goal(x)

    def set_parameters(self, Kp, Ki, Kd):
        """
        Set PID parameters.

        Parameters
        ----------
        Kp : float
            Gain proportional to position error
        Ki : float
            Gain proportional to integrated error
        Kd : float
            Gain proportional to error derivative
        """
        self.pid_con.set_parameters(Kp, Ki, Kd)

    def init_(self):
        """
        Initialize controller.
        Initalizes Gravity compensation and PID controller.
        """
        self.grav_con.init()
        self.pid_con.init()

    def get_control_output_(self, x, t=None):
        """
        The function to compute the control input for the double pendulum's
        actuator(s).

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

        grav_u = self.grav_con.get_control_output(x, t)
        pid_u = self.pid_con.get_control_output(x, t)

        u = grav_u + pid_u

        u[0] = np.clip(u[0], -self.torque_limit[0], self.torque_limit[0])
        u[1] = np.clip(u[1], -self.torque_limit[1], self.torque_limit[1])

        return u

    def save_(self, save_dir):
        """
        Save controller parameters

        Parameters
        ----------
        save_dir : string or path object
            directory where the parameters will be saved
        """

        self.grav_con.save_(save_dir)
        self.pid_con.save_(save_dir)
