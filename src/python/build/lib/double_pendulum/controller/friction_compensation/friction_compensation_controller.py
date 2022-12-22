import os
import yaml
import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum


class FrictionCompensationController(AbstractController):
    """
    Friction compensation controller.

    Parameters
    ----------
    mass : array_like, optional
        shape=(2,), dtype=float, default=[1.0, 1.0]
        masses of the double pendulum,
        [m1, m2], units=[kg]
    length : array_like, optional
        shape=(2,), dtype=float, default=[0.5, 0.5]
        link lengths of the double pendulum,
        [l1, l2], units=[m]
    com : array_like, optional
        shape=(2,), dtype=float, default=[0.5, 0.5]
        center of mass lengths of the double pendulum links
        [r1, r2], units=[m]
    damping : array_like, optional
        shape=(2,), dtype=float, default=[0.5, 0.5]
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
    """
    def __init__(self,
                 mass=[1.0, 1.0],
                 length=[0.5, 0.5],
                 com=[0.5, 0.5],
                 damping=[0.1, 0.1],
                 coulomb_fric=[0.0, 0.0],
                 gravity=9.81,
                 inertia=[None, None],
                 torque_limit=[0.0, 1.0],
                 model_pars=None):

        super().__init__()

        self.mass = mass
        self.length = length
        self.com = com
        self.damping = damping
        self.cfric = coulomb_fric
        self.gravity = gravity
        self.inertia = inertia
        self.torque_limit = torque_limit

        if model_pars is not None:
            self.mass = model_pars.m
            self.length = model_pars.l
            self.com = model_pars.r
            self.damping = model_pars.b
            self.cfric = model_pars.cf
            self.gravity = model_pars.g
            self.inertia = model_pars.I
            # self.Ir = model_pars.Ir
            # self.gr = model_pars.gr
            self.torque_limit = model_pars.tl

        self.splant = SymbolicDoublePendulum(mass=self.mass,
                                             length=self.length,
                                             com=self.com,
                                             damping=self.damping,
                                             gravity=self.gravity,
                                             coulomb_fric=self.cfric,
                                             inertia=self.inertia,
                                             torque_limit=self.torque_limit)

    def get_control_output_(self, x, t=None):
        """
        The function to compute the control input for the double pendulum's
        actuator(s).
        Compensates the actuator friction.

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

        cf = self.splant.coulomb_vector(x)
        u = np.dot(self.splant.B, cf)

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
        par_dict = {
                "mass1" : self.mass[0],
                "mass2" : self.mass[1],
                "length1" : self.length[0],
                "length2" : self.length[1],
                "com1" : self.com[0],
                "com2" : self.com[1],
                "damping1" : self.damping[0],
                "damping2" : self.damping[1],
                "cfric1" : self.cfric[0],
                "cfric2" : self.cfric[1],
                "gravity" : self.gravity,
                "inertia1" : self.inertia[0],
                "inertia2" : self.inertia[1],
                "torque_limit1" : self.torque_limit[0],
                "torque_limit2" : self.torque_limit[1],
        }

        with open(os.path.join(save_dir, "controller_friction_compensation_parameters.yml"), 'w') as f:
            yaml.dump(par_dict, f)
