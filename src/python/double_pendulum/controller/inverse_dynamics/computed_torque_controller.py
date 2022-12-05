import os
import yaml
import numpy as np
import pandas as pd

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum


class ComputedTorqueController(AbstractController):
    """
    Computed torque controller.
    Controller which computes torque from the inverse dynamics and PID.
    Only useful for the fully actuated double pendulum.

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
        shape=(2,), dtype=float, default=[0.3, 0.3]
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
    dt : float
        timestep, unit=[s]
    csv_path : string or path obj
        path to csv file containing a reference trajectory
        (Default value=None)
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
                 dt=0.01,
                 csv_path=None):

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
        self.dt = dt

        data = pd.read_csv(csv_path)
        time_traj = np.asarray(data["time"])
        pos1_traj = np.asarray(data["pos1"])
        pos2_traj = np.asarray(data["pos2"])
        vel1_traj = np.asarray(data["vel1"])
        vel2_traj = np.asarray(data["vel2"])
        acc1_traj = np.asarray(data["acc1"])
        acc2_traj = np.asarray(data["acc2"])

        self.T = time_traj.T
        self.X = np.asarray([pos1_traj, pos2_traj,
                             vel1_traj, vel2_traj]).T
        self.ACC = np.asarray([acc1_traj, acc2_traj]).T

        # default weights
        self.Kp = 10.0
        self.Ki = 0.0
        self.Kd = 0.1

        # init pars
        self.errors1 = []
        self.errors2 = []
        self.counter = 0

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
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def init_(self):
        """
        Initialize controller.
        """
        self.errors1 = []
        self.errors2 = []
        self.counter = 0

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
        acc = self.splant.inverse_dynamics(self.X[self.counter], self.ACC[self.counter])

        p = self.X[self.counter, :2]

        e1 = p[0] - x[0]
        e2 = p[1] - x[1]
        e1 = (e1 + np.pi) % (2*np.pi) - np.pi
        e2 = (e2 + np.pi) % (2*np.pi) - np.pi
        self.errors1.append(e1)
        self.errors2.append(e2)

        P1 = self.Kp*e1
        P2 = self.Kp*e2

        I1 = self.Ki*np.sum(np.asarray(self.errors1))*self.dt
        I2 = self.Ki*np.sum(np.asarray(self.errors2))*self.dt

        if len(self.errors1) > 2:
            D1 = self.Kd*(self.errors1[-1]-self.errors1[-2]) / self.dt
            D2 = self.Kd*(self.errors2[-1]-self.errors2[-2]) / self.dt
        else:
            D1 = 0.0
            D2 = 0.0

        u1 = P1 + I1 + D1
        u2 = P2 + I2 + D2
        u = np.asarray([u1, u2])

        u[0] = np.clip(u[0], -self.torque_limit[0], self.torque_limit[0])
        u[1] = np.clip(u[1], -self.torque_limit[1], self.torque_limit[1])

        self.counter += 1
        return u

    def get_init_trajectory(self):
        """
        Get the initial (reference) trajectory used by the controller.

        Returns
        -------
        numpy_array
            time points, unit=[s]
            shape=(N,)
        numpy_array
            shape=(N, 4)
            states, units=[rad, rad, rad/s, rad/s]
            order=[angle1, angle2, velocity1, velocity2]
        None
            Does not return reference torques
        """
        return self.T, self.X, None

    def save_(self, save_dir):
        """
        Save the energy trajectory to file.

        Parameters
        ----------
        path : string or path object
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
                "Kp" : self.Kp,
                "Ki" : self.Ki,
                "Kd" : self.Kd,
        }

        with open(os.path.join(save_dir, "controller_inverse_dynamics_parameters.yml"), 'w') as f:
            yaml.dump(par_dict, f)
