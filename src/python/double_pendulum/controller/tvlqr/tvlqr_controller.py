import os
import yaml
import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.controller.lqr.lqr import lqr, iterative_riccati
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.utils.csv_trajectory import load_trajectory, save_trajectory
from double_pendulum.utils.wrap_angles import wrap_angles_diff
from double_pendulum.utils.pcw_polynomial import InterpolateVector, InterpolateMatrix


class TVLQRController(AbstractController):
    """TVLQRController
    Controller to stabilize a trajectory with TVLQR

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
    csv_path : string or path object
        path to csv file where the trajectory is stored.
        csv file should use standarf formatting used in this repo.
        If T, X, or U are provided they are preferred.
        (Default value="")
    num_break : int
        number of break points used for interpolation
        (Default value = 40)
        (Default value=100)
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
                 csv_path="",
                 num_break=40,
                 ):

        super().__init__()

        # model parameters
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
            self.Ir = model_pars.Ir
            self.gr = model_pars.gr
            self.torque_limit = model_pars.tl

        self.splant = SymbolicDoublePendulum(
                mass=self.mass,
                length=self.length,
                com=self.com,
                damping=self.damping,
                gravity=self.gravity,
                coulomb_fric=self.cfric,
                inertia=self.inertia,
                torque_limit=self.torque_limit)

        self.num_break = num_break

        # load trajectory
        self.T, self.X, self.U = load_trajectory(csv_path=csv_path,
                                                 with_tau=True)
        self.max_t = self.T[-1]
        self.dt = self.T[1] - self.T[0]

        # interpolate trajectory
        self.X_interp = InterpolateVector(
                T=self.T,
                X=self.X,
                num_break=num_break,
                poly_degree=3)

        self.U_interp = InterpolateVector(
                T=self.T,
                X=self.U,
                num_break=num_break,
                poly_degree=3)

        # default parameters
        self.Q = np.diag([4., 4., 0.1, 0.1])
        self.R = 2*np.eye(1)
        self.Qf = np.diag([4., 4., 0.1, 0.1])
        self.goal = np.array([np.pi, 0., 0., 0.])

        # initializations
        self.K = []
        # self.k = []

    def set_cost_parameters(self,
                            Q=np.diag([4., 4., 0.1, 0.1]),
                            R=2*np.eye(1),
                            Qf=np.diag([4., 4., 0.1, 0.1])):
        """set_cost_parameters
        Set the cost matrices Q, R and Qf.
        (Qf for the final stabilization)

        Parameters
        ----------
        Q : numpy_array
            shape=(4,4)
            Q-matrix describing quadratic state cost
            (Default value=np.diag([4., 4., 0.1, 0.1]))
        R : numpy_array
            shape=(2,2)
            R-matrix describing quadratic control cost
            (Default value=2*np.eye(1))
        Qf : numpy_array
            shape=(4,4)
            Q-matrix describing quadratic state cost
            for the final point stabilization
            (Default value=np.diag([4., 4., 0.1, 0.1]))
        """
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)
        self.Qf = np.asarray(Qf)

    def set_goal(self, x=[np.pi, 0., 0., 0.]):
        """set_goal.
        Set goal for the controller.

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
            (Default value=[np.pi, 0., 0., 0.])
        """
        y = x.copy()
        y[0] = y[0] % (2*np.pi)
        y[1] = (y[1] + np.pi) % (2*np.pi) - np.pi
        self.goal = np.asarray(y)

    def init_(self):
        """
        Initalize the controller.
        """

        self.K, _ = iterative_riccati(
            self.splant, self.Q, self.R, self.Qf, self.dt, self.X, self.U)

        self.K_interp = InterpolateMatrix(
            T=self.T,
            X=self.K,
            num_break=self.num_break,
            poly_degree=3)

    def get_control_output_(self, x, t):
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

        tt = min(t, self.max_t)
        x_error = wrap_angles_diff(np.asarray(x) - self.X_interp.get_value(tt))

        tau = self.U_interp.get_value(tt) - np.dot(self.K_interp.get_value(tt), x_error)
        u = [tau[0], tau[1]]

        u[0] = np.clip(u[0], -self.torque_limit[0], self.torque_limit[0])
        u[1] = np.clip(u[1], -self.torque_limit[1], self.torque_limit[1])
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
        numpy_array
            shape=(N, 2)
            actuations/motor torques
            order=[u1, u2],
            units=[Nm]
        """
        return self.T, self.X, self.U

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
                #"Ir" : self.Ir,
                #"gr" : self.gr,
                "torque_limit1" : self.torque_limit[0],
                "torque_limit2" : self.torque_limit[1],
                "num_break" : self.num_break,
                "goal1" : float(self.goal[0]),
                "goal2" : float(self.goal[1]),
                "goal3" : float(self.goal[2]),
                "goal4" : float(self.goal[3]),
                "dt" : float(self.dt),
                "max_t" : float(self.max_t),
        }

        with open(os.path.join(save_dir, "controller_tvlqr_parameters.yml"), 'w') as f:
            yaml.dump(par_dict, f)

        np.savetxt(os.path.join(save_dir, "controller_tvlqr_Qmatrix.txt"), self.Q)
        np.savetxt(os.path.join(save_dir, "controller_tvlqr_Rmatrix.txt"), self.R)
        np.savetxt(os.path.join(save_dir, "controller_tvlqr_Qfmatrix.txt"), self.Qf)

        save_trajectory(os.path.join(save_dir, "controller_tvlqr_initial_traj.csv"), self.T, self.X, self.U)
