import os
import yaml
import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.controller.lqr.lqr import lqr
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.plant import DoublePendulumPlant


class LQRController(AbstractController):
    """
    LQRController.
    Controller which uses LQR to stabilize a (unstable) fixpoint.

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
    """

    def __init__(
        self,
        mass=[0.5, 0.6],
        length=[0.3, 0.2],
        com=[0.3, 0.2],
        damping=[0.1, 0.1],
        coulomb_fric=[0.0, 0.0],
        gravity=9.81,
        inertia=[None, None],
        torque_limit=[0.0, 1.0],
        model_pars=None,
    ):
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

        self.splant = SymbolicDoublePendulum(
            mass=self.mass,
            length=self.length,
            com=self.com,
            damping=self.damping,
            gravity=self.gravity,
            coulomb_fric=self.cfric,
            inertia=self.inertia,
            torque_limit=self.torque_limit,
        )

        # set default parameters
        self.set_goal()
        self.set_parameters()
        self.set_cost_parameters()

    def set_goal(self, x=[np.pi, 0.0, 0.0, 0.0]):
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
        y[0] = y[0] % (2 * np.pi)
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
        self.xd = np.asarray(y)

    def set_parameters(self, failure_value=np.nan, cost_to_go_cut=15.0):
        """set_parameters.
        Set parameters for this controller.

        Parameters
        ----------
        failure_value : float
            if the cost-to-go exceeds cost_to_go_cut this value is retured as torque
            (Default value=np.nan)
        cost_to_go_cut : float
            if the cost-to-go exceeds this values the controller
            returns failure_value
            (Default value=15.)
        """
        self.failure_value = failure_value
        self.cost_to_go_cut = cost_to_go_cut

    def set_cost_parameters(
        self,
        p1p1_cost=1.0,  # 1000., 0.001
        p2p2_cost=1.0,  # 1000., 0.001
        v1v1_cost=1.0,  # 1000.
        v2v2_cost=1.0,  # 1000.
        p1p2_cost=0.0,  # -500
        v1v2_cost=0.0,  # -500
        p1v1_cost=0.0,
        p1v2_cost=0.0,
        p2v1_cost=0.0,
        p2v2_cost=0.0,
        u1u1_cost=0.01,  # 100., 0.01
        u2u2_cost=0.01,  # 100., 0.01
        u1u2_cost=0.0,
    ):
        """set_cost_parameters.
        Parameters of Q and R matrices. The parameters are

        Q = ((p1p1, p1p2, p1v1, p1v2),
             (p1p2, p2p2, p2v1, p2v2),
             (p1v1, p2v1, v1v1, v1v2),
             (p1v2, p2v2, v1v2, v2v2))
        R = ((u1u1, u1u2),
             (u1u2, u2u2))

        Parameters
        ----------
        p1p1_cost : float
            p1p1_cost
            (Default value=1.)
        p2p2_cost : float
            p2p2_cost
            (Default value=1.)
        v1v1_cost : float
            v1v1_cost
            (Default value=1.)
        v2v2_cost : float
            v2v2_cost
            (Default value=0.)
        p1p2_cost : float
            p1p2_cost
            (Default value=0.)
        v1v2_cost : float
            v1v2_cost
            (Default value=0.)
        p1v1_cost : float
            p1v1_cost
            (Default value=0.)
        p1v2_cost : float
            p1v2_cost
            (Default value=0.)
        p2v1_cost : float
            p2v1_cost
            (Default value=0.)
        p2v2_cost : float
            p2v2_cost
            (Default value=0.)
        u1u1_cost : float
            u1u1_cost
            (Default value=0.01)
        u2u2_cost : float
            u2u2_cost
            (Default value=0.01)
        u1u2_cost : float
            u1u2_cost
            (Default value=0.)
        """
        # state cost matrix
        self.Q = np.array(
            [
                [p1p1_cost, p1p2_cost, p1v1_cost, p1v2_cost],
                [p1p2_cost, p2p2_cost, p2v1_cost, p2v2_cost],
                [p1v1_cost, p2v1_cost, v1v1_cost, v1v2_cost],
                [p1v2_cost, p2v2_cost, v1v2_cost, v2v2_cost],
            ]
        )

        # control cost matrix
        self.R = np.array([[u1u1_cost, u1u2_cost], [u1u2_cost, u2u2_cost]])
        # self.R = np.array([[u2u2_cost]])

    # def set_cost_parameters_(self,
    #                          pars=[1., 1., 1., 1.,
    #                                0., 0., 0., 0., 0., 0.,
    #                                0.01, 0.01, 0.]):
    #     self.set_cost_parameters(p1p1_cost=pars[0],
    #                              p2p2_cost=pars[1],
    #                              v1v1_cost=pars[2],
    #                              v2v2_cost=pars[3],
    #                              p1v1_cost=pars[4],
    #                              p1v2_cost=pars[5],
    #                              p2v1_cost=pars[6],
    #                              p2v2_cost=pars[7],
    #                              u1u1_cost=pars[8],
    #                              u2u2_cost=pars[9],
    #                              u1u2_cost=pars[10])

    def set_cost_parameters_(self, pars=[1.0, 1.0, 1.0, 1.0, 1.0]):
        """
        Set the diagonal parameters of Q and R matrices with a list.

        The parameters are
        Q = ((pars[0], 0, 0, 0),
             (0, pars[1], 0, 0),
             (0, 0, pars[2], 0),
             (0, 0, 0, pars[3]))
        R = ((pars[4], 0)
             (0, pars[4]))


        Parameters
        ----------
        pars : list
            shape=(5,), dtype=float
            (Default value=[1., 1., 1., 1., 1.])
        """
        self.set_cost_parameters(
            p1p1_cost=pars[0],
            p2p2_cost=pars[1],
            v1v1_cost=pars[2],
            v2v2_cost=pars[3],
            p1v1_cost=0.0,
            p1v2_cost=0.0,
            p2v1_cost=0.0,
            p2v2_cost=0.0,
            u1u1_cost=pars[4],
            u2u2_cost=pars[4],
            u1u2_cost=0.0,
        )

    def set_cost_matrices(self, Q, R):
        """
        Set the Q and R matrices directly.

        Parameters
        ----------
        Q : numpy_array
            shape=(4,4)
            Q-matrix describing quadratic state cost
        R : numpy_array
            shape=(2,2)
            R-matrix describing quadratic control cost
        """

        self.Q = np.asarray(Q)
        self.R = np.asarray(R)

    def init_(self):
        """
        Initalize the controller.
        """
        Alin, Blin = self.splant.linear_matrices(x0=self.xd, u0=[0.0, 0.0])
        self.K, self.S, _ = lqr(Alin, Blin, self.Q, self.R)

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
        y = x.copy().astype(float)

        # y[0] = y[0] % (2*np.pi)
        y[0] = (y[0] + np.pi - self.xd[0]) % (2 * np.pi) - (np.pi - self.xd[0])
        y[1] = (y[1] + np.pi - self.xd[1]) % (2 * np.pi) - (np.pi - self.xd[1])

        y -= self.xd

        u = -self.K.dot(y)
        u = np.asarray(u)[0]

        if y.dot(np.asarray(self.S.dot(y))[0]) > self.cost_to_go_cut:  # old value:0.1
            u = [self.failure_value, self.failure_value]

        u[0] = np.clip(u[0], -self.torque_limit[0], self.torque_limit[0])
        u[1] = np.clip(u[1], -self.torque_limit[1], self.torque_limit[1])

        # print(x, u)
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
            "mass1": self.mass[0],
            "mass2": self.mass[1],
            "length1": self.length[0],
            "length2": self.length[1],
            "com1": self.com[0],
            "com2": self.com[1],
            "damping1": self.damping[0],
            "damping2": self.damping[1],
            "cfric1": self.cfric[0],
            "cfric2": self.cfric[1],
            "gravity": self.gravity,
            "inertia1": self.inertia[0],
            "inertia2": self.inertia[1],
            # "Ir" : self.Ir,
            # "gr" : self.gr,
            "torque_limit1": self.torque_limit[0],
            "torque_limit2": self.torque_limit[1],
            "xd1": self.xd[0],
            "xd2": self.xd[1],
            "xd3": self.xd[2],
            "xd4": self.xd[3],
        }

        with open(os.path.join(save_dir, "controller_lqr_parameters.yml"), "w") as f:
            yaml.dump(par_dict, f)

        np.savetxt(os.path.join(save_dir, "controller_lqr_Qmatrix.txt"), self.Q)
        np.savetxt(os.path.join(save_dir, "controller_lqr_Rmatrix.txt"), self.R)
        np.savetxt(os.path.join(save_dir, "controller_lqr_Kmatrix.txt"), self.K)
        np.savetxt(os.path.join(save_dir, "controller_lqr_Smatrix.txt"), self.S)


class LQRController_nonsymbolic(AbstractController):
    """
    LQRController.
    Controller which uses LQR to stabilize a (unstable) fixpoint.  This version
    of the LQR controller does not use the symbolic plant and thus is
    compatible with the cma-es optimizer for parameter optimization.

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
    """

    def __init__(
        self,
        mass=[0.5, 0.6],
        length=[0.3, 0.2],
        com=[0.3, 0.2],
        damping=[0.1, 0.1],
        coulomb_fric=[0.0, 0.0],
        gravity=9.81,
        inertia=[None, None],
        torque_limit=[0.0, 1.0],
        model_pars=None,
    ):
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

        self.plant = DoublePendulumPlant(
            mass=self.mass,
            length=self.length,
            com=self.com,
            damping=self.damping,
            gravity=self.gravity,
            coulomb_fric=self.cfric,
            inertia=self.inertia,
            torque_limit=self.torque_limit,
        )

        # set default parameters
        self.set_goal()
        self.set_parameters()
        self.set_cost_parameters()

    def set_goal(self, x=[np.pi, 0.0, 0.0, 0.0]):
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
        y[0] = y[0] % (2 * np.pi)
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
        self.xd = np.asarray(y)

    def set_parameters(self, failure_value=np.nan, cost_to_go_cut=15.0):
        """set_parameters.
        Set parameters for this controller.

        Parameters
        ----------
        failure_value : float
            if the cost-to-go exceeds cost_to_go_cut this value is retured as torque
            (Default value=np.nan)
        cost_to_go_cut : float
            if the cost-to-go exceeds this values the controller
            returns failure_value
            (Default value=15.)
        """
        self.failure_value = failure_value
        self.cost_to_go_cut = cost_to_go_cut

    def set_cost_parameters(
        self,
        p1p1_cost=1.0,  # 1000., 0.001
        p2p2_cost=1.0,  # 1000., 0.001
        v1v1_cost=1.0,  # 1000.
        v2v2_cost=1.0,  # 1000.
        p1p2_cost=0.0,  # -500
        v1v2_cost=0.0,  # -500
        p1v1_cost=0.0,
        p1v2_cost=0.0,
        p2v1_cost=0.0,
        p2v2_cost=0.0,
        u1u1_cost=0.01,  # 100., 0.01
        u2u2_cost=0.01,  # 100., 0.01
        u1u2_cost=0.0,
    ):
        """set_cost_parameters.
        Parameters of Q and R matrices. The parameters are

        Q = ((p1p1, p1p2, p1v1, p1v2),
             (p1p2, p2p2, p2v1, p2v2),
             (p1v1, p2v1, v1v1, v1v2),
             (p1v2, p2v2, v1v2, v2v2))
        R = ((u1u1, u1u2),
             (u1u2, u2u2))

        Parameters
        ----------
        p1p1_cost : float
            p1p1_cost
            (Default value=1.)
        p2p2_cost : float
            p2p2_cost
            (Default value=1.)
        v1v1_cost : float
            v1v1_cost
            (Default value=1.)
        v2v2_cost : float
            v2v2_cost
            (Default value=0.)
        p1p2_cost : float
            p1p2_cost
            (Default value=0.)
        v1v2_cost : float
            v1v2_cost
            (Default value=0.)
        p1v1_cost : float
            p1v1_cost
            (Default value=0.)
        p1v2_cost : float
            p1v2_cost
            (Default value=0.)
        p2v1_cost : float
            p2v1_cost
            (Default value=0.)
        p2v2_cost : float
            p2v2_cost
            (Default value=0.)
        u1u1_cost : float
            u1u1_cost
            (Default value=0.01)
        u2u2_cost : float
            u2u2_cost
            (Default value=0.01)
        u1u2_cost : float
            u1u2_cost
            (Default value=0.)
        """
        # state cost matrix
        self.Q = np.array(
            [
                [p1p1_cost, p1p2_cost, p1v1_cost, p1v2_cost],
                [p1p2_cost, p2p2_cost, p2v1_cost, p2v2_cost],
                [p1v1_cost, p2v1_cost, v1v1_cost, v1v2_cost],
                [p1v2_cost, p2v2_cost, v1v2_cost, v2v2_cost],
            ]
        )

        # control cost matrix
        self.R = np.array([[u1u1_cost, u1u2_cost], [u1u2_cost, u2u2_cost]])
        # self.R = np.array([[u2u2_cost]])

    # def set_cost_parameters_(self,
    #                          pars=[1., 1., 1., 1.,
    #                                0., 0., 0., 0., 0., 0.,
    #                                0.01, 0.01, 0.]):
    #     self.set_cost_parameters(p1p1_cost=pars[0],
    #                              p2p2_cost=pars[1],
    #                              v1v1_cost=pars[2],
    #                              v2v2_cost=pars[3],
    #                              p1v1_cost=pars[4],
    #                              p1v2_cost=pars[5],
    #                              p2v1_cost=pars[6],
    #                              p2v2_cost=pars[7],
    #                              u1u1_cost=pars[8],
    #                              u2u2_cost=pars[9],
    #                              u1u2_cost=pars[10])

    def set_cost_parameters_(self, pars=[1.0, 1.0, 1.0, 1.0, 1.0]):
        """
        Set the diagonal parameters of Q and R matrices with a list.

        The parameters are
        Q = ((pars[0], 0, 0, 0),
             (0, pars[1], 0, 0),
             (0, 0, pars[2], 0),
             (0, 0, 0, pars[3]))
        R = ((pars[4], 0)
             (0, pars[4]))


        Parameters
        ----------
        pars : list
            shape=(5,), dtype=float
            (Default value=[1., 1., 1., 1., 1.])
        """
        self.set_cost_parameters(
            p1p1_cost=pars[0],
            p2p2_cost=pars[1],
            v1v1_cost=pars[2],
            v2v2_cost=pars[3],
            p1v1_cost=0.0,
            p1v2_cost=0.0,
            p2v1_cost=0.0,
            p2v2_cost=0.0,
            u1u1_cost=pars[4],
            u2u2_cost=pars[4],
            u1u2_cost=0.0,
        )

    def set_cost_matrices(self, Q, R):
        """
        Set the Q and R matrices directly.

        Parameters
        ----------
        Q : numpy_array
            shape=(4,4)
            Q-matrix describing quadratic state cost
        R : numpy_array
            shape=(2,2)
            R-matrix describing quadratic control cost
        """
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)

    def init_(self):
        """
        Initalize the controller.
        """
        Alin, Blin = self.plant.linear_matrices(x0=self.xd, u0=[0.0, 0.0])
        self.K, self.S, _ = lqr(Alin, Blin, self.Q, self.R)

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
        y = x.copy()
        y[0] = y[0] % (2 * np.pi)
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi

        y -= self.xd

        u = -self.K.dot(y)
        u = np.asarray(u)[0]

        if y.dot(np.asarray(self.S.dot(y))[0]) > self.cost_to_go_cut:  # old value:0.1
            u = [self.failure_value, self.failure_value]

        u[0] = np.clip(u[0], -self.torque_limit[0], self.torque_limit[0])
        u[1] = np.clip(u[1], -self.torque_limit[1], self.torque_limit[1])

        return u

    def save_(self, save_dir):
        """
        Save the energy trajectory to file.

        Parameters
        ----------
        path : string or path object
            directory where the parameters will be saved
        """

        par_dict = {
            "mass1": self.mass[0],
            "mass2": self.mass[1],
            "length1": self.length[0],
            "length2": self.length[1],
            "com1": self.com[0],
            "com2": self.com[1],
            "damping1": self.damping[0],
            "damping2": self.damping[1],
            "cfric1": self.cfric[0],
            "cfric2": self.cfric[1],
            "gravity": self.gravity,
            "inertia1": self.inertia[0],
            "inertia2": self.inertia[1],
            # "Ir" : self.Ir,
            # "gr" : self.gr,
            "torque_limit1": self.torque_limit[0],
            "torque_limit2": self.torque_limit[1],
            "xd1": float(self.xd[0]),
            "xd2": float(self.xd[1]),
            "xd3": float(self.xd[2]),
            "xd4": float(self.xd[3]),
        }

        with open(os.path.join(save_dir, "lqr_controller_parameters.yml"), "w") as f:
            yaml.dump(par_dict, f)

        np.savetxt(os.path.join(save_dir, "lqr_controller_Qmatrix.txt"), self.Q)
        np.savetxt(os.path.join(save_dir, "lqr_controller_Rmatrix.txt"), self.R)
        np.savetxt(os.path.join(save_dir, "lqr_controller_Kmatrix.txt"), self.K)
        np.savetxt(os.path.join(save_dir, "lqr_controller_Smatrix.txt"), self.S)
