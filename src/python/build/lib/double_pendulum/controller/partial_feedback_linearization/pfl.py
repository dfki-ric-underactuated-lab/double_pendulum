import os
import yaml
import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.controller.lqr.lqr_controller import LQRController


class EnergyShapingPFLController(AbstractController):
    """EnergyShapingPFLController
    Controller based on partial feedback linearization (PFL) which controls the nergy
    of the double pendulum. Uses collocated pfl for the acrobot.
    For non-collocated pfl and/or the pendubot use the SymbolicPFLController.
    It is based in these papers by Spong:
        - https://www.sciencedirect.com/science/article/pii/S1474667017474040?via%3Dihub
        - https://ieeexplore.ieee.org/document/341864
        - https://www.sciencedirect.com/science/article/pii/S1474667017581057?via%3Dihub

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
        shape=(2,), dtype=float, default=[np.inf, np.inf]
        torque limit of the motors
        [tl1, tl2], units=[Nm, Nm]
    """
    def __init__(self,
                 mass=[1.0, 1.0],
                 length=[0.5, 0.5],
                 com=[0.5, 0.5],
                 damping=[0.1, 0.1],
                 gravity=9.81,
                 coulomb_fric=[0.0, 0.0],
                 inertia=[None, None],
                 torque_limit=[np.inf, np.inf]):

        super().__init__()

        self.plant = SymbolicDoublePendulum(mass,
                                            length,
                                            com,
                                            damping,
                                            gravity,
                                            coulomb_fric,
                                            inertia,
                                            torque_limit)

        self.damping = damping
        self.torque_limit = torque_limit

        self.counter = 0
        self.u1 = 0.0
        self.u2 = 0.0
        self.desired_energy = 0.0

        # self.alpha = np.pi/6.0
        self.en = []

        self.set_parameters()

    def set_cost_parameters(self, kpos=0.3, kvel=0.005, ken=1.0):
        """
        Set controller gains

        Parameters
        ----------
        kpos : float
            Gain for position error
            (Default value = 0.3)
        kvel : float
            Gain for velocity error
            (Default value = 0.005)
        ken : float
            Gain for energy error
            (Default value = 1.0)
        """
        self.k1 = kpos
        self.k2 = kvel
        self.k3 = ken

    def set_cost_parameters_(self, pars=[0.3, 0.005, 1.0]):
        """
        Set controller gains with a list.
        (Useful for parameter optimization)

        Parameters
        ----------
        pars : list
            shape=(3,)
            list containing the controller gains in the order
            [kpos, kvel, ken]
            (Default value = [0.3, 0.005, 1.0])
        """
        self.k1 = pars[0]
        self.k2 = pars[1]
        self.k3 = pars[2]

    def set_goal(self, x):
        """set_goal.
        Set goal for the controller.

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        """
        self.desired_energy = self.plant.total_energy(x)
        self.desired_x = x

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
        pos = np.copy(x[:2])
        vel = np.copy(x[2:])

        pos[1] = (pos[1] + np.pi) % (2*np.pi) - np.pi
        pos[0] = pos[0] % (2*np.pi)

        M = self.plant.mass_matrix(x)
        C = self.plant.coriolis_matrix(x)
        G = self.plant.gravity_vector(x)
        F = self.plant.coulomb_vector(x)

        H = C.dot(vel)
        M11_inv = 1 / M[0][0]  # np.linalg.inv(M[0,0])
        MMMM = M[1][1] - M[1][0]*M11_inv*M[0][1]
        MM = M[1][0]*M11_inv

        energy = self.plant.total_energy(x)
        self.en.append(energy)

        ubar = vel[0]*(energy - self.desired_energy)

        # ubar = 2*self.alpha*np.arctan(vel[0])
        # ubar = self.torque_limit[1]/(0.5*np.pi)*np.arctan(ubar)

        u = (-self.k1*(pos[1]-self.desired_x[1])
             - self.k2*(vel[1]-self.desired_x[3])
             + self.k3*ubar)  # + F[1] + F[0]

        tau = MMMM*u + (H[1] - MM*H[0]) + (MM*G[0] - G[1]) + (MM*F[0] - F[1])

        self.u2 = np.clip(tau, -self.torque_limit[1], self.torque_limit[1])
        u = [self.u1, self.u2]
        # print(x, u)

        # this works if both joints are actuated
        # B = self.plant.B
        # v = np.array([[-(x[0] - np.pi) - x[2]], [-(x[1] - 0) - x[3]]])
        # GG = np.array([[G[0]], [G[1]]])
        # vvel = np.array([[vel[0]], [vel[1]]])
        # u = np.linalg.inv(B).dot(M.dot(v) + C.dot(vvel) - GG)
        # u = [u[0][0], u[1][0]]

        return u

    def save_(self, save_dir):
        """
        Save controller parameters

        Parameters
        ----------
        save_dir : string or path object
            directory where the parameters will be saved
        """
        np.savetxt(os.path.join(save_dir, "energy_log.csv"), self.en)

        par_dict = {
                "mass1" : self.plant.m[0],
                "mass2" : self.plant.m[1],
                "length1" : self.plant.l[0],
                "length2" : self.plant.l[1],
                "com1" : self.plant.com[0],
                "com2" : self.plant.com[1],
                "damping1" : self.damping[0],
                "damping2" : self.damping[1],
                "cfric1" : self.plant.coulomb_fric[0],
                "cfric2" : self.plant.coulomb_fric[1],
                "gravity" : self.plant.g,
                "inertia1" : self.plant.I[0],
                "inertia2" : self.plant.I[1],
                "Ir" : self.plant.Ir,
                "gr" : self.plant.gr,
                "torque_limit1" : self.torque_limit[0],
                "torque_limit2" : self.torque_limit[1],
                "k1" : self.k1,
                "k2" : self.k2,
                "k3" : self.k3,
                "desired_x1" : self.desired_x[0],
                "desired_x2" : self.desired_x[1],
                "desired_x3" : self.desired_x[2],
                "desired_x4" : self.desired_x[3],
                "desired_energy" : float(self.desired_energy),
        }

        with open(os.path.join(save_dir, "controller_pfl_parameters.yml"), 'w') as f:
            yaml.dump(par_dict, f)


class EnergyShapingPFLAndLQRController(AbstractController):
    """EnergyShapingPFLAndLQRController
    Controller based on partial feedback linearization which controls the nergy
    of the double pendulum. If the LQR controller returns a feasible value the
    control switches to LQR control.

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
        shape=(2,), dtype=float, default=[np.inf, np.inf]
        torque limit of the motors
        [tl1, tl2], units=[Nm, Nm]
    """
    def __init__(self,
                 mass=[1.0, 1.0],
                 length=[0.5, 0.5],
                 com=[0.5, 0.5],
                 damping=[0.1, 0.1],
                 gravity=9.81,
                 coulomb_fric=[0.0, 0.0],
                 inertia=[None, None],
                 torque_limit=[np.inf, np.inf]):

        super().__init__()

        self.en_controller = EnergyShapingPFLController(
            mass=mass,
            length=length,
            com=com,
            damping=damping,
            gravity=gravity,
            coulomb_fric=coulomb_fric,
            inertia=inertia,
            torque_limit=torque_limit)

        self.lqr_controller = LQRController(mass=mass,
                                            length=length,
                                            com=com,
                                            damping=damping,
                                            gravity=gravity,
                                            coulomb_fric=coulomb_fric,
                                            inertia=inertia,
                                            torque_limit=torque_limit)

        self.active_controller = "energy"
        self.en = []

    def set_cost_parameters_(self, pars=[0.3, 0.005, 1.0]):
        """
        Set controller gains for the PFL controller with a list.

        Parameters
        ----------
        pars : list
            shape=(3,)
            list containing the controller gains in the order
            [kpos, kvel, ken]
            (Default value = [0.3, 0.005, 1.0])
        """
        self.en_controller.set_cost_parameters_(pars)

    def set_goal(self, x):
        """set_goal.
        Set goal for the controller.

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        """
        self.en_controller.set_goal(x)
        self.lqr_controller.set_goal(x)
        self.desired_energy = self.en_controller.plant.total_energy(x)

    def init_(self):
        """
        Initialize the PFL and LQR controller.
        """
        self.en_controller.init_()
        self.lqr_controller.init_()

    def get_control_output_(self, x, t=None, verbose=False):
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
        verbose : bool
            Whether to print when the active controller is switched.
            (Default value = False)

        Returns
        -------
        array_like
            shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
        """
        u = self.lqr_controller.get_control_output_(x)

        energy = self.en_controller.plant.total_energy(x)
        self.en.append(energy)

        if True in np.isnan(u):
            if self.active_controller == "lqr":
                self.active_controller = "energy"
                if verbose:
                    print("Switching to energy shaping control")
            u = self.en_controller.get_control_output_(x)
        else:
            if self.active_controller == "energy":
                self.active_controller = "lqr"
                if verbose:
                    print("Switching to lqr control")

        return u
