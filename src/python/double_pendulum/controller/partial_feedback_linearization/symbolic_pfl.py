import os
import yaml
import numpy as np
import sympy as smp

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.controller.lqr.lqr_controller import LQRController


class SymbolicPFLController(AbstractController):
    """SymbolicPFLController
    Controller based on partial feedback linearization (PFL) which controls the nergy
    of the double pendulum. Can be used for collocated and non-collocated pfl and for
    acrobot and pendubot.
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
    model_pars : model_parameters object, optional
        object of the model_parameters class, default=None
        Can be used to set all model parameters above
        If provided, the model_pars parameters overwrite
        the other provided parameters
    robot : string
        the system to be used
            - "acrobot"
            - "pendubot"
        (Default value="acrobot")
    pfl_method : string
        the PFL method to be used
            - "collocated"
            - "noncollocated"
        (Default value="collocated")
    reference : string
        the property to be controlled
            - "energy"
            - "energysat"
            - "q1sat"
            - "q1"
        (Default value="energy")
    """

    def __init__(
        self,
        mass=[1.0, 1.0],
        length=[0.5, 0.5],
        com=[0.5, 0.5],
        damping=[0.1, 0.1],
        gravity=9.81,
        coulomb_fric=[0.0, 0.0],
        inertia=[None, None],
        torque_limit=[np.inf, np.inf],
        model_pars=None,
        robot="acrobot",
        pfl_method="collocated",
        reference="energy",
    ):
        super().__init__()

        self.mass = mass
        self.length = length
        self.com = com
        self.damping = damping
        self.coulomb_fric = coulomb_fric
        self.gravity = gravity
        self.inertia = inertia
        self.torque_limit = torque_limit

        if model_pars is not None:
            self.mass = model_pars.m
            self.length = model_pars.l
            self.com = model_pars.r
            self.damping = model_pars.b
            self.coulomb_fric = model_pars.cf
            self.gravity = model_pars.g
            self.inertia = model_pars.I
            # self.Ir = model_pars.Ir
            # self.gr = model_pars.gr
            self.torque_limit = model_pars.tl

        self.robot = robot
        self.pfl_method = pfl_method
        self.reference = reference

        if robot == "acrobot":
            self.active_motor_ind = 1
            self.passive_motor_ind = 0
            if pfl_method == "collocated":
                self.eliminate_ind = 0
                self.desired_ind = 1
            elif pfl_method == "noncollocated":
                self.eliminate_ind = 1
                self.desired_ind = 0
        elif robot == "pendubot":
            self.active_motor_ind = 0
            self.passive_motor_ind = 1
            if pfl_method == "collocated":
                self.eliminate_ind = 1
                self.desired_ind = 0
            elif pfl_method == "noncollocated":
                self.eliminate_ind = 0
                self.desired_ind = 1

        self.plant = SymbolicDoublePendulum(
            mass=self.mass,
            length=self.length,
            com=self.com,
            damping=self.damping,
            gravity=self.gravity,
            coulomb_fric=self.coulomb_fric,
            inertia=self.inertia,
            torque_limit=self.torque_limit,
        )

        M11, M12, M21, M22 = smp.symbols("M11 M12 M21 M22")
        C11, C12, C21, C22 = smp.symbols("C11 C12 C21 C22")
        G1, G2 = smp.symbols("G1 G2")
        F1, F2 = smp.symbols("F1 F2")

        M = smp.Matrix([[M11, M12], [M21, M22]])
        C = smp.Matrix([[C11, C12], [C21, C22]])
        G = smp.Matrix([G1, G2])
        F = smp.Matrix([F1, F2])

        eom = M * self.plant.qdd + C * self.plant.qd - G - F - self.plant.u
        eom = eom.subs(self.plant.u[self.passive_motor_ind], 0.0)
        qdd_el = smp.solve(
            eom[self.passive_motor_ind], self.plant.xd[2 + self.eliminate_ind]
        )[0]
        u_eq = eom[self.active_motor_ind].subs(
            self.plant.xd[2 + self.eliminate_ind], qdd_el
        )
        self.u_out = smp.solve(u_eq, self.plant.u[self.active_motor_ind])[0]

        self.g1, self.g2, self.gd1, self.gd2 = smp.symbols("g1 g2 \dot{g}_1 \dot{g}_2")
        self.goal = smp.Matrix([self.g1, self.g2, self.gd1, self.gd2])
        energy = self.plant.symbolic_total_energy()

        desired_energy = energy.subs(self.plant.x[0], self.goal[0])
        desired_energy = desired_energy.subs(self.plant.x[1], self.goal[1])
        desired_energy = desired_energy.subs(self.plant.x[2], self.goal[2])
        desired_energy = desired_energy.subs(self.plant.x[3], self.goal[3])

        if reference == "energy":
            ubar = self.plant.x[2 + self.eliminate_ind] * (
                energy - desired_energy
            )  # todo check index for non-collocated
            # ubar = self.plant.x[2]*(energy - desired_energy)  # todo check index for non-collocated
        elif reference == "energysat":
            ubar = smp.functions.elementary.hyperbolic.tanh(
                self.plant.x[2 + self.eliminate_ind] * (energy - desired_energy)
            )
        elif reference == "q1sat":
            ubar = smp.functions.elementary.hyperbolic.tanh(
                self.plant.x[2 + self.eliminate_ind]
            )
        elif reference == "q1":
            ubar = self.plant.x[self.eliminate_ind] - self.goal[self.eliminate_ind]

        self.k1s, self.k2s, self.k3s = smp.symbols("k1 k2 k3")

        qdd_des = (
            -self.k1s * (self.plant.x[self.desired_ind] - self.goal[self.desired_ind])
            - self.k2s
            * (self.plant.x[2 + self.desired_ind] - self.goal[2 + self.desired_ind])
            + self.k3s * ubar
        )  # + F[1] + F[0]

        self.u_out = self.u_out.subs(self.plant.xd[2 + self.desired_ind], qdd_des)

        M_plant = self.plant.symbolic_mass_matrix()
        C_plant = self.plant.symbolic_coriolis_matrix()
        G_plant = self.plant.symbolic_gravity_vector()
        F_plant = self.plant.symbolic_coulomb_vector()

        self.u_out = self.u_out.subs(M11, M_plant[0, 0])
        self.u_out = self.u_out.subs(M12, M_plant[0, 1])
        self.u_out = self.u_out.subs(M21, M_plant[1, 0])
        self.u_out = self.u_out.subs(M22, M_plant[1, 1])
        self.u_out = self.u_out.subs(C11, C_plant[0, 0])
        self.u_out = self.u_out.subs(C12, C_plant[0, 1])
        self.u_out = self.u_out.subs(C21, C_plant[1, 0])
        self.u_out = self.u_out.subs(C22, C_plant[1, 1])
        self.u_out = self.u_out.subs(G1, G_plant[0])
        self.u_out = self.u_out.subs(G2, G_plant[1])
        self.u_out = self.u_out.subs(F1, F_plant[0])
        self.u_out = self.u_out.subs(F2, F_plant[1])

        self.u_out = self.plant.replace_parameters(self.u_out)

        # self.alpha = np.pi/6.0
        self.en = []

        self.set_parameters()
        self.set_goal()

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
        self.desired_energy = self.plant.total_energy(x)  # *1.1
        self.desired_x = x

    def init_(self):
        """
        Initialize the controller.
        """
        u_out = self.u_out.subs(self.k1s, self.k1)
        u_out = u_out.subs(self.k2s, self.k2)
        u_out = u_out.subs(self.k3s, self.k3)

        u_out = u_out.subs(self.goal[0], self.desired_x[0])
        u_out = u_out.subs(self.goal[1], self.desired_x[1])
        u_out = u_out.subs(self.goal[2], self.desired_x[2])
        u_out = u_out.subs(self.goal[3], self.desired_x[3])

        self.u_out_la = smp.utilities.lambdify(self.plant.x, u_out)

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

        pos[1] = (pos[1] + np.pi) % (2 * np.pi) - np.pi
        pos[0] = pos[0] % (2 * np.pi)

        if np.abs(pos[0]) < 0.01 and pos[1] < 0.01 and vel[0] < 0.01 and vel[1] < 0.01:
            u_out = self.torque_limit[self.active_motor_ind]
        else:
            u_out = self.u_out_la(pos[0], pos[1], vel[0], vel[1])

        # print(u_out, self.active_motor_ind)
        # u_out = np.clip(
        #     u_out,
        #     -self.torque_limit[self.active_motor_ind],
        #     self.torque_limit[self.active_motor_ind],
        # )
        u = [0.0, 0.0]
        u[self.active_motor_ind] = u_out
        # print(x, u)
        #
        u[0] = np.clip(u[0], -self.torque_limit[0], self.torque_limit[0])
        u[1] = np.clip(u[1], -self.torque_limit[1], self.torque_limit[1])

        # for logging only:
        energy = self.plant.total_energy(x)
        self.en.append(energy)

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
            "mass1": self.plant.m[0],
            "mass2": self.plant.m[1],
            "length1": self.plant.l[0],
            "length2": self.plant.l[1],
            "com1": self.plant.com[0],
            "com2": self.plant.com[1],
            "damping1": self.damping[0],
            "damping2": self.damping[1],
            "cfric1": self.plant.coulomb_fric[0],
            "cfric2": self.plant.coulomb_fric[1],
            "gravity": self.plant.g,
            "inertia1": self.plant.I[0],
            "inertia2": self.plant.I[1],
            "Ir": self.plant.Ir,
            "gr": self.plant.gr,
            "torque_limit1": self.torque_limit[0],
            "torque_limit2": self.torque_limit[1],
            "k1": self.k1,
            "k2": self.k2,
            "k3": self.k3,
            "desired_x1": self.desired_x[0],
            "desired_x2": self.desired_x[1],
            "desired_x3": self.desired_x[2],
            "desired_x4": self.desired_x[3],
            "desired_energy": float(self.desired_energy),
            "robot": self.robot,
            "pfl_method": self.pfl_method,
            "reference": self.reference,
        }

        with open(
            os.path.join(save_dir, "controller_pfl_symbolic_parameters.yml"), "w"
        ) as f:
            yaml.dump(par_dict, f)


class SymbolicPFLAndLQRController(AbstractController):
    """SymbolicPFLAndLQRController
    Controller based on partial feedback linearization (PFL) which controls the nergy
    of the double pendulum. Can be used for collocated and non-collocated pfl and for
    acrobot and pendubot. If the LQR controller returns a feasible value the
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
    model_pars : model_parameters object, optional
        object of the model_parameters class, default=None
        Can be used to set all model parameters above
        If provided, the model_pars parameters overwrite
        the other provided parameters
    robot : string
        the system to be used
            - "acrobot"
            - "pendubot"
        (Default value="acrobot")
    pfl_method : string
        the PFL method to be used
            - "collocated"
            - "noncollocated"
        (Default value="collocated")
    reference : string
        the property to be controlled
            - "energy"
            - "energysat"
            - "q1sat"
            - "q1"
        (Default value="energy")
    """

    def __init__(
        self,
        mass=[1.0, 1.0],
        length=[0.5, 0.5],
        com=[0.5, 0.5],
        damping=[0.1, 0.1],
        gravity=9.81,
        coulomb_fric=[0.0, 0.0],
        inertia=[None, None],
        torque_limit=[np.inf, np.inf],
        model_pars=None,
        robot="acrobot",
        pfl_method="collocated",
        reference="energy",
    ):
        super().__init__()

        self.mass = mass
        self.length = length
        self.com = com
        self.damping = damping
        self.coulomb_fric = coulomb_fric
        self.gravity = gravity
        self.inertia = inertia
        self.torque_limit = torque_limit

        if model_pars is not None:
            self.mass = model_pars.m
            self.length = model_pars.l
            self.com = model_pars.r
            self.damping = model_pars.b
            self.coulomb_fric = model_pars.cf
            self.gravity = model_pars.g
            self.inertia = model_pars.I
            # self.Ir = model_pars.Ir
            # self.gr = model_pars.gr
            self.torque_limit = model_pars.tl

        self.en_controller = SymbolicPFLController(
            mass=self.mass,
            length=self.length,
            com=self.com,
            damping=self.damping,
            gravity=self.gravity,
            coulomb_fric=self.coulomb_fric,
            inertia=self.inertia,
            torque_limit=self.torque_limit,
            robot=robot,
            pfl_method=pfl_method,
            reference=reference,
        )

        self.lqr_controller = LQRController(
            mass=self.mass,
            length=self.length,
            com=self.com,
            damping=self.damping,
            gravity=self.gravity,
            coulomb_fric=self.coulomb_fric,
            inertia=self.inertia,
            torque_limit=self.torque_limit,
        )

        self.active_controller = "energy"
        self.en = []

    def set_cost_parameters_(self, pars=[0.3, 0.005, 1.0]):
        """
        Set PFL controller gains with a list.
        (Useful for parameter optimization)

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
        Initialize the controller.
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
        u = self.lqr_controller.get_control_output_(x, t)

        energy = self.en_controller.plant.total_energy(x)
        self.en.append(energy)

        if True in np.isnan(u):
            if self.active_controller == "lqr":
                self.active_controller = "energy"
                if verbose:
                    print("Switching to energy shaping control")
            u = self.en_controller.get_control_output_(x, t)
        else:
            if self.active_controller == "energy":
                self.active_controller = "lqr"
                if verbose:
                    print("Switching to lqr control")

        return u

    def save_(self, save_dir):
        """
        Save controller parameters

        Parameters
        ----------
        save_dir : string or path object
            directory where the parameters will be saved
        """
        self.en_controller.save_(save_dir)
        self.lqr_controller.save_(save_dir)
