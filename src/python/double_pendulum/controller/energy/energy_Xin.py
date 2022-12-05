import os
import yaml
import numpy as np
from scipy.optimize import minimize

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum


class EnergyController(AbstractController):
    """EnergyController
    Energy-based controller for acrobot swingup based on this paper:
    https://onlinelibrary.wiley.com/doi/abs/10.1002/rnc.1184

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
    motor_inertia : float, optional
        default=0.0
        inertia of the actuators/motors
        [Ir1, Ir2], units=[kg*m²]
    gear_ratio : int, optional
        gear ratio of the motors, default=6
    torque_limit : array_like, optional
        shape=(2,), dtype=float, default=[np.inf, np.inf]
        torque limit of the motors
        [tl1, tl2], units=[Nm, Nm]
    model_pars : model_parameters object, optional
        object of the model_parameters class, default=None
        Can be used to set all model parameters above
        If provided, the model_pars parameters overwrite
        the other provided parameters
        (Default value=None)
    """

    def __init__(self,
                 mass=[1.0, 1.0],
                 length=[0.5, 0.5],
                 com=[0.5, 0.5],
                 damping=[0.1, 0.1],
                 gravity=9.81,
                 coulomb_fric=[0.0, 0.0],
                 inertia=[None, None],
                 motor_inertia=0.,
                 gear_ratio=6,
                 torque_limit=[np.inf, np.inf],
                 model_pars=None):

        super().__init__()

        self.mass = mass
        self.length = length
        self.com = com
        self.damping = damping
        self.cfric = coulomb_fric
        self.gravity = gravity
        self.inertia = inertia
        self.Ir = motor_inertia
        self.gr = gear_ratio
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

        self.plant = SymbolicDoublePendulum(mass=self.mass,
                                            length=self.length,
                                            com=self.com,
                                            damping=self.damping,
                                            gravity=self.gravity,
                                            coulomb_fric=self.cfric,
                                            inertia=self.inertia,
                                            motor_inertia=self.Ir,
                                            gear_ratio=self.gr,
                                            torque_limit=self.torque_limit)

    def set_parameters(self, kp=1., kd=1., kv=1.):
        """set_parameters.
        Set controller gains.

        Parameters
        ----------
        kp : float
            gain for position error
        kd : float
            gain
        kv : float
            gain for velocity error
        """
        self.kp = kp
        self.kd = kd
        self.kv = kv

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
        self.desired_x = x
        self.desired_energy = self.plant.total_energy(x)

    def check_parameters(self):
        """
        Check if the parameters fulfill the convergence conditions presented in
        the paper.
        """
        a1 = self.inertia[0] + self.mass[1]*self.length[0]**2.
        a2 = self.inertia[1]
        a3 = self.mass[1]*self.com[1]*self.length[0]
        b1 = self.gravity*(self.mass[0]*self.com[0] + self.mass[1]*self.length[0])
        b2 = self.mass[1]*self.com[1]*self.gravity

        # kd
        res = minimize(fun=kd_func,
                       x0=(0.1,),
                       bounds=((0, 2.*np.pi),),
                       args=(a1, a2, a3, b1, b2, self.desired_energy))

        d_val = -res.fun[0]

        # kp
        p_val = 2/np.pi*np.min([b1**2., b2**2.])

        if self.kp > p_val:
            print(f"Kp={self.kp} fulfills the convergence property kp > {p_val}")
        else:
            print(f"Kp={self.kp} does NOT fulfill convergence property kp > {p_val}")

        if self.kd > d_val:
            print(f"Kd={self.kd} fulfills the convergence property kd > {d_val}")
        else:
            print(f"Kd={self.kd} does NOT fulfill the convergence property kd > {d_val}")

        if self.kv > 0:
            print(f"Kv={self.kv} fulfills the convergence property kv > 0")
        else:
            print(f"Kv={self.kv} does NOT fulfill the convergence property kv > 0")

        tmax = (a2*b1 + a3*b1 - a1*b2 - a3*b2) / (a1 + a2 + a3)
        print("Bound on torque: ", tmax)

    def init_(self):
        """
        Initialize the controller.
        """
        self.en = []

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

        energy = self.plant.total_energy(x)
        self.en.append(energy)
        en_diff = energy - self.desired_energy

        Del = M[0][0]*M[1][1] - M[0][1]*M[1][0]
        H = np.dot(C, vel)

        u2 = -((self.kv*vel[1]+self.kp*pos[1])*Del + \
                self.kd*(M[1][0]*(H[0]-G[0]+F[0]) - \
                         M[0][0]*(H[1]-G[1]+F[1]))) \
              / (self.kd*M[0][0]+en_diff*Del)

        u2 = np.clip(u2, -self.torque_limit[1], self.torque_limit[1])
        u = [0., u2]

        return u

    def save_(self, save_dir):
        """
        Save controller parameters

        Parameters
        ----------
        save_dir : string or path object
            directory where the parameters will be saved
        """

        np.savetxt(os.path.join(save_dir, "energy_log.txt"), self.en)

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
                "Ir" : self.Ir,
                "gr" : self.gr,
                "torque_limit1" : self.torque_limit[0],
                "torque_limit2" : self.torque_limit[1],
                "kp" : self.kp,
                "kd" : self.kd,
                "kv" : self.kv,
                "desired_x1" : self.desired_x[0],
                "desired_x2" : self.desired_x[1],
                "desired_x3" : self.desired_x[2],
                "desired_x4" : self.desired_x[3],
                "desired_energy" : float(self.desired_energy),
        }

        with open(os.path.join(save_dir, "controller_energy_Xin_parameters.yml"), 'w') as f:
            yaml.dump(par_dict, f)


def kd_func(q2, a1, a2, a3, b1, b2, Er):
    """
    Function to check the convergence property of kd.

    Parameters
    ----------
    q2 : float
    a1 : float
    a2 : float
    a3 : float
    b1 : float
    b2 : float
    Er : float
    """
    Del = a1*a2 - (a3*np.cos(q2))**2.
    Phi = np.sqrt(b1**2.+b2**2.+2.*b1*b2*np.cos(q2))
    M11 = a1 + a2 + 2.*a3*np.cos(q2)

    f = (Phi+Er)*Del / M11
    return -f
