import numpy as np
from scipy.optimize import minimize

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum


class EnergyController(AbstractController):
    """
    Energy-based controller for acrobot swingup based on this paper:
    https://onlinelibrary.wiley.com/doi/abs/10.1002/rnc.1184
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
        self.kp = kp
        self.kd = kd
        self.kv = kv

    def set_goal(self, x):
        self.desired_x = x
        self.desired_energy = self.plant.total_energy(x)

    def check_parameters(self):
        # check parameters
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
        self.en = []

    def get_control_output_(self, x, t=None):
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

    def save(self, path="log_energy.csv"):
        np.savetxt(path, self.en)


def kd_func(q2, a1, a2, a3, b1, b2, Er):
    Del = a1*a2 - (a3*np.cos(q2))**2.
    Phi = np.sqrt(b1**2.+b2**2.+2.*b1*b2*np.cos(q2))
    M11 = a1 + a2 + 2.*a3*np.cos(q2)

    f = (Phi+Er)*Del / M11
    return -f
