import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.controller.lqr.lqr_controller import LQRController


class EnergyShapingPFLController(AbstractController):
    def __init__(self,
                 mass=[1.0, 1.0],
                 length=[0.5, 0.5],
                 com=[0.5, 0.5],
                 damping=[0.1, 0.1],
                 gravity=9.81,
                 coulomb_fric=[0.0, 0.0],
                 inertia=[None, None],
                 torque_limit=[np.inf, np.inf]):

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

        self.k1 = 4.6
        self.k2 = 1.0
        self.k3 = 0.3

        # self.alpha = np.pi/6.0

        self.en = []

    def set_hyperpar(self, kpos=0.3, kvel=0.005, ken=1.0):
        self.k1 = kpos
        self.k2 = kvel
        self.k3 = ken

    def set_parameters(self, pars=[0.3, 0.005, 1.0]):
        self.k1 = pars[0]
        self.k2 = pars[1]
        self.k3 = pars[2]

    def set_goal(self, x):
        self.desired_energy = self.plant.total_energy(x)  # *1.1
        self.desired_x = x

    def get_control_output(self, x):
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

        tau = MMMM*u + (H[1] - MM*H[0]) - (G[1] - MM*G[0]) + (F[1] - MM*F[0])

        self.u2 = np.clip(tau, -self.torque_limit[1], self.torque_limit[1])
        u = [self.u1, self.u2]

        # this works if both joints are actuated
        # B = self.plant.B
        # v = np.array([[-(x[0] - np.pi) - x[2]], [-(x[1] - 0) - x[3]]])
        # GG = np.array([[G[0]], [G[1]]])
        # vvel = np.array([[vel[0]], [vel[1]]])
        # u = np.linalg.inv(B).dot(M.dot(v) + C.dot(vvel) - GG)
        # u = [u[0][0], u[1][0]]

        return u

    def save(self, path="log_energy.csv"):
        np.savetxt(path, self.en)


class EnergyShapingPFLAndLQRController(AbstractController):
    def __init__(self,
                 mass=[1.0, 1.0],
                 length=[0.5, 0.5],
                 com=[0.5, 0.5],
                 damping=[0.1, 0.1],
                 gravity=9.81,
                 coulomb_fric=[0.0, 0.0],
                 inertia=[None, None],
                 torque_limit=[np.inf, np.inf]):

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

    def set_hyperpar(self, kpos=0.3, kvel=0.005, ken=1.0):
        self.en_controller.set_hyperpar(kpos=kpos, kvel=kvel, ken=ken)

    def set_parameters(self, pars=[0.3, 0.005, 1.0]):
        self.en_controller.set_parameters(pars)

    def set_goal(self, x):
        self.en_controller.set_goal(x)
        self.lqr_controller.set_goal(x)
        self.desired_energy = self.en_controller.plant.total_energy(x)

    def init(self):
        self.en_controller.init()
        self.lqr_controller.init()

    def get_control_output(self, x, verbose=False):
        u = self.lqr_controller.get_control_output(x)

        energy = self.en_controller.plant.total_energy(x)
        self.en.append(energy)

        if True in np.isnan(u):
            if self.active_controller == "lqr":
                self.active_controller = "energy"
                if verbose:
                    print("Switching to energy shaping control")
            u = self.en_controller.get_control_output(x)
        else:
            if self.active_controller == "energy":
                self.active_controller = "lqr"
                if verbose:
                    print("Switching to lqr control")

        return u
