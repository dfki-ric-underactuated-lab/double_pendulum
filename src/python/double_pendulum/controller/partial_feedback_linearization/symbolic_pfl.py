import numpy as np
import sympy as smp

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.controller.lqr.lqr_controller import LQRController


class SymbolicPFLController(AbstractController):
    def __init__(self,
                 mass=[1.0, 1.0],
                 length=[0.5, 0.5],
                 com=[0.5, 0.5],
                 damping=[0.1, 0.1],
                 gravity=9.81,
                 coulomb_fric=[0.0, 0.0],
                 inertia=[None, None],
                 torque_limit=[np.inf, np.inf],
                 robot="acrobot",
                 pfl_method="collocated"):

        self.torque_limit = torque_limit

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

        self.plant = SymbolicDoublePendulum(mass,
                                            length,
                                            com,
                                            damping,
                                            gravity,
                                            coulomb_fric,
                                            inertia,
                                            torque_limit)

        M11, M12, M21, M22 = smp.symbols("M11 M12 M21 M22")
        C11, C12, C21, C22 = smp.symbols("C11 C12 C21 C22")
        G1, G2 = smp.symbols("G1 G2")
        F1, F2 = smp.symbols("F1 F2")

        M = smp.Matrix([[M11, M12], [M21, M22]])
        C = smp.Matrix([[C11, C12], [C21, C22]])
        G = smp.Matrix([G1, G2])
        F = smp.Matrix([F1, F2])

        eom = M*self.plant.qdd + C*self.plant.qd - G - F - self.plant.u
        eom = eom.subs(self.plant.u[self.passive_motor_ind], 0.)
        qdd_el = smp.solve(eom[self.passive_motor_ind],
                           self.plant.xd[2+self.eliminate_ind])[0]
        u_eq = eom[self.active_motor_ind].subs(
            self.plant.xd[2+self.eliminate_ind],
            qdd_el)
        self.u_out = smp.solve(u_eq, self.plant.u[self.active_motor_ind])[0]

        self.g1, self.g2, self.gd1, self.gd2 = smp.symbols(
            "g1 g2 \dot{g}_1 \dot{g}_2")
        self.goal = smp.Matrix([self.g1, self.g2, self.gd1, self.gd2])
        energy = self.plant.symbolic_total_energy()

        desired_energy = energy.subs(self.plant.x[0], self.goal[0])
        desired_energy = desired_energy.subs(self.plant.x[1], self.goal[1])
        desired_energy = desired_energy.subs(self.plant.x[2], self.goal[2])
        desired_energy = desired_energy.subs(self.plant.x[3], self.goal[3])

        ubar = self.plant.x[2+self.eliminate_ind]*(energy - desired_energy)  # todo check index for non-collocated
        #ubar = self.plant.x[2]*(energy - desired_energy)  # todo check index for non-collocated

        self.k1s, self.k2s, self.k3s = smp.symbols("k1 k2 k3")

        qdd_des = (-self.k1s*(self.plant.x[self.desired_ind] -
                              self.goal[self.desired_ind])
                   - self.k2s*(self.plant.x[2+self.desired_ind] -
                               self.goal[2+self.desired_ind])
                   + self.k3s*ubar)  # + F[1] + F[0]

        self.u_out = self.u_out.subs(self.plant.xd[2+self.desired_ind],
                                     qdd_des)

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
        self.k1 = kpos
        self.k2 = kvel
        self.k3 = ken

    def set_cost_parameters_(self, pars=[0.3, 0.005, 1.0]):
        self.k1 = pars[0]
        self.k2 = pars[1]
        self.k3 = pars[2]

    def set_goal(self, x=[np.pi, 0., 0., 0.]):
        self.desired_energy = self.plant.total_energy(x)  # *1.1
        self.desired_x = x

    def init(self):
        u_out = self.u_out.subs(self.k1s, self.k1)
        u_out = u_out.subs(self.k2s, self.k2)
        u_out = u_out.subs(self.k3s, self.k3)

        u_out = u_out.subs(self.goal[0], self.desired_x[0])
        u_out = u_out.subs(self.goal[1], self.desired_x[1])
        u_out = u_out.subs(self.goal[2], self.desired_x[2])
        u_out = u_out.subs(self.goal[3], self.desired_x[3])

        self.u_out_la = smp.utilities.lambdify(self.plant.x, u_out)

    def get_control_output(self, x, t=None):
        pos = np.copy(x[:2])
        vel = np.copy(x[2:])

        pos[1] = (pos[1] + np.pi) % (2*np.pi) - np.pi
        pos[0] = pos[0] % (2*np.pi)

        u_out = self.u_out_la(pos[0], pos[1], vel[0], vel[1])

        # print(u_out, self.active_motor_ind)
        u_out = np.clip(u_out,
                        -self.torque_limit[self.active_motor_ind],
                        self.torque_limit[self.active_motor_ind])
        u = [0., 0.]
        u[self.active_motor_ind] = u_out
        # print(x, u)

        # for logging only:
        energy = self.plant.total_energy(x)
        self.en.append(energy)

        return u

    def save(self, path="log_energy.csv"):
        np.savetxt(path, self.en)


class SymbolicPFLAndLQRController(AbstractController):
    def __init__(self,
                 mass=[1.0, 1.0],
                 length=[0.5, 0.5],
                 com=[0.5, 0.5],
                 damping=[0.1, 0.1],
                 gravity=9.81,
                 coulomb_fric=[0.0, 0.0],
                 inertia=[None, None],
                 torque_limit=[np.inf, np.inf],
                 robot="acrobot",
                 pfl_method="collocated"):

        self.en_controller = SymbolicPFLController(
            mass=mass,
            length=length,
            com=com,
            damping=damping,
            gravity=gravity,
            coulomb_fric=coulomb_fric,
            inertia=inertia,
            torque_limit=torque_limit,
            robot=robot,
            pfl_method=pfl_method)

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
        self.en_controller.set_cost_parameters_(pars)

    def set_goal(self, x):
        self.en_controller.set_goal(x)
        self.lqr_controller.set_goal(x)
        self.desired_energy = self.en_controller.plant.total_energy(x)

    def init(self):
        self.en_controller.init()
        self.lqr_controller.init()

    def get_control_output(self, x, t=None, verbose=False):
        u = self.lqr_controller.get_control_output(x, t)

        energy = self.en_controller.plant.total_energy(x)
        self.en.append(energy)

        if True in np.isnan(u):
            if self.active_controller == "lqr":
                self.active_controller = "energy"
                if verbose:
                    print("Switching to energy shaping control")
            u = self.en_controller.get_control_output(x, t)
        else:
            if self.active_controller == "energy":
                self.active_controller = "lqr"
                if verbose:
                    print("Switching to lqr control")

        return u
