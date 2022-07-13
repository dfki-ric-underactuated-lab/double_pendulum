import numpy as np
import sympy as smp
from sympy.utilities import lambdify


def diff_to_matrix(diff):
    mat = np.zeros((diff.shape[2], diff.shape[0])).tolist()
    for row in range(diff.shape[0]):
        for column in range(diff.shape[2]):
            mat[column][row] = diff[row][0][column][0]
    return smp.Matrix(mat)


def sub_symbols(mat, symbols, new_symbols):
    for i, sym in enumerate(symbols):
        mat = mat.subs(sym, new_symbols[i])
    return mat


def vector_mult(vec1, vec2):
    v = 0
    for i in range(len(vec1)):
        v += vec1[i]*vec2[i]
    return v


class SymbolicDoublePendulum():

    # Acrobot parameters
    dof = 2
    n_actuators = 2
    base = [0, 0]
    n_links = 2

    m1, m2 = smp.symbols("m1 m2")
    l1, l2 = smp.symbols("l1 l2")
    r1, r2 = smp.symbols("r1 r2")  # center of masses
    I1, I2 = smp.symbols("I1 I2")
    Ir_sym = smp.symbols("Ir")
    gr_sym = smp.symbols("gr")
    g_sym = smp.symbols("g")
    b1, b2 = smp.symbols("b1 b2")
    cf1, cf2 = smp.symbols("cf1 cf2")
    tl1, tl2 = smp.symbols("tl1 tl2")

    # state space variables
    q1, q2, qd1, qd2, qdd1, qdd2 = smp.symbols(
        "q1 q2 \dot{q}_1 \dot{q}_2 \ddot{q}_1 \ddot{q}_2")

    q = smp.Matrix([q1, q2])
    qd = smp.Matrix([qd1, qd2])
    qdd = smp.Matrix([qdd1, qdd2])
    x = smp.Matrix([q1, q2, qd1, qd2])
    xd = smp.Matrix([qd1, qd2, qdd1, qdd2])

    u1, u2 = smp.symbols("u1 u2")
    u = smp.Matrix([u1, u2])

    # definition of linearization point
    q01, q02, q0d1, q0d2 = smp.symbols(
            "\hat{q}_1 \hat{q}_2 \hat{\dot{q}}_1 \hat{\dot{q}}_2")
    x0 = smp.Matrix([q01, q02, q0d1, q0d2])

    u01, u02 = smp.symbols("\hat{u}_1 \hat{u}_2")
    u0 = smp.Matrix([u01, u02])

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
        self.m = mass
        self.l = length
        self.com = com
        self.b = damping
        self.g = gravity
        self.coulomb_fric = coulomb_fric
        self.I = []
        self.Ir = motor_inertia
        self.gr = gear_ratio
        self.torque_limit = torque_limit
        for i in range(len(inertia)):
            if inertia[i] is None:
                self.I.append(mass[i]*com[i]*com[i])
            else:
                self.I.append(inertia[i])

        if model_pars is not None:
            self.m = model_pars.m
            self.l = model_pars.l
            self.com = model_pars.r
            self.b = model_pars.b
            self.coulomb_fric = model_pars.cf
            self.g = model_pars.g
            self.I = model_pars.I
            self.Ir = model_pars.Ir
            self.gr = model_pars.gr
            self.torque_limit = model_pars.tl

        # Actuator selection Matrix
        if torque_limit[0] == 0:
            self.B_sym = smp.Matrix([[0, 0], [0, 1]])
            self.B = np.array([[0, 0], [0, 1]])
        elif torque_limit[1] == 0:
            self.B_sym = smp.Matrix([[1, 0], [0, 0]])
            self.B = np.array([[1, 0], [0, 0]])
        else:
            self.B_sym = smp.Matrix([[1, 0], [0, 1]])
            self.B = np.array([[1, 0], [0, 1]])

        # needed for plotting
        self.workspace_range = [[-1.2*np.sum(self.l), 1.2*np.sum(self.l)],
                                [-1.2*np.sum(self.l), 1.2*np.sum(self.l)]]

        self.formulas = "UnderactuatedLecture"
        # self.formulas = "Spong"

        self.M = self.symbolic_mass_matrix()
        self.C = self.symbolic_coriolis_matrix()
        self.G = self.symbolic_gravity_vector()
        self.F = self.symbolic_coulomb_vector()

        self.eom = self.equation_of_motion(order="2nd")
        self.f = self.equation_of_motion(order="1st")

        self.Ekin = self.symbolic_kinetic_energy()
        self.Epot = self.symbolic_potential_energy()
        self.E = self.symbolic_total_energy()

        self.Alin, self.Blin = self.symbolic_linear_matrices()

        self.lambdify_matrices()

    def symbolic_mass_matrix(self):
        # Spong eq. have additional self.m2*self.r2**2.0 term in all entries
        # and self.m1*self.r1**2.0 in M11.
        # why? this has different results!
        # Guess: The axes for the inertias I1, I2 are defined different
        # Underactuated: center at rotation point, Spong: at com
        if self.formulas == "UnderactuatedLecture":
            M11 = self.I1 + self.I2 + self.m2*self.l1**2.0 + \
                2*self.m2*self.l1*self.r2*smp.cos(self.q2) + \
                self.gr_sym**2.0*self.Ir_sym + self.Ir_sym
            M12 = self.I2 + self.m2*self.l1*self.r2*smp.cos(self.q2) - \
                    self.gr_sym*self.Ir_sym
            M21 = self.I2 + self.m2*self.l1*self.r2*smp.cos(self.q2) - \
                    self.gr_sym*self.Ir_sym
            M22 = self.I2 + self.gr_sym**2.0*self.Ir_sym
        elif self.formulas == "Spong":
            M11 = self.I1 + self.I2 + self.m1*self.r1**2.0 + \
                  self.m2*(self.l1**2.0
                           + self.r2**2.0
                           + 2*self.l1*self.r2*smp.cos(self.q2))
            M12 = self.I2 + self.m2*(self.r2**2.0
                                     + self.l1*self.r2*smp.cos(self.q2))
            M21 = self.I2 + self.m2*(self.r2**2.0
                                     + self.l1*self.r2*smp.cos(self.q2))
            M22 = self.I2 + self.m2*self.r2**2.0
        M = smp.Matrix([[M11, M12], [M21, M22]])
        return M

    def symbolic_coriolis_matrix(self):
        # equal
        if self.formulas == "UnderactuatedLecture":
            C11 = -2*self.m2*self.l1*self.r2*smp.sin(self.q2)*self.qd2
            C12 = -self.m2*self.l1*self.r2*smp.sin(self.q2)*self.qd2
            C21 = self.m2*self.l1*self.r2*smp.sin(self.q2)*self.qd1
            C22 = 0
            C = smp.Matrix([[C11, C12], [C21, C22]])
        elif self.formulas == "Spong":
            C11 = -2*self.m2*self.l1*self.r2*smp.sin(self.q2)*self.qd2
            C12 = -self.m2*self.l1*self.r2*smp.sin(self.q2)*self.qd2
            C21 = self.m2*self.l1*self.r2*smp.sin(self.q2)*self.qd1
            C22 = 0
            C = smp.Matrix([[C11, C12], [C21, C22]])
        return C

    def symbolic_gravity_vector(self):
        # equivalent
        if self.formulas == "UnderactuatedLecture":
            G1 = (-self.m1*self.g_sym*self.r1*smp.sin(self.q1)
                  - self.m2*self.g_sym*(self.l1*smp.sin(self.q1)
                                        + self.r2*smp.sin(self.q1+self.q2)))
            G2 = -self.m2*self.g_sym*self.r2*smp.sin(self.q1+self.q2)
        elif self.formulas == "Spong":
            G1 = -(self.m1*self.r1 + self.m2*self.l1)*self.g_sym*smp.cos(self.q1-0.5*np.pi) - \
                  self.m2*self.r2*self.g_sym*smp.cos(self.q1-0.5*np.pi+self.q2)
            G2 = -self.m2*self.r2*self.g_sym*smp.cos(self.q1-0.5*np.pi+self.q2)
        G = smp.Matrix([[G1], [G2]])
        return G

    def symbolic_coulomb_vector(self):
        F1 = self.b1*self.qd1 + self.cf1*smp.atan(100*self.qd1)
        F2 = self.b2*self.qd2 + self.cf2*smp.atan(100*self.qd2)
        F = smp.Matrix([[F1], [F2]])
        return F

    def symbolic_kinetic_energy(self):
        Ekin = 0.5*vector_mult(self.qd.T, self.M*self.qd)
        return Ekin

    def symbolic_potential_energy(self):
        h1 = -self.r1*smp.cos(self.q1)
        h2 = -self.l1*smp.cos(self.q1) - self.r2*smp.cos(self.q1+self.q2)
        Epot = self.m1*self.g_sym*h1 + self.m2*self.g_sym*h2
        return Epot

    def symbolic_total_energy(self):
        E = self.Ekin + self.Epot
        return E

    def symbolic_linear_matrices(self):
        Alin = diff_to_matrix(smp.diff(self.f, self.x))
        Alin = sub_symbols(Alin, self.x, self.x0)
        Alin = sub_symbols(Alin, self.u, self.u0)

        Blin = diff_to_matrix(smp.diff(self.f, self.u)).T
        Blin = sub_symbols(Blin, self.x, self.x0)
        Blin = sub_symbols(Blin, self.u, self.u0)
        return Alin, Blin.T

    def mass_matrix(self, x):
        M = self.M_la(x[0], x[1], x[2], x[3])
        return np.asarray(M, dtype=float)

    def coriolis_matrix(self, x):
        C = self.C_la(x[0], x[1], x[2], x[3])
        return np.asarray(C, dtype=float)

    def gravity_vector(self, x):
        G = self.G_la(x[0], x[1], x[2], x[3])
        return np.asarray(G, dtype=float).reshape(self.dof)

    def coulomb_vector(self, x):
        F = self.F_la(x[0], x[1], x[2], x[3])
        return np.asarray(F, dtype=float).reshape(self.dof)

    def equation_of_motion(self, order="2nd"):
        Minv = self.M.inv()
        if order == "2nd":
            # eom = (self.M*self.qdd
            #        + self.C*self.qd
            #        - self.G - self.B_sym*self.u + self.F)
            eom = Minv*(-self.C*self.qd + self.G + self.B_sym*self.u - self.F)
            return eom
        elif order == "1st":
            f1 = self.qd
            f2 = Minv*(-self.C*self.qd + self.G + self.B_sym*self.u - self.F)
            f = smp.Matrix([f1, f2])
            return f

    def kinetic_energy(self, x):
        Ekin = self.Ekin_la(x[0], x[1], x[2], x[3])
        return np.asarray(Ekin, dtype=float)

    def potential_energy(self, x):
        Epot = self.Epot_la(x[0], x[1], x[2], x[3])
        return np.asarray(Epot, dtype=float)

    def total_energy(self, x):
        E = self.E_la(x[0], x[1], x[2], x[3])
        return np.asarray(E, dtype=float)

    def linear_matrices(self, x0, u0):
        Alin = self.Alin_la(x0, u0)
        Blin = self.Blin_la(x0, u0)
        return np.asarray(Alin, dtype=float), np.asarray(Blin, dtype=float)

    def replace_parameters(self, mat):
        mat_rep = sub_symbols(mat, [self.m1, self.m2], self.m)
        mat_rep = sub_symbols(mat_rep, [self.l1, self.l2], self.l)
        mat_rep = sub_symbols(mat_rep, [self.r1, self.r2], self.com)
        mat_rep = sub_symbols(mat_rep, [self.I1, self.I2], self.I)
        mat_rep = sub_symbols(mat_rep, [self.g_sym], [self.g])
        mat_rep = sub_symbols(mat_rep, [self.b1, self.b2], self.b)
        mat_rep = sub_symbols(mat_rep, [self.cf1, self.cf2], self.coulomb_fric)
        mat_rep = sub_symbols(mat_rep, [self.Ir_sym], [self.Ir])
        mat_rep = sub_symbols(mat_rep, [self.gr_sym], [self.gr])
        return mat_rep

    def lambdify_matrices(self):
        M = self.replace_parameters(self.M)
        C = self.replace_parameters(self.C)
        G = self.replace_parameters(self.G)
        F = self.replace_parameters(self.F)
        Ekin = self.replace_parameters(self.Ekin)
        Epot = self.replace_parameters(self.Epot)
        E = self.replace_parameters(self.E)
        Alin = self.replace_parameters(self.Alin)
        Blin = self.replace_parameters(self.Blin)

        self.M_la = lambdify(self.x, M)
        self.C_la = lambdify(self.x, C)
        self.G_la = lambdify(self.x, G)
        self.F_la = lambdify(self.x, F)
        self.Ekin_la = lambdify(self.x, Ekin)
        self.Epot_la = lambdify(self.x, Epot)
        self.E_la = lambdify(self.x, E)
        self.Alin_la = lambdify((self.x0, self.u0), Alin)
        self.Blin_la = lambdify((self.x0, self.u0), Blin)

    def forward_kinematics(self, pos):
        """
        forward kinematics, origin at fixed point
        """
        ee1_pos_x = self.l[0] * np.sin(pos[0])
        ee1_pos_y = -self.l[0]*np.cos(pos[0])

        ee2_pos_x = ee1_pos_x + self.l[1]*np.sin(pos[0]+pos[1])
        ee2_pos_y = ee1_pos_y - self.l[1]*np.cos(pos[0]+pos[1])

        return [[ee1_pos_x, ee1_pos_y], [ee2_pos_x, ee2_pos_y]]

    def forward_dynamics(self, x, u):
        # pos = np.copy(x[:self.dof])
        vel = np.copy(x[self.dof:])

        M = self.mass_matrix(x)
        C = self.coriolis_matrix(x)
        G = self.gravity_vector(x)
        F = self.coulomb_vector(x)

        Minv = np.linalg.inv(M)

        # print("x", x)
        # print("u", u)
        # print("M", M)
        # print("C", C)
        # print("G", G)
        # print("F", F)
        # print("Minv", Minv[0, 0], Minv[0, 1], Minv[1, 0], Minv[1, 1])

        force = G + self.B.dot(u) - C.dot(vel)
        # print("force", force)
        # friction = np.where(np.abs(F) > np.abs(force),
        #                     np.abs(force)*np.sign(F),
        #                     F)
        friction = F

        accn = Minv.dot(force - friction)
        # accn = Minv.dot(G + self.B.dot(tau) - C.dot(vel) - F)
        # print("accn", accn)
        return accn

    def inverse_dynamics(self, x, acc):

        vel = np.copy(x[self.dof:])

        M = self.mass_matrix(x)
        C = self.coriolis_matrix(x)
        G = self.gravity_vector(x)
        F = self.coulomb_vector(x)

        tau = np.dot(M, acc) + np.dot(C, vel) - G + F
        return tau

    def rhs(self, t, x, u):
        # Forward dynamics
        accn = self.forward_dynamics(x, u)

        # Next state
        res = np.zeros(2*self.dof)
        res[0] = x[2]
        res[1] = x[3]
        res[2] = accn[0]
        res[3] = accn[1]
        return res
