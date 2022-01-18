import numpy as np


class DoublePendulumPlant():
    def __init__(self,
                 mass=[1.0, 1.0],
                 length=[0.5, 0.5],
                 com=[0.5, 0.5],
                 damping=[0.1, 0.1],
                 gravity=9.81,
                 coulomb_fric=[0.0, 0.0],
                 inertia=[None, None],
                 torque_limit=[np.inf, np.inf]):

        self.m = mass
        self.l = length
        self.com = com
        self.b = damping
        self.g = gravity
        self.coulomb_fric = coulomb_fric
        self.I = []
        self.torque_limit = torque_limit

        self.dof = 2
        self.n_actuators = 2
        self.base = [0, 0]
        self.n_links = 2
        self.workspace_range = [[-1.2*np.sum(self.l), 1.2*np.sum(self.l)],
                                [-1.2*np.sum(self.l), 1.2*np.sum(self.l)]]

        for i in range(self.dof):
            if inertia[i] is None:
                self.I.append(mass[i]*com[i]*com[i])
            else:
                self.I.append(inertia[i])

        if torque_limit[0] == 0:
            self.B = np.array([[0, 0], [0, 1]])
        elif torque_limit[1] == 0:
            self.B = np.array([[1, 0], [0, 0]])
        else:
            self.B = np.array([[1, 0], [0, 1]])

        self.formulas = "Spong"

    def forward_kinematics(self, pos):
        """
        forward kinematics, origin at fixed point
        """
        ee1_pos_x = self.l[0] * np.sin(pos[0])
        ee1_pos_y = -self.l[0]*np.cos(pos[0])

        ee2_pos_x = ee1_pos_x + self.l[1]*np.sin(pos[0]+pos[1])
        ee2_pos_y = ee1_pos_y - self.l[1]*np.cos(pos[0]+pos[1])

        return [[ee1_pos_x, ee1_pos_y], [ee2_pos_x, ee2_pos_y]]

    def mass_matrix(self, x):
        pos = np.copy(x[:self.dof])
        # vel = np.copy(x[self.dof:])

        # Spong eq. have additional self.m2*self.r2**2.0 term in all entries
        # why? this has different results!
        if self.formulas == "UnderactuatedLecture":
            m00 = self.I[0] + self.I[1] + self.m[1]*self.l[0]**2.0 + \
                    2*self.m[1]*self.l[0]*self.com[1]*np.cos(pos[1])
            m01 = self.I[1] + self.m[1]*self.l[0]*self.com[1]*np.cos(pos[1])
            m10 = self.I[1] + self.m[1]*self.l[0]*self.com[1]*np.cos(pos[1])
            m11 = self.I[1]
            M = np.array([[m00, m01], [m10, m11]])

        elif self.formulas == "Spong":
            pos[0] -= 0.5*np.pi  # Spong uses different 0 position
            m00 = self.I[0] + self.I[1] + self.m[0]*self.com[0]**2.0 + \
                self.m[1]*(self.l[0]**2.0 + self.com[1]**2.0 +
                           2*self.l[0]*self.com[1]*np.cos(pos[1]))
            m01 = self.I[1] + self.m[1]*(self.com[1]**2.0 + self.l[0]*self.com[1]*np.cos(pos[1]))
            m10 = self.I[1] + self.m[1]*(self.com[1]**2.0 + self.l[0]*self.com[1]*np.cos(pos[1]))
            m11 = self.I[1] + self.m[1]*self.com[1]**2.0
            M = np.array([[m00, m01], [m10, m11]])
        return M

    def coriolis_matrix(self, x):
        pos = np.copy(x[:self.dof])
        vel = np.copy(x[self.dof:])

        # equal
        if self.formulas == "UnderactuatedLecture":
            C00 = -2*self.m[1]*self.l[0]*self.com[1]*np.sin(pos[1])*vel[1]
            C01 = -self.m[1]*self.l[0]*self.com[1]*np.sin(pos[1])*vel[1]
            C10 = self.m[1]*self.l[0]*self.com[1]*np.sin(pos[1])*vel[0]
            C11 = 0
            C = np.array([[C00, C01], [C10, C11]])

        elif self.formulas == "Spong":  # same as UnderacteadLecture
            pos[0] -= 0.5*np.pi  # Spong uses different 0 position
            C00 = -2*self.m[1]*self.l[0]*self.com[1]*np.sin(pos[1])*vel[1]
            C01 = -self.m[1]*self.l[0]*self.com[1]*np.sin(pos[1])*vel[1]
            C10 = self.m[1]*self.l[0]*self.com[1]*np.sin(pos[1])*vel[0]
            C11 = 0
            C = np.array([[C00, C01], [C10, C11]])

        return C

    def gravity_vector(self, x):
        pos = np.copy(x[:self.dof])
        # vel = np.copy(x[self.dof:])

        if self.formulas == "UnderactuatedLecture":
            G0 = -self.m[0]*self.g*self.com[0]*np.sin(pos[0]) - \
                 self.m[1]*self.g*(self.l[0]*np.sin(pos[0]) +
                                   self.com[1]*np.sin(pos[0]+pos[1]))
            G1 = -self.m[1]*self.g*self.com[1]*np.sin(pos[0]+pos[1])
            G = np.array([G0, G1])
        elif self.formulas == "Spong":
            pos[0] -= 0.5*np.pi  # Spong uses different 0 position, 
            # in the end the formulas are equal bc. sin(x) = cos(x-0.5pi)
            G0 = -(self.m[0]*self.com[0] + self.m[1]*self.l[0])*self.g*np.cos(pos[0]) - \
                self.m[1]*self.com[1]*self.g*np.cos(pos[0]+pos[1])
            G1 = -self.m[1]*self.com[1]*self.g*np.cos(pos[0]+pos[1])
            G = np.array([G0, G1])
        return G

    def coulomb_vector(self, x):
        # pos = np.copy(x[:self.dof])
        vel = np.copy(x[self.dof:])

        F = np.zeros(self.dof)
        for i in range(self.dof):
            F[i] = self.b[i]*vel[i] + self.coulomb_fric[i]*np.sign(vel[i])
        return F

    def kinetic_energy(self, x):
        # pos = np.copy(x[:self.dof])
        vel = np.copy(x[self.dof:])
        M = self.mass_matrix(x)
        kin = M.dot(vel)
        kin = 0.5*np.dot(vel, kin)
        return kin

    def potential_energy(self, x):
        pos = np.copy(x[:self.dof])
        # vel = np.copy(x[self.dof:])

        # 0 level at hinge
        y1 = -self.com[0]*np.cos(pos[0])  # + self.l[0] + self.l[1]
        y2 = -self.l[0]*np.cos(pos[0]) - self.com[1]*np.cos(pos[1]+pos[0])  # + self.l[0] + self.l[1]
        pot = self.m[0]*self.g*y1 + self.m[1]*self.g*y2
        return pot

    def total_energy(self, x):
        E = self.kinetic_energy(x) + self.potential_energy(x)
        return E

    def forward_dynamics(self, x, tau):
        # pos = np.copy(x[:self.dof])
        vel = np.copy(x[self.dof:])

        M = self.mass_matrix(x)
        C = self.coriolis_matrix(x)
        G = self.gravity_vector(x)
        F = self.coulomb_vector(x)

        Minv = np.linalg.inv(M)

        force = G + self.B.dot(tau) - C.dot(vel)
        friction = np.where(np.abs(F) > np.abs(force), force*np.sign(F), F)

        accn = Minv.dot(force - friction)
        # accn = Minv.dot(G + self.B.dot(tau) - C.dot(vel) - F)
        return accn

    def rhs(self, t, state, tau):
        # Forward dynamics
        accn = self.forward_dynamics(state, tau)

        # Next state
        res = np.zeros(2*self.dof)
        res[0] = state[2]
        res[1] = state[3]
        res[2] = accn[0]
        res[3] = accn[1]
        return res
