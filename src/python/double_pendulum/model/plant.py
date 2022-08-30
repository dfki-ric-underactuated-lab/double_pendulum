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

        self.dof = 2
        self.n_actuators = 2
        self.base = [0, 0]
        self.n_links = 2
        self.workspace_range = [[-1.2*np.sum(self.l), 1.2*np.sum(self.l)],
                                [-1.2*np.sum(self.l), 1.2*np.sum(self.l)]]

        if torque_limit[0] == 0:
            self.B = np.array([[0, 0], [0, 1]])
        elif torque_limit[1] == 0:
            self.B = np.array([[1, 0], [0, 0]])
        else:
            self.B = np.array([[1, 0], [0, 1]])

        self.formulas = "UnderactuatedLecture"

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

        if self.formulas == "UnderactuatedLecture":
            m00 = self.I[0] + self.I[1] + self.m[1]*self.l[0]**2.0 + \
                    2*self.m[1]*self.l[0]*self.com[1]*np.cos(pos[1]) + \
                    self.gr**2.0*self.Ir + self.Ir
            m01 = self.I[1] + self.m[1]*self.l[0]*self.com[1]*np.cos(pos[1]) - \
                    self.gr*self.Ir
            m10 = self.I[1] + self.m[1]*self.l[0]*self.com[1]*np.cos(pos[1]) - \
                    self.gr*self.Ir
            m11 = self.I[1] + self.gr**2.0*self.Ir
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
            F[i] = self.b[i]*vel[i] + self.coulomb_fric[i]*np.arctan(100*vel[i])
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
        #friction = np.where(np.abs(F) > np.abs(force), force*np.sign(F), F)
        friction = F

        accn = Minv.dot(force - friction)
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

    def get_Mx(self, x, tau):
        Mx = np.zeros((2*self.dof, self.dof, self.dof))
        Mx[1, 0, 0] = -2*self.l[0]*self.m[1]*self.com[1]*np.sin(x[1])
        Mx[1, 0, 1] = -self.l[0]*self.m[1]*self.com[1]*np.sin(x[1])
        Mx[1, 1, 0] = -self.l[0]*self.m[1]*self.com[1]*np.sin(x[1])
        Mx[1, 1, 1] = 0
        return Mx

    def get_Minvx(self, x, tau):
        Minvx = np.zeros((2*self.dof, self.dof, self.dof))

        den = -self.I[0]*self.I[1] - self.I[1]*self.l[0]**2.*self.m[1] + \
                (self.l[0]*self.m[1]*self.com[1]*np.cos(x[1]))**2.

        h1 = self.l[0]*self.m[1]*self.com[1]

        Minvx[1, 0, 0] = -2.*self.I[1]*h1**2.*np.sin(x[1])*np.cos(x[1]) / den**2.
        Minvx[1, 0, 1] = 2*h1**2.*(self.I[1] + h1*np.cos(x[1])**2.*np.sin(x[1])) / den**2. - h1*np.sin(x[0]) / den
        Minvx[1, 1, 0] = 2*h1**2.*(self.I[1] + h1*np.cos(x[1])**2.*np.sin(x[1])) / den**2. - h1*np.sin(x[0]) / den
        Minvx[1, 1, 1] = 2*h1**2*(-self.I[0]-self.I[1]*self.l[0]**2.*self.m[1] - 2*h1*np.cos(x[1])**2.*np.sin(x[1])) / den**2.
        return Minvx

    def get_Cx(self, x, tau):
        Cx = np.zeros((2*self.dof, self.dof, self.dof))

        h1 = self.l[0]*self.m[1]*self.com[1]

        Cx[1, 0, 0] = -2.*h1*np.cos(x[1])*x[3]
        Cx[1, 0, 1] = -h1*np.cos(x[1])*x[3]
        Cx[1, 1, 0] = h1*np.cos(x[1])*x[2]

        Cx[2, 1, 0] = h1*np.sin(x[1])

        Cx[3, 0, 0] = -2*h1*np.sin(x[1])
        Cx[3, 0, 1] = -h1*np.sin(x[1])

        return Cx

    def get_Gx(self, x, tau):
        Gx = np.zeros((self.dof, 2*self.dof))

        Gx[0, 0] = -self.g*self.m[0]*self.com[0]*np.cos(x[0]) - \
                    self.g*self.m[1]*(self.l[0]*np.cos(x[0]) + self.com[1]*np.cos(x[0]+x[1]))
        Gx[0, 1] = -self.g*self.m[1]*self.com[1]*np.cos(x[0]+x[1])

        Gx[1, 0] = -self.g*self.m[1]*self.com[1]*np.cos(x[0]+x[1])
        Gx[1, 1] = -self.g*self.m[1]*self.com[1]*np.cos(x[0]+x[1])

        return Gx

    def get_Fx(self, x, tau):
        Fx = np.zeros((self.dof, 2*self.dof))

        Fx[0, 2] = self.b[0]
        Fx[1, 3] = self.b[1]
        return Fx

    def get_Alin(self, x, u):
        M = self.mass_matrix(x)
        C = self.coriolis_matrix(x)
        G = self.gravity_vector(x)
        F = self.coulomb_vector(x)

        Minv = np.linalg.inv(M)

        #Mx = self.get_Mx(x, u)
        Minvx = self.get_Minvx(x, u)
        Cx = self.get_Cx(x, u)
        Gx = self.get_Gx(x, u)
        Fx = self.get_Fx(x, u)

        Alin = np.zeros((2*self.dof, 2*self.dof))
        Alin[0, 2] = 1.
        Alin[1, 3] = 1.

        qddx = np.zeros((self.dof, 2*self.dof))
        qddx[0, 2] = 1.
        qddx[1, 3] = 1.

        Diff1 = np.zeros((self.dof, 2*self.dof))
        Diff2 = np.zeros((self.dof, 2*self.dof))
        tmpCx = np.zeros((self.dof, 2*self.dof))

        # ToDo: Make simple without loop and tmps
        for i in range(2*self.dof):
            tmp = np.dot(Minvx[i], (np.dot(self.B, u) - np.dot(C, x[2:]) + G - F))
            Diff1[0, i] = tmp[0]
            Diff2[1, i] = tmp[1]

            tmp2 = np.dot(Cx[i], x[2:])
            tmpCx[0, i] = tmp2[0]
            tmpCx[1, i] = tmp2[1]
            
        Diff2 = np.dot(Minv, -tmpCx - np.dot(C, qddx) + Gx - Fx)

        Alin[2:, : ] = Diff1 + Diff2
        return Alin

    def get_Blin(self, x, u):
        Blin = np.zeros((2*self.dof, self.dof))
        M = self.mass_matrix(x)
        Minv = np.linalg.inv(M)
        Blin[2:, :] = np.dot(Minv, self.B)
        return Blin

    def linear_matrices(self, x0, u0):
        Alin = self.get_Alin(x0, u0)
        Blin = self.get_Blin(x0, u0)
        return Alin, Blin
