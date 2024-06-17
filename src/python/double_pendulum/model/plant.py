import numpy as np


class DoublePendulumPlant():
    """
    Double pendulum plant
    The double pendulum plant class calculates:
        - forward kinematics
        - forward dynamics
        - dynamics matrices (mass, coriolis, gravity, friction)
        - state dynamics matrices (mass, coriolis, gravity, friction)
        - linearized dynamics
        - kinetic, potential, total energy

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

        Parameters
        ----------
        pos : array_like, shape=(2,), dtype=float,
            positions of the double pendulum,
            order=[angle1, angle2],
            units=[rad]

        Returns
        -------
        list of 2 lists=[[x1, y1], [x2, y2]]
            cartesian coordinates of the link end points
            units=[m]
        """
        ee1_pos_x = self.l[0] * np.sin(pos[0])
        ee1_pos_y = -self.l[0]*np.cos(pos[0])

        ee2_pos_x = ee1_pos_x + self.l[1]*np.sin(pos[0]+pos[1])
        ee2_pos_y = ee1_pos_y - self.l[1]*np.cos(pos[0]+pos[1])

        return [[ee1_pos_x, ee1_pos_y], [ee2_pos_x, ee2_pos_y]]

    def mass_matrix(self, x):
        """
        mass matrix from the equations of motion

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        Returns
        -------
        numpy array, shape=(2,2),
            mass matrix
        """
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
        """
        coriolis matrix from the equations of motion

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]


        Returns
        -------
        numpy array, shape=(2,2),
            coriolis matrix
        """
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
        """
        gravity vector from the equations of motion

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        Returns
        -------
        numpy array, shape=(2,),
            gravity vector
        """
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
        """
        coulomb vector from the equations of motion

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        Returns
        -------
        numpy array, shape=(2,),
            coulomb vector

        """
        # pos = np.copy(x[:self.dof])
        vel = np.copy(x[self.dof:])

        F = np.zeros(self.dof)
        for i in range(self.dof):
            F[i] = self.b[i]*vel[i] + self.coulomb_fric[i]*np.arctan(100*vel[i])
        return F

    def kinetic_energy(self, x):
        """
        kinetic energy of the double pendulum

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        Returns
        -------
        float
            kinetic energy, units=[J]
        """
        # pos = np.copy(x[:self.dof])
        vel = np.copy(x[self.dof:])
        M = self.mass_matrix(x)
        kin = M.dot(vel)
        kin = 0.5*np.dot(vel, kin)
        return kin

    def potential_energy(self, x):
        """
        potential energy of the double pendulum

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        Returns
        -------
        float
            potential energy, units=[J]
        """
        pos = np.copy(x[:self.dof])
        # vel = np.copy(x[self.dof:])

        # 0 level at hinge
        y1 = -self.com[0]*np.cos(pos[0])  # + self.l[0] + self.l[1]
        y2 = -self.l[0]*np.cos(pos[0]) - self.com[1]*np.cos(pos[1]+pos[0])  # + self.l[0] + self.l[1]
        pot = self.m[0]*self.g*y1 + self.m[1]*self.g*y2
        return pot

    def total_energy(self, x):
        """
        total energy of the double pendulum

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        Returns
        -------
        float
            total energy, units=[J]
        """
        E = self.kinetic_energy(x) + self.potential_energy(x)
        return E

    def forward_dynamics(self, x, tau):
        """
        forward dynamics of the double pendulum

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        tau : array_like, shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]

        Returns
        -------
        numpy array, shape=(2,)
            joint acceleration, [acc1, acc2], units=[m/s²]
        """
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
        """
        integrand of the equations of motion

        Parameters
        ----------
        t : float,
            time, units=[s], not used
        state : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        tau : array_like, shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]

        Returns
        -------
        numpy array
            shape=(4,), dtype=float
            integrand, [vel1, vel2, acc1, acc2]
        """
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
        """
        state derivative of mass matrix

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        tau : array_like, shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
            not used

        Returns
        -------
        numpy array
            shape=(4, 2, 2)
            derivative of mass matrix,
            Mx[i]=del(M)/del(x_i)
        """
        Mx = np.zeros((2*self.dof, self.dof, self.dof))
        Mx[1, 0, 0] = -2*self.l[0]*self.m[1]*self.com[1]*np.sin(x[1])
        Mx[1, 0, 1] = -self.l[0]*self.m[1]*self.com[1]*np.sin(x[1])
        Mx[1, 1, 0] = -self.l[0]*self.m[1]*self.com[1]*np.sin(x[1])
        Mx[1, 1, 1] = 0
        return Mx

    def get_Minvx(self, x, tau):
        """
        state derivative of inverse mass matrix

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        tau : array_like, shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
            not used

        Returns
        -------
        numpy array
            shape=(4, 2, 2)
            derivative of inverse mass matrix,
            Minvx[i]=del(Minv)/del(x_i)
        """
        Minvx = np.zeros((2*self.dof, self.dof, self.dof))

        den = -self.I[0]*self.I[1] - self.I[1]*self.l[0]**2.*self.m[1] + \
                (self.l[0]*self.m[1]*self.com[1]*np.cos(x[1]))**2.

        h1 = self.l[0]*self.m[1]*self.com[1]

        Minvx[1, 0, 0] = -2.*self.I[1]*h1**2.*np.sin(x[1])*np.cos(x[1]) / den**2.
        Minvx[1, 0, 1] = 2*h1**2.*(self.I[1] + h1*np.cos(x[1]))*np.cos(x[1])*np.sin(x[1]) / den**2. - \
                h1*np.sin(x[1]) / den
        Minvx[1, 1, 0] = 2*h1**2.*(self.I[1] + h1*np.cos(x[1]))*np.cos(x[1])*np.sin(x[1]) / den**2. - \
                h1*np.sin(x[1]) / den
        Minvx[1, 1, 1] = 2*h1**2*(-self.I[0]-self.I[1]-self.l[0]**2.*self.m[1]
                                  -2*h1*np.cos(x[1]))*np.cos(x[1])*np.sin(x[1]) / den**2. + \
                2*h1*np.sin(x[1])/den
        return Minvx

    def get_Cx(self, x, tau):
        """
        state derivative of coriolis matrix

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        tau : array_like, shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
            not used

        Returns
        -------
        numpy array
            shape=(4, 2, 2)
            derivative of coriolis matrix,
            Cx[i]=del(C)/del(x_i)
        """
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
        """
        state derivative of gravity vector

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        tau : array_like, shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
            not used

        Returns
        -------
        numpy array
            shape=(2, 4)
            derivative of gravity vector,
            Gx[:, i]=del(G)/del(x_i)
        """
        Gx = np.zeros((self.dof, 2*self.dof))

        Gx[0, 0] = -self.g*self.m[0]*self.com[0]*np.cos(x[0]) - \
                    self.g*self.m[1]*(self.l[0]*np.cos(x[0]) + self.com[1]*np.cos(x[0]+x[1]))
        Gx[0, 1] = -self.g*self.m[1]*self.com[1]*np.cos(x[0]+x[1])

        Gx[1, 0] = -self.g*self.m[1]*self.com[1]*np.cos(x[0]+x[1])
        Gx[1, 1] = -self.g*self.m[1]*self.com[1]*np.cos(x[0]+x[1])

        return Gx

    def get_Fx(self, x, tau):
        """
        state derivative of coulomb vector

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        tau : array_like, shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
            not used

        Returns
        -------
        numpy array
            shape=(2, 4)
            derivative of coulomb vector,
            Fx[:, i]=del(F)/del(x_i)

        """
        Fx = np.zeros((self.dof, 2*self.dof))

        Fx[0, 2] = self.b[0] + 100*self.coulomb_fric[0] / (1+(100*x[2])**2)
        Fx[1, 3] = self.b[1] + 100*self.coulomb_fric[1] / (1+(100*x[3])**2)
        return Fx

    def get_Alin(self, x, u):
        """
        A-matrix of the linearized dynamics (xd = Ax+Bu)

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        u : array_like, shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]

        Returns
        -------
        array_like
            shape=(4,4),
            A-matrix

        """
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

        lower = np.dot(Minvx, (np.dot(self.B, u) - np.dot(C, x[2:]) + G - F)).T + \
                np.dot(Minv, -np.dot(Cx, x[2:]).T - np.dot(C, qddx) + Gx - Fx)
        Alin[2:, :] = lower

        return Alin

    def get_Blin(self, x, u):
        """
        B-matrix of the linearized dynamics (xd = Ax+Bu)

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        u : array_like, shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]

        Returns
        -------
        array_like
            shape=(4,2),
            B-matrix

        """
        Blin = np.zeros((2*self.dof, self.dof))
        M = self.mass_matrix(x)
        Minv = np.linalg.inv(M)
        Blin[2:, :] = np.dot(Minv, self.B)
        return Blin

    def linear_matrices(self, x0, u0):
        """
        get A- and B-matrix of the linearized dynamics (xd = Ax+Bu)

        Parameters
        ----------
        x0 : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        u0 : array_like, shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]


        Returns
        -------
        array_like
            shape=(4,4),
            A-matrix
        array_like
            shape=(4,2),
            B-matrix

        """
        Alin = self.get_Alin(x0, u0)
        Blin = self.get_Blin(x0, u0)
        return Alin, Blin
