import numpy as np
import sympy as smp
from sympy.utilities import lambdify


def diff_to_matrix(diff):
    """
    function to convert a sympy derivative to a sympy matrix

    Parameters
    ----------
    diff : sympy derivative


    Returns
    -------
    sympy matrix

    """
    mat = np.zeros((diff.shape[2], diff.shape[0])).tolist()
    for row in range(diff.shape[0]):
        for column in range(diff.shape[2]):
            mat[column][row] = diff[row][0][column][0]
    return smp.Matrix(mat)


def sub_symbols(mat, symbols, new_symbols):
    """
    substitute symbols with new symbols/values in mat

    Parameters
    ----------
    mat : sympy matrix
        matrix where symbols shall be replaced

    symbols : list of sympy variables
        symbols to replace

    new_symbols : list of sympy variables or floats
        will replace the symbols


    Returns
    -------
    sympy matrix
        matrix with replaced symbols

    """
    for i, sym in enumerate(symbols):
        mat = mat.subs(sym, new_symbols[i])
    return mat


def vector_mult(vec1, vec2):
    """
    scalar product of sympy vectors

    Parameters
    ----------
    vec1 : sympy vector

    vec2 : sympy vector


    Returns
    -------
    sympy variable
        vec1*vec2
    """
    v = 0
    for i in range(len(vec1)):
        v += vec1[i] * vec2[i]
    return v


class SymbolicDoublePendulum:
    """
    Symbolic double pendulum plant
    The double pendulum plant class calculates:
        - forward kinematics
        - forward/inverse dynamics
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
        "q1 q2 \dot{q}_1 \dot{q}_2 \ddot{q}_1 \ddot{q}_2"
    )

    q = smp.Matrix([q1, q2])
    qd = smp.Matrix([qd1, qd2])
    qdd = smp.Matrix([qdd1, qdd2])
    x = smp.Matrix([q1, q2, qd1, qd2])
    xd = smp.Matrix([qd1, qd2, qdd1, qdd2])

    u1, u2 = smp.symbols("u1 u2")
    u = smp.Matrix([u1, u2])

    # definition of linearization point
    q01, q02, q0d1, q0d2 = smp.symbols(
        "\hat{q}_1 \hat{q}_2 \hat{\dot{q}}_1 \hat{\dot{q}}_2"
    )
    x0 = smp.Matrix([q01, q02, q0d1, q0d2])

    u01, u02 = smp.symbols("\hat{u}_1 \hat{u}_2")
    u0 = smp.Matrix([u01, u02])

    def __init__(
        self,
        mass=[1.0, 1.0],
        length=[0.5, 0.5],
        com=[0.5, 0.5],
        damping=[0.1, 0.1],
        gravity=9.81,
        coulomb_fric=[0.0, 0.0],
        inertia=[None, None],
        motor_inertia=0.0,
        gear_ratio=6,
        torque_limit=[np.inf, np.inf],
        model_pars=None,
    ):
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
                self.I.append(mass[i] * com[i] * com[i])
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
        self.workspace_range = [
            [-1.2 * np.sum(self.l), 1.2 * np.sum(self.l)],
            [-1.2 * np.sum(self.l), 1.2 * np.sum(self.l)],
        ]

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
        """
        symbolic mass matrix from the equations of motion

        Returns
        -------
        sympy Matrix
            shape=(2,2)
            mass matrix
        """
        # Spong eq. have additional self.m2*self.r2**2.0 term in all entries
        # and self.m1*self.r1**2.0 in M11.
        # Guess: The axes for the inertias I1, I2 are defined different
        # Underactuated: center at rotation point, Spong: at com
        if self.formulas == "UnderactuatedLecture":
            M11 = (
                self.I1
                + self.I2
                + self.m2 * self.l1**2.0
                + 2 * self.m2 * self.l1 * self.r2 * smp.cos(self.q2)
                + self.gr_sym**2.0 * self.Ir_sym
                + self.Ir_sym
            )
            M12 = (
                self.I2
                + self.m2 * self.l1 * self.r2 * smp.cos(self.q2)
                - self.gr_sym * self.Ir_sym
            )
            M21 = (
                self.I2
                + self.m2 * self.l1 * self.r2 * smp.cos(self.q2)
                - self.gr_sym * self.Ir_sym
            )
            M22 = self.I2 + self.gr_sym**2.0 * self.Ir_sym
        elif self.formulas == "Spong":
            M11 = (
                self.I1
                + self.I2
                + self.m1 * self.r1**2.0
                + self.m2
                * (
                    self.l1**2.0
                    + self.r2**2.0
                    + 2 * self.l1 * self.r2 * smp.cos(self.q2)
                )
            )
            M12 = self.I2 + self.m2 * (
                self.r2**2.0 + self.l1 * self.r2 * smp.cos(self.q2)
            )
            M21 = self.I2 + self.m2 * (
                self.r2**2.0 + self.l1 * self.r2 * smp.cos(self.q2)
            )
            M22 = self.I2 + self.m2 * self.r2**2.0
        M = smp.Matrix([[M11, M12], [M21, M22]])
        return M

    def symbolic_coriolis_matrix(self):
        """
        symbolic coriolis matrix from the equations of motion

        Returns
        -------
        sympy Matrix
            shape=(2,2)
            coriolis matrix
        """
        # equal
        if self.formulas == "UnderactuatedLecture":
            C11 = -2 * self.m2 * self.l1 * self.r2 * smp.sin(self.q2) * self.qd2
            C12 = -self.m2 * self.l1 * self.r2 * smp.sin(self.q2) * self.qd2
            C21 = self.m2 * self.l1 * self.r2 * smp.sin(self.q2) * self.qd1
            C22 = 0
            C = smp.Matrix([[C11, C12], [C21, C22]])
        elif self.formulas == "Spong":
            C11 = -2 * self.m2 * self.l1 * self.r2 * smp.sin(self.q2) * self.qd2
            C12 = -self.m2 * self.l1 * self.r2 * smp.sin(self.q2) * self.qd2
            C21 = self.m2 * self.l1 * self.r2 * smp.sin(self.q2) * self.qd1
            C22 = 0
            C = smp.Matrix([[C11, C12], [C21, C22]])
        return C

    def symbolic_gravity_vector(self):
        """
        symbolic gravity vector from the equations of motion

        Returns
        -------
        sympy Matrix
            shape=(1,2)
            gravity vector
        """
        # equivalent
        if self.formulas == "UnderactuatedLecture":
            G1 = -self.m1 * self.g_sym * self.r1 * smp.sin(
                self.q1
            ) - self.m2 * self.g_sym * (
                self.l1 * smp.sin(self.q1) + self.r2 * smp.sin(self.q1 + self.q2)
            )
            G2 = -self.m2 * self.g_sym * self.r2 * smp.sin(self.q1 + self.q2)
        elif self.formulas == "Spong":
            G1 = -(self.m1 * self.r1 + self.m2 * self.l1) * self.g_sym * smp.cos(
                self.q1 - 0.5 * np.pi
            ) - self.m2 * self.r2 * self.g_sym * smp.cos(
                self.q1 - 0.5 * np.pi + self.q2
            )
            G2 = (
                -self.m2
                * self.r2
                * self.g_sym
                * smp.cos(self.q1 - 0.5 * np.pi + self.q2)
            )
        G = smp.Matrix([[G1], [G2]])
        return G

    def symbolic_coulomb_vector(self):
        """
        symbolic coulomb vector from the equations of motion

        Returns
        -------
        sympy Matrix
            shape=(1,2)
            coulomb vector
        """
        F1 = self.b1 * self.qd1 + self.cf1 * smp.atan(100 * self.qd1)
        F2 = self.b2 * self.qd2 + self.cf2 * smp.atan(100 * self.qd2)
        F = smp.Matrix([[F1], [F2]])
        return F

    def symbolic_kinetic_energy(self):
        """
        symbolic kinetic energy of the double pendulum
        """
        Ekin = 0.5 * vector_mult(self.qd.T, self.M * self.qd)
        return Ekin

    def symbolic_potential_energy(self):
        """
        symbolic potential energy of the double pendulum
        """
        h1 = -self.r1 * smp.cos(self.q1)
        h2 = -self.l1 * smp.cos(self.q1) - self.r2 * smp.cos(self.q1 + self.q2)
        Epot = self.m1 * self.g_sym * h1 + self.m2 * self.g_sym * h2
        return Epot

    def symbolic_total_energy(self):
        """
        symbolic total energy of the double pendulum
        """
        E = self.Ekin + self.Epot
        return E

    def symbolic_linear_matrices(self):
        """
        symbolic A- and B-matrix of the linearized dynamics (xd = Ax+Bu)
        """
        Alin = diff_to_matrix(smp.diff(self.f, self.x))
        Alin = sub_symbols(Alin, self.x, self.x0)
        Alin = sub_symbols(Alin, self.u, self.u0)

        Blin = diff_to_matrix(smp.diff(self.f, self.u)).T
        Blin = sub_symbols(Blin, self.x, self.x0)
        Blin = sub_symbols(Blin, self.u, self.u0)

        return Alin, Blin.T

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
        M = self.M_la(x[0], x[1], x[2], x[3])
        return np.asarray(M, dtype=float)

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
        C = self.C_la(x[0], x[1], x[2], x[3])
        return np.asarray(C, dtype=float)

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
        numpy array, shape=(1,2),
            gravity vector
        """
        G = self.G_la(x[0], x[1], x[2], x[3])
        return np.asarray(G, dtype=float).reshape(self.dof)

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
        numpy array, shape=(1,2),
            coulomb vector

        """
        F = self.F_la(x[0], x[1], x[2], x[3])
        return np.asarray(F, dtype=float).reshape(self.dof)

    def equation_of_motion(self, order="2nd"):
        """
        symbolic equation of motion

        Parameters
        ----------
        order : string
            string specifying the order of the eom ("1st" or "2nd")
            default="2nd"

        Returns
        -------
        sympy matrix
            if order=="1st"
                returns f from f=[qd, qdd] in the eom
            if order=="2nd"
                returns f from f=qdd in the eom
        """
        Minv = self.M.inv()
        if order == "2nd":
            # eom = (self.M*self.qdd
            #        + self.C*self.qd
            #        - self.G - self.B_sym*self.u + self.F)
            eom = Minv * (-self.C * self.qd + self.G + self.B_sym * self.u - self.F)
            return eom
        elif order == "1st":
            f1 = self.qd
            f2 = Minv * (-self.C * self.qd + self.G + self.B_sym * self.u - self.F)
            f = smp.Matrix([f1, f2])
            return f

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
        Ekin = self.Ekin_la(x[0], x[1], x[2], x[3])
        return np.asarray(Ekin, dtype=float)

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
        Epot = self.Epot_la(x[0], x[1], x[2], x[3])
        return np.asarray(Epot, dtype=float)

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
        E = self.E_la(x[0], x[1], x[2], x[3])
        return np.asarray(E, dtype=float)

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
        Alin = self.Alin_la(x0, u0)
        Blin = self.Blin_la(x0, u0)

        return np.asarray(Alin, dtype=float), np.asarray(Blin, dtype=float)

    def linear_matrices_discrete(self, x0, u0, dt):
        """
        get discrete A- and B-matrix of the linearized dynamics (xd = Ax+Bu)

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
        Alin, Blin = self.linear_matrices(x0, u0)
        Alin_disc = np.identity(np.shape(Alin)[0]) + dt * Alin
        Blin_disc = Blin * dt

        return np.asarray(Alin_disc, dtype=float), np.asarray(Blin_disc, dtype=float)

    def replace_parameters(self, mat):
        """

        function to replace the symbolic system parameters in the input matrix
        with the actual values of this plant

        Parameters
        ----------
        mat : sympy matrix
            matrix, where the symbolic parameters shall be replaced


        Returns
        -------
        sympy matrix
            matrix with replaced parameters

        """
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
        """
        function to lambdify the symbolic matrices of this plant to make them
        functions of state x and actuation u
        """
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
        ee1_pos_y = -self.l[0] * np.cos(pos[0])

        ee2_pos_x = ee1_pos_x + self.l[1] * np.sin(pos[0] + pos[1])
        ee2_pos_y = ee1_pos_y - self.l[1] * np.cos(pos[0] + pos[1])

        return [[ee1_pos_x, ee1_pos_y], [ee2_pos_x, ee2_pos_y]]

    def center_of_mass(self, pos):
        """
        calculate the center of mass of the whole system

        Parameters
        ----------
        pos : array_like, shape=(2,), dtype=float,
            positions of the double pendulum,
            order=[angle1, angle2],
            units=[rad]


        Returns
        -------
        list
            shape=(2,)
            cartesian coordinates of the center of mass, units=[m]

        """
        pre = 1.0 / (self.m[0] + self.m[1])
        cx = pre * (
            self.m[0] * self.com[0] * np.cos(pos[0])
            + self.m[1] * self.l[0] * np.cos(pos[0])
            + self.m[1] * self.com[1] * np.cos(pos[0] + pos[1])
        )
        cy = pre * (
            self.m[0] * self.r1 * np.sin(pos[0])
            + self.m[1] * self.l[0] * np.sin(pos[0])
            + self.m[1] * self.com[1] * np.sin(pos[0] + pos[1])
        )
        return [cx, cy]

    def com_dot(self, x):
        """
        calculate the time derivative of the center of mass
        of the whole system

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        Returns
        -------
        list
            shape=(2,)
            center of mass time derivative, units=[m/s]

        """
        pre = 1.0 / (self.m[0] + self.m[1])
        cx = pre * (
            -self.m[0] * self.com[0] * x[2] * np.sin(x[0])
            + -self.m[1] * self.l[0] * x[2] * np.sin(x[0])
            + -self.m[1] * self.com[1] * (x[2] + x[3]) * np.sin(x[0] + x[1])
        )
        cy = pre * (
            self.m[0] * self.r1 * x[2] * np.cos(x[0])
            + self.m[1] * self.l[0] * x[2] * np.cos(x[0])
            + self.m[1] * self.com[1] * (x[2] + x[3]) * np.cos(x[0] + x[1])
        )
        return [cx, cy]

    def angular_momentum_base(self, x):
        """
        angular momentum at the base

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        Returns
        -------
        float
            angular momentum, unit=[kg*m²/s]

        """
        M = self.mass_matrix(x)
        L = M[0, 0] * x[2] + M[0, 1] * x[3]
        return L

    def angular_momentum_dot_base(self, x):
        """
        first time derivative of the angular momentum at the base

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        Returns
        -------
        float
            first time derivative of the angular momentum, unit=[kg*m²/s²]

        """
        cx = self.center_of_mass(x[:2])[0]
        Ld = -(self.m[0] + self.m[1]) * self.g * cx
        return Ld

    def angular_momentum_ddot_base(self, x):
        """
        second time derivative of the angular momentum at the base

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        Returns
        -------
        float
            second time derivative of the angular momentum, unit=[kg*m²/s³]

        """
        cx_dot = self.com_dot(x)[0]
        Ldd = -(self.m[0] + self.m[1]) * self.g * cx_dot
        return Ldd

    def forward_dynamics(self, x, u):
        """
        forward dynamics of the double pendulum

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
        numpy array, shape=(2,)
            joint acceleration, [acc1, acc2], units=[m/s²]
        """
        # pos = np.copy(x[:self.dof])
        vel = np.copy(x[self.dof :])

        M = self.mass_matrix(x)
        C = self.coriolis_matrix(x)
        G = self.gravity_vector(x)
        F = self.coulomb_vector(x)

        Minv = np.linalg.inv(M)

        force = np.dot(self.B, u) - np.dot(C, vel) + G - F

        accn = Minv.dot(force)
        return accn

    def inverse_dynamics(self, x, acc):
        """
        inverse dynamics of the double pendulum

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        acc : array_like, shape=(2,), dtype=float
            joint acceleration,
            order=[acc1, acc2],
            units=[rad/s²]

        Returns
        -------
        array_like
            shape=(2,)
            actuation input/motor torque
            units=[Nm]

        """

        vel = np.copy(x[self.dof :])

        M = self.mass_matrix(x)
        C = self.coriolis_matrix(x)
        G = self.gravity_vector(x)
        F = self.coulomb_vector(x)

        tau = np.dot(M, acc) + np.dot(C, vel) - G + F
        return tau

    def rhs(self, t, x, u):
        """
        integrand of the equations of motion

        Parameters
        ----------
        t : float,
            time, units=[s], not used
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
        numpy array
            shape=(4,), dtype=float
            integrand, [vel1, vel2, acc1, acc2]
        """
        # Forward dynamics
        accn = self.forward_dynamics(x, u)

        # Next state
        res = np.zeros(2 * self.dof)
        res[0] = x[2]
        res[1] = x[3]
        res[2] = accn[0]
        res[3] = accn[1]
        return res
