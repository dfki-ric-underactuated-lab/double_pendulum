import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.controller.lqr.lqr import lqr, solve_differential_ricatti
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.utils.csv_trajectory import load_trajectory
from double_pendulum.utils.wrap_angles import wrap_angles_diff
from double_pendulum.utils.pcw_polynomial import InterpolateVector, InterpolateMatrix


class TVLQRController(AbstractController):
    def __init__(self,
                 mass=[0.5, 0.6],
                 length=[0.3, 0.2],
                 com=[0.3, 0.2],
                 damping=[0.1, 0.1],
                 coulomb_fric=[0.0, 0.0],
                 gravity=9.81,
                 inertia=[None, None],
                 torque_limit=[0.0, 1.0],
                 model_pars=None,
                 csv_path="",
                 num_break=40,
                 horizon=100,
                 ):

        super().__init__()

        # model parameters
        self.mass = mass
        self.length = length
        self.com = com
        self.damping = damping
        self.cfric = coulomb_fric
        self.gravity = gravity
        self.inertia = inertia
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

        self.splant = SymbolicDoublePendulum(
                mass=self.mass,
                length=self.length,
                com=self.com,
                damping=self.damping,
                gravity=self.gravity,
                coulomb_fric=self.cfric,
                inertia=self.inertia,
                torque_limit=self.torque_limit)

        self.num_break = num_break
        self.horizon = horizon

        # load trajectory
        self.T, self.X, self.U = load_trajectory(csv_path=csv_path,
                                                 with_tau=True)
        self.max_t = self.T[-1]
        self.dt = self.T[1] - self.T[0]

        # interpolate trajectory
        self.X_interp = InterpolateVector(
                T=self.T,
                X=self.X,
                num_break=num_break,
                poly_degree=3)

        self.U_interp = InterpolateVector(
                T=self.T,
                X=self.U,
                num_break=num_break,
                poly_degree=3)

        # default parameters
        self.Q = np.diag([4., 4., 0.1, 0.1])
        self.R = 2*np.eye(1)
        self.Qf = np.diag([4., 4., 0.1, 0.1])
        self.goal = np.array([np.pi, 0., 0., 0.])

        # initializations
        self.K = []
        # self.k = []

    def set_cost_parameters(self,
                            Q=np.diag([4., 4., 0.1, 0.1]),
                            R=2*np.eye(1),
                            Qf=np.diag([4., 4., 0.1, 0.1])):

        self.Q = np.asarray(Q)
        self.R = np.asarray(R)
        self.Qf = np.asarray(Qf)

    def set_goal(self, x=[np.pi, 0., 0., 0.]):
        y = x.copy()
        y[0] = y[0] % (2*np.pi)
        y[1] = (y[1] + np.pi) % (2*np.pi) - np.pi
        self.goal = np.asarray(y)

    def init_(self):
        self.K = []
        # self.k = []
        # for i in range(len(self.T[:-1])):
        for i in range(len(self.T)):
            A, B = self.splant.linear_matrices(x0=self.X[i], u0=self.U[i])
            # K, S, _ = lqr(A, B, self.Q, self.R)
            K, S = solve_differential_ricatti(A, B, self.Q, self.R, self.horizon, self.dt)
            K = K[0]
            S = S[0]
            # print(np.shape(K))
            self.K.append(K)

        A, B = self.splant.linear_matrices(x0=self.X[-1], u0=self.U[-1])
        self.K_final, _, _ = lqr(A, B, self.Qf, self.R)
        # self.K.append(K)
        self.K = np.asarray(self.K)

        self.K_interp = InterpolateMatrix(
                T=self.T,
                X=self.K,
                num_break=self.num_break,
                poly_degree=3)

    def get_control_output_(self, x, t):

        if t <= self.max_t:
            tt = min(t, self.max_t)

            x_error = wrap_angles_diff(np.asarray(x) - self.X_interp.get_value(tt))

            tau = self.U_interp.get_value(tt) - np.dot(self.K_interp.get_value(tt), x_error)
            u = [tau[0], tau[1]]
        else:
            x_error = wrap_angles_diff(np.asarray(x) - self.goal)

            u = - np.asarray(self.K_final.dot(x_error))[0]
            # u = [tau[1], tau[1]]

        # print(self.K_interp.get_value(tt))

        u[0] = np.clip(u[0], -self.torque_limit[0], self.torque_limit[0])
        u[1] = np.clip(u[1], -self.torque_limit[1], self.torque_limit[1])
        return u

    def get_init_trajectory(self):
        return self.T, self.X, self.U
