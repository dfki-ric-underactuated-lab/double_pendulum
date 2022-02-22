# import sys
import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController
# sys.path.append("../../../../cpp/python/")
# sys.path.append("/home/vhuser/Git/underactuated-robotics/caprr-release-version/src/cpp/python")
# sys.path.append("/home/felix/Work/DFKI/Development/underactuated_lab/double_pendulum/caprr-release-version/src/cpp/python")
import cppilqr


class ILQRMPCCPPController(AbstractController):
    def __init__(self,
                 mass=[0.608, 0.630],
                 length=[0.3, 0.2],
                 com=[0.275, 0.166],
                 damping=[0.081, 0.0],
                 coulomb_fric=[0.093, 0.186],
                 gravity=9.81,
                 inertia=[0.05472, 0.02522],
                 torque_limit=[0.0, 6.0]):

        # n_x = 4
        # n_u = 1
        self.mass = mass
        self.length = length
        self.com = com
        self.damping = damping
        self.coulomb_fric = coulomb_fric
        self.gravity = gravity
        self.inertia = inertia
        self.torque_limit = torque_limit

        self.counter = 0

    def set_start(self, x=[0., 0., 0., 0.]):
        self.start = np.asarray(x)

    def set_goal(self, x=[np.pi, 0., 0., 0.]):
        self.goal = np.asarray(x)

    def set_parameters(self,
                       N=1000,
                       dt=0.005,
                       max_iter=1,
                       regu_init=100,
                       max_regu=10000.,
                       min_regu=0.01,
                       break_cost_redu=1e-6,
                       integrator="runge_kutta"):
        self.N = N
        self.dt = dt
        self.max_iter = max_iter
        self.regu_init = regu_init
        self.max_regu = max_regu
        self.min_regu = min_regu
        self.break_cost_redu = break_cost_redu

        if integrator == "euler":
            self.integrator_int = 0
        else:
            self.integrator_int = 1

    def set_cost_parameters(self,
                            sCu=[0.005, 0.005],
                            sCp=[0., 0.],
                            sCv=[0., 0.],
                            sCen=0.,
                            fCp=[1000., 1000.],
                            fCv=[10., 10.],
                            fCen=0.):
        self.sCu = sCu
        self.sCp = sCp
        self.sCv = sCv
        self.sCen = sCen
        self.fCp = fCp
        self.fCv = fCv
        self.fCen = fCen

    def set_cost_parameters_(self,
                             pars=[0.005,
                                   0., 0.,
                                   0., 0.,
                                   1000., 1000.,
                                   10., 10.]):
        self.sCu = [pars[0], pars[0]]
        self.sCp = [pars[1], pars[2]]
        self.sCv = [pars[3], pars[4]]
        self.sCen = 0.0
        self.fCp = [pars[5], pars[6]]
        self.fCv = [pars[7], pars[8]]
        self.fCen = 0.0

    def compute_init_traj(self,
                          N=1000,
                          dt=0.005,
                          max_iter=100,
                          regu_init=100,
                          max_regu=10000.,
                          min_regu=0.01,
                          break_cost_redu=1e-6,
                          sCu=[0.005, 0.005],
                          sCp=[0., 0.],
                          sCv=[0., 0.],
                          sCen=0.,
                          fCp=[1000., 1000.],
                          fCv=[10., 10.],
                          fCen=0.,
                          integrator="runge_kutta"):

        if integrator == "euler":
            integrator_int = 0
        else:
            integrator_int = 1

        self.N_init = N

        il = cppilqr.cppilqr(N)
        il.set_parameters(integrator_int, dt)
        il.set_start(self.start[0], self.start[1],
                     self.start[2], self.start[3])
        il.set_model_parameters(
            self.mass[0], self.mass[1],
            self.length[0], self.length[1],
            self.com[0], self.com[1],
            self.inertia[0], self.inertia[1],
            self.damping[0], self.damping[1],
            self.coulomb_fric[0], self.coulomb_fric[1],
            self.gravity,
            self.torque_limit[0], self.torque_limit[1])
        il.set_cost_parameters(sCu[0], sCu[1],
                               sCp[0], sCp[1],
                               sCv[0], sCv[1],
                               sCen,
                               fCp[0], fCp[1],
                               fCv[0], fCv[1],
                               fCen)
        il.set_goal(self.goal[0], self.goal[1],
                    self.goal[2], self.goal[3])
        il.run_ilqr(max_iter, break_cost_redu, regu_init,
                    max_regu, min_regu)

        self.u1_init_traj = il.get_u1_traj()
        self.u2_init_traj = il.get_u2_traj()
        self.p1_init_traj = il.get_p1_traj()
        self.p2_init_traj = il.get_p2_traj()
        self.v1_init_traj = il.get_v1_traj()
        self.v2_init_traj = il.get_v2_traj()

    def load_init_traj(self,
                       csv_path):
        trajectory = np.loadtxt(csv_path, skiprows=1, delimiter=",")

        self.N_init = np.shape(trajectory)[0]

        self.u1_init_traj = np.ascontiguousarray(trajectory.T[5])
        self.u2_init_traj = np.ascontiguousarray(trajectory.T[6])
        self.p1_init_traj = np.ascontiguousarray(trajectory.T[1])
        self.p2_init_traj = np.ascontiguousarray(trajectory.T[2])
        self.v1_init_traj = np.ascontiguousarray(trajectory.T[3])
        self.v2_init_traj = np.ascontiguousarray(trajectory.T[4])

    def init(self):
        self.ilmpc = cppilqr.cppilqrmpc(self.N, self.N_init)
        self.ilmpc.set_parameters(self.integrator_int,
                                  self.dt,
                                  self.max_iter,
                                  self.break_cost_redu,
                                  self.regu_init,
                                  self.max_regu,
                                  self.min_regu)
        self.ilmpc.set_goal(self.goal[0], self.goal[1],
                            self.goal[2], self.goal[3])
        self.ilmpc.set_model_parameters(
            self.mass[0], self.mass[1],
            self.length[0], self.length[1],
            self.com[0], self.com[1],
            self.inertia[0], self.inertia[1],
            self.damping[0], self.damping[1],
            self.coulomb_fric[0], self.coulomb_fric[1],
            self.gravity,
            self.torque_limit[0], self.torque_limit[1])
        self.ilmpc.set_cost_parameters(self.sCu[0], self.sCu[1],
                                       self.sCp[0], self.sCp[1],
                                       self.sCv[0], self.sCv[1],
                                       self.sCen,
                                       self.fCp[0], self.fCp[1],
                                       self.fCv[0], self.fCv[1],
                                       self.fCen)
        # self.il.set_start(x[0], x[1], x[2], x[3])
        self.ilmpc.set_u_init_traj(self.u1_init_traj, self.u2_init_traj)
        self.ilmpc.set_x_init_traj(self.p1_init_traj, self.p2_init_traj,
                                   self.v1_init_traj, self.v2_init_traj)

    def get_control_output(self, x, t=None):
        # print("get control output")
        u2 = self.ilmpc.get_control_output(x[0], x[1], x[2], x[3])

        # u = [self.u1_traj[0], self.u2_traj[0]]
        u = [0.0, u2]

        self.counter += 1
        # print(self.counter)
        return u

    def get_init_trajectory(self):

        u1_traj = self.u1_init_traj
        u2_traj = self.u2_init_traj
        p1_traj = self.p1_init_traj
        p2_traj = self.p2_init_traj
        v1_traj = self.v1_init_traj
        v2_traj = self.v2_init_traj

        T = np.linspace(0, self.N*self.dt, self.N)
        X = np.asarray([p1_traj, p2_traj, v1_traj, v2_traj]).T
        U = np.asarray([u1_traj, u2_traj]).T

        return T, X, U

    def get_forecast(self):

        # throws seg fault
        # u1_traj = self.ilmpc.get_u1_traj()
        # u2_traj = self.ilmpc.get_u2_traj()
        # p1_traj = self.ilmpc.get_p1_traj()
        # p2_traj = self.ilmpc.get_p2_traj()
        # v1_traj = self.ilmpc.get_v1_traj()
        # v2_traj = self.ilmpc.get_v2_traj()

        u1_traj = self.u1_init_traj
        u2_traj = self.u2_init_traj
        p1_traj = self.p1_init_traj
        p2_traj = self.p2_init_traj
        v1_traj = self.v1_init_traj
        v2_traj = self.v2_init_traj

        T = np.linspace(0, self.N*self.dt, self.N)
        X = np.asarray([p1_traj, p2_traj, v1_traj, v2_traj]).T
        U = np.asarray([u1_traj, u2_traj]).T

        return T, X, U
