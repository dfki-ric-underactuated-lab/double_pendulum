import sys
import numpy as np

# sys.path.append("../../../../cpp/python/")
sys.path.append("/home/felix/Work/DFKI/Development/underactuated_lab/double_pendulum/caprr-release-version/src/cpp/python/")
from cppilqr import cppilqr


class ilqr_calculator():
    def _init__(self):

        # set default parameters
        self.set_model_parameters()
        self.set_parameters()

    def set_model_parameters(self,
                             mass=[0.608, 0.630],
                             length=[0.3, 0.2],
                             com=[0.275, 0.166],
                             damping=[0.081, 0.0],
                             coulomb_fric=[0.093, 0.186],
                             gravity=9.81,
                             inertia=[0.05472, 0.02522],
                             torque_limit=[0.0, 6.0]):

        self.mass = mass
        self.length = length
        self.com = com
        self.damping = damping
        self.coulomb_fric = coulomb_fric
        self.gravity = gravity
        self.inertia = inertia
        self.torque_limit = torque_limit

    def set_parameters(self,
                       N=1000,
                       dt=0.005,
                       max_iter=100,
                       regu_init=100,
                       max_regu=10000.,
                       min_regu=0.01,
                       break_cost_redu=1e-6,
                       integrator="runge_kutta"):
        self.N = N
        self.dt = 0.005
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
                             pars=[0.005, 0.005,  # sCu
                                   0., 0.,        # sCp
                                   0., 0.,        # sCv
                                   0.,            # sCen
                                   1000., 1000.,  # fCp
                                   10., 10.,      # fCv
                                   0.]):          # fCen
        self.sCu = [pars[0], pars[1]]
        self.sCp = [pars[2], pars[3]]
        self.sCv = [pars[4], pars[5]]
        self.sCen = pars[6]
        self.fCp = [pars[7], pars[8]]
        self.fCv = [pars[9], pars[10]]
        self.fCen = pars[11]

    def set_start(self, x):
        self.start = x

    def set_goal(self, x):
        self.goal = x

    def compute_trajectory(self):

        il = cppilqr(self.N)
        il.set_parameters(self.integrator_int, self.dt)
        il.set_start(self.start[0], self.start[1],
                     self.start[2], self.start[3])
        il.set_goal(self.goal[0], self.goal[1],
                    self.goal[2], self.goal[3])
        il.set_model_parameters(
            self.mass[0], self.mass[1],
            self.length[0], self.length[1],
            self.com[0], self.com[1],
            self.inertia[0], self.inertia[1],
            self.damping[0], self.damping[1],
            self.coulomb_fric[0], self.coulomb_fric[1],
            self.gravity,
            self.torque_limit[0], self.torque_limit[1])
        il.set_cost_parameters(self.sCu[0], self.sCu[1],
                               self.sCp[0], self.sCp[1],
                               self.sCv[0], self.sCv[1],
                               self.sCen,
                               self.fCp[0], self.fCp[1],
                               self.fCv[0], self.fCv[1],
                               self.fCen)
        il.run_ilqr(self.max_iter,
                    self.break_cost_redu,
                    self.regu_init,
                    self.max_regu,
                    self.min_regu)

        u1_traj = il.get_u1_traj()
        u2_traj = il.get_u2_traj()
        p1_traj = il.get_p1_traj()
        p2_traj = il.get_p2_traj()
        v1_traj = il.get_v1_traj()
        v2_traj = il.get_v2_traj()

        T = np.linspace(0, self.N*self.dt, self.N)
        X = np.asarray([p1_traj, p2_traj, v1_traj, v2_traj]).T
        U = np.asarray([u1_traj, u2_traj]).T

        return T, X, U
