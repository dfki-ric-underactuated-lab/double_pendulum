import numpy as np

from double_pendulum.utils.wrap_angles import wrap_angles_top
from double_pendulum.trajectory_optimization.ilqr.ilqr_cpp import ilqr_calculator


class ilqr_trajopt_loss():
    def __init__(self,
                 par_prefactors,
                 loss_weights,
                 start,
                 goal):
        self.par_prefactors = par_prefactors
        self.weights = loss_weights
        self.start = start
        self.goal = goal

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
        self.dt = dt
        self.max_iter = max_iter
        self.regu_init = regu_init
        self.max_regu = max_regu
        self.min_regu = min_regu
        self.break_cost_redu = break_cost_redu
        self.integrator = integrator

    def __call__(self, pars):
        il = ilqr_calculator()
        il.set_model_parameters(mass=self.mass,
                                length=self.length,
                                com=self.com,
                                damping=self.damping,
                                gravity=self.gravity,
                                coulomb_fric=self.coulomb_fric,
                                inertia=self.inertia,
                                torque_limit=self.torque_limit)
        il.set_parameters(N=self.N,
                          dt=self.dt,
                          max_iter=self.max_iter,
                          regu_init=self.regu_init,
                          max_regu=self.max_regu,
                          min_regu=self.min_regu,
                          break_cost_redu=self.break_cost_redu,
                          integrator=self.integrator)
        il.set_cost_parameters_(self.par_prefactors*pars)
        il.set_start(self.start)
        il.set_goal(self.goal)

        # computing the trajectory
        T, X, U = il.compute_trajectory()
        y = wrap_angles_top(X[-1])
        dist = np.max(np.abs(y - self.goal))
        U1 = np.asarray(U).T[0]
        U2 = np.asarray(U).T[1]
        smooth = np.max(np.abs(np.diff(U1))) + np.max(np.abs(np.diff(U2)))
        loss = float(self.weights[0]*dist + self.weights[1]*smooth)
        return loss
