import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.ilqr.ilqr_mpc_cpp import ILQRMPCCPPController
from double_pendulum.utils.wrap_angles import wrap_angles_top, wrap_angles_diff


class ilqrmpc_swingup_loss():
    def __init__(self,
                 par_prefactors,
                 loss_weights,
                 start,
                 goal,
                 csv_path):

        self.start = np.asarray(start)
        self.goal = np.asarray(goal)
        self.par_prefactors = np.asarray(par_prefactors)
        self.csv_path = csv_path
        self.loss_weights = loss_weights

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
        self.t_final = N*dt
        self.max_iter = max_iter
        self.regu_init = regu_init
        self.max_regu = max_regu
        self.min_regu = min_regu
        self.break_cost_redu = break_cost_redu
        self.integrator = integrator

    def init(self):
        self.plant = SymbolicDoublePendulum(mass=self.mass,
                                            length=self.length,
                                            com=self.com,
                                            damping=self.damping,
                                            gravity=self.gravity,
                                            coulomb_fric=self.coulomb_fric,
                                            inertia=self.inertia,
                                            torque_limit=self.torque_limit)
        self.simulator = Simulator(plant=self.plant)

    def __call__(self, pars):

        controller = ILQRMPCCPPController(mass=self.mass,
                                          length=self.length,
                                          com=self.com,
                                          damping=self.damping,
                                          gravity=self.gravity,
                                          coulomb_fric=self.coulomb_fric,
                                          inertia=self.inertia,
                                          torque_limit=self.torque_limit)

        controller.set_start(self.start)
        controller.set_goal(self.goal)
        controller.set_parameters(N=self.N,
                                  dt=self.dt,
                                  max_iter=self.max_iter,
                                  regu_init=self.regu_init,
                                  max_regu=self.max_regu,
                                  min_regu=self.min_regu,
                                  break_cost_redu=self.break_cost_redu,
                                  integrator=self.integrator)
        controller.set_cost_parameters_(np.asarray(pars)*self.par_prefactors)
        controller.load_init_traj(csv_path=self.csv_path)
        controller.init()

        T, X, U = controller.get_init_trajectory()

        time = 0.0
        self.simulator.set_state(time, self.start)
        # sim.set_state(time, 0.1*np.random.uniform(size=4))
        self.simulator.reset_data_recorder()
        t, x = self.simulator.get_state()
        # closest_state = np.copy(x0)

        closest_dist = 99999.
        max_traj_dist = 0.0
        smoothness = 0.0
        last_u = np.asarray(controller.get_control_output(self.start))

        while (time <= self.t_final):
            i = int(np.around(min(time, self.t_final) / self.dt))
            tau = controller.get_control_output(x)
            self.simulator.step(tau, self.dt, integrator=self.integrator)
            t, x = self.simulator.get_state()

            y = wrap_angles_top(x)
            # y = np.copy(x)

            time = np.copy(t)
            goal_dist = np.max(np.abs(y - self.goal))
            traj_dist = np.max(np.abs(wrap_angles_diff(y - X[min(i+1, self.N-2)])))
            if goal_dist < closest_dist:
                closest_dist = np.copy(goal_dist)
                # closest_state = np.copy(y)
            if traj_dist > max_traj_dist:
                max_traj_dist = np.copy(traj_dist)

            u_jump = np.max(np.abs(last_u - np.asarray(tau)))
            if u_jump > smoothness:
                smoothness = u_jump
            last_u = np.copy(tau)

            # if np.max(np.abs(y - goal) - goal_accuracy) < 0:
            #     break

        loss = (self.loss_weights[0]*goal_dist +
                self.loss_weights[1]*max_traj_dist +
                self.loss_weights[2]*smoothness)

        return float(loss)
