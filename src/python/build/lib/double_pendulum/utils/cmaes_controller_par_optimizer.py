import numpy as np

from double_pendulum.utils.wrap_angles import wrap_angles_top


# def swingup_test(pars,
#                  simulator,
#                  controller,
#                  t_final,
#                  dt,
#                  x0,
#                  integrator,
#                  goal,
#                  goal_accuracy,
#                  par_prefactors):  # kpos_pre, kvel_pre, ken_pre):
# 
#     # controller.set_cost_parameters_([kpos_pre*pars[0],
#     #                                  kvel_pre*pars[1],
#     #                                  ken_pre*pars[2]])
#     controller.set_cost_parameters_(par_prefactors*pars)
# 
#     controller.init()
# 
#     time = 0.0
#     simulator.set_state(time, x0)
#     # sim.set_state(time, 0.1*np.random.uniform(size=4))
#     simulator.reset_data_recorder()
#     t, x = simulator.get_state()
#     # closest_state = np.copy(x0)
#     closest_dist = 99999.
# 
#     while (time <= t_final):
#         tau = controller.get_control_output(x)
#         simulator.step(tau, dt, integrator=integrator)
#         t, x = simulator.get_state()
# 
#         y = wrap_angles_top(x)
# 
#         time = np.copy(t)
#         goal_dist = np.max(np.abs(y - goal))
#         if goal_dist < closest_dist:
#             closest_dist = np.copy(goal_dist)
#             # closest_state = np.copy(y)
#         # if np.max(np.abs(y - goal) - goal_accuracy) < 0:
#         #     break
#     return float(closest_dist)


class swingup_loss():
    def __init__(self,
                 simulator,
                 controller,
                 t_final,
                 dt,
                 x0,
                 integrator,
                 goal,
                 goal_accuracy,
                 bounds,
                 repetitions=1,
                 loss_weights=[1., 0., 0.]):

        self.simulator = simulator
        self.controller = controller
        self.t_final = t_final
        self.dt = dt
        self.x0 = np.asarray(x0)
        self.integrator = integrator
        self.goal = np.asarray(goal)
        self.goal_accuracy = goal_accuracy
        # self.par_prefactors = np.asarray(par_prefactors)
        self.bounds = bounds
        self.repetitions = repetitions
        self.weights = loss_weights

    def rescale_pars(self, pars):
        # [0, 1] -> real values
        p = np.copy(pars)
        p = self.bounds.T[0] + p*(self.bounds.T[1]-self.bounds.T[0])
        return p

    def unscale_pars(self, pars):
        # real values -> [0, 1]
        p = np.copy(pars)
        p = (p - self.bounds.T[0]) / (self.bounds.T[1]-self.bounds.T[0])
        return p

    def __call__(self, pars):
        p = self.rescale_pars(pars)
        self.controller.set_cost_parameters_(p)
        self.controller.init()

        dists = []
        smoothnesses = []
        max_vels = []
        for i in range(self.repetitions):
            time = 0.0
            self.simulator.set_state(time, self.x0)
            # sim.set_state(time, 0.1*np.random.uniform(size=4))
            self.simulator.reset_data_recorder()
            t, x = self.simulator.get_state()
            # closest_state = np.copy(x0)

            closest_dist = np.inf
            last_tau = np.asarray([0., 0.])
            max_vel = 0.
            s = 0.

            while (time <= self.t_final):
                tau = self.controller.get_control_output(x)
                self.simulator.step(tau, self.dt, integrator=self.integrator)
                t, x = self.simulator.get_state()

                y = wrap_angles_top(x)
                # y = np.copy(x)

                time = np.copy(t)

                tau_diff = np.abs(last_tau[0] - tau[0]) + np.abs(last_tau[1] - tau[0])
                if tau_diff > s:
                    s = tau_diff
                last_tau = tau

                if np.abs(y[2]) > max_vel:
                    max_vel = np.abs(y[2])
                if np.abs(y[3]) > max_vel:
                    max_vel = np.abs(y[3])

                goal_dist = np.max(np.abs(y - self.goal))
                if goal_dist < closest_dist:
                    closest_dist = np.copy(goal_dist)
                    # closest_state = np.copy(y)
                # if np.max(np.abs(y - goal) - goal_accuracy) < 0:
                #     break
            dists.append(float(closest_dist))
            smoothnesses.append(s)
            max_vels.append(max_vel)
        dist_avg = np.mean(dists)
        smoothness_avg = np.mean(smoothnesses)
        max_vel_avg = np.mean(max_vels)
        loss = float(self.weights[0]*dist_avg +
                     self.weights[1]*smoothness_avg +
                     self.weights[2]*max_vel_avg)
        return loss


class traj_opt_loss():
    def __init__(self,
                 traj_opt,
                 goal,
                 bounds,
                 repetitions=1,
                 loss_weights=[1., 0., 0.]):

        self.traj_opt = traj_opt
        self.goal = goal
        self.bounds = bounds
        self.repetitions = repetitions
        self.weights = loss_weights

    def rescale_pars(self, pars):
        # [0, 1] -> real values
        p = np.copy(pars)
        p = self.bounds.T[0] + p*(self.bounds.T[1]-self.bounds.T[0])
        return p

    def unscale_pars(self, pars):
        # real values -> [0, 1]
        p = np.copy(pars)
        p = (p - self.bounds.T[0]) / (self.bounds.T[1]-self.bounds.T[0])
        return p

    def __call__(self, pars):
        p = self.rescale_pars(pars)
        self.traj_opt.set_cost_parameters_(p)
        dists = []
        smoothnesses = []
        max_vels = []
        for i in range(self.repetitions):
            T, X, U = self.traj_opt.compute_trajectory()
            y = wrap_angles_top(X[-1])
            dists.append(np.max(np.abs(y - self.goal)))
            max_vels.append(np.max(np.abs(X.T[2:])))
            U1 = np.asarray(U).T[0]
            U2 = np.asarray(U).T[1]
            s = np.max(np.abs(np.diff(U1))) + np.max(np.abs(np.diff(U2)))
            smoothnesses.append(s)
        goal_dist = np.mean(dists)
        smoothness = np.mean(smoothnesses)
        max_vel = np.mean(max_vels)
        loss = float(self.weights[0]*goal_dist +
                     self.weights[1]*smoothness +
                     self.weights[2]*max_vel)
        return loss
