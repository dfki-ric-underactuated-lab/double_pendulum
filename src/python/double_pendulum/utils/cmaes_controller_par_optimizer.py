import os
import numpy as np
import cma
from cma.fitness_transformations import EvalParallel2
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from double_pendulum.utils.wrap_angles import wrap_angles_top


def swingup_test(pars,
                 simulator,
                 controller,
                 t_final,
                 dt,
                 x0,
                 integrator,
                 goal,
                 goal_accuracy,
                 par_prefactors):  # kpos_pre, kvel_pre, ken_pre):

    # controller.set_cost_parameters_([kpos_pre*pars[0],
    #                                  kvel_pre*pars[1],
    #                                  ken_pre*pars[2]])
    controller.set_cost_parameters_(par_prefactors*pars)

    controller.init()

    time = 0.0
    simulator.set_state(time, x0)
    # sim.set_state(time, 0.1*np.random.uniform(size=4))
    simulator.reset_data_recorder()
    t, x = simulator.get_state()
    # closest_state = np.copy(x0)
    closest_dist = 99999.

    while (time <= t_final):
        tau = controller.get_control_output(x)
        simulator.step(tau, dt, integrator=integrator)
        t, x = simulator.get_state()

        y = wrap_angles_top(x)

        time = np.copy(t)
        goal_dist = np.max(np.abs(y - goal))
        if goal_dist < closest_dist:
            closest_dist = np.copy(goal_dist)
            # closest_state = np.copy(y)
        # if np.max(np.abs(y - goal) - goal_accuracy) < 0:
        #     break
    return float(closest_dist)


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
                 par_prefactors,
                 repetitions=1,
                 loss_weights=[1.0, 0.0]):

        self.simulator = simulator
        self.controller = controller
        self.t_final = t_final
        self.dt = dt
        self.x0 = np.asarray(x0)
        self.integrator = integrator
        self.goal = np.asarray(goal)
        self.goal_accuracy = goal_accuracy
        self.par_prefactors = np.asarray(par_prefactors)
        self.repetitions = repetitions
        self.weights = loss_weights

    def __call__(self, pars):
        p = self.par_prefactors*np.asarray(pars)
        self.controller.set_cost_parameters_(p)
        self.controller.init()

        dists = []
        smoothnesses = []
        for i in range(self.repetitions):
            time = 0.0
            self.simulator.set_state(time, self.x0)
            # sim.set_state(time, 0.1*np.random.uniform(size=4))
            self.simulator.reset_data_recorder()
            t, x = self.simulator.get_state()
            # closest_state = np.copy(x0)
            closest_dist = 99999.
            s = 0.

            last_tau = np.asarray([0., 0.])
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

                goal_dist = np.max(np.abs(y - self.goal))
                if goal_dist < closest_dist:
                    closest_dist = np.copy(goal_dist)
                    # closest_state = np.copy(y)
                # if np.max(np.abs(y - goal) - goal_accuracy) < 0:
                #     break
            dists.append(float(closest_dist))
            smoothnesses.append(s)
        goal_dist = np.mean(dists)
        smoothness = np.mean(smoothnesses)
        loss = float(self.weights[0]*goal_dist + self.weights[1]*smoothness)
        return loss


class traj_opt_loss():
    def __init__(self,
                 traj_opt,
                 goal,
                 par_prefactors,
                 repetitions=1,
                 loss_weights=[1.0, 0.0]):

        self.traj_opt = traj_opt
        self.goal = goal
        self.par_prefactors = par_prefactors
        self.repetitions = repetitions
        self.weights = loss_weights

    def __call__(self, pars):
        self.traj_opt.set_cost_parameters_(self.par_prefactors*pars)
        dists = []
        smoothnesses = []
        for i in range(self.repetitions):
            T, X, U = self.traj_opt.compute_trajectory()
            y = wrap_angles_top(X[-1])
            dists.append(np.max(np.abs(y - self.goal)))
            U1 = np.asarray(U).T[0]
            U2 = np.asarray(U).T[1]
            s = np.max(np.abs(np.diff(U1))) + np.max(np.abs(np.diff(U2)))
            smoothnesses.append(s)
        goal_dist = np.mean(dists)
        smoothness = np.mean(smoothnesses)
        loss = float(self.weights[0]*goal_dist + self.weights[1]*smoothness)
        return loss


def cma_par_optimization(loss_func, init_pars, bounds,
                         save_dir="outcmaes/",
                         popsize_factor=3,
                         maxfevals=10000,
                         tolfun=1e-11,
                         tolx=1e-11,
                         tolstagnation=100,
                         num_proc=0):
    if save_dir[-1] != "/":
        sd = save_dir + "/"
    else:
        sd = save_dir
    es = cma.CMAEvolutionStrategy(init_pars,
                                  0.4,
                                  {'bounds': bounds,
                                   'verbose': -3,
                                   'popsize_factor': popsize_factor,
                                   'verb_filenameprefix': sd,
                                   'maxfevals': maxfevals,
                                   'tolfun': tolfun,
                                   'tolx': tolx,
                                   'tolstagnation': tolstagnation})

    with EvalParallel2(loss_func, num_proc) as eval_all:
        while not es.stop():
            X = es.ask()
            es.tell(X, eval_all(X))
            es.disp()
            es.logger.add()  # doctest:+ELLIPSIS

    #es.optimize(loss_func)
    return es.result.xbest


def plot_cma_results(data_path, sign=1., save_to=None, show=False):
    fit_path = os.path.join(data_path, "fit.dat")

    data = np.loadtxt(fit_path, skiprows=1)

    evaluations = data.T[1]
    best = data.T[4]*sign

    plt.plot(evaluations, best)
    plt.xlabel("Evaluations")
    plt.ylabel("ROA Volume")
    if not (save_to is None):
        plt.savefig(save_to)
    if show:
        plt.show()
    plt.close()


def scipy_par_optimization(loss_func,
                           init_pars,
                           bounds,
                           method="Nelder-Mead"):

    res = minimize(fun=loss_func,
                   x0=init_pars,
                   method=method,
                   bounds=bounds)

    return res.x
