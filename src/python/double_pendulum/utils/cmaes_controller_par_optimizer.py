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


def plot_cma_altopt_results(data_path, save_to=None, show=False):
    dirs = os.listdir(data_path)

    sequence = []

    ev_lists = []
    volume_lists = []

    for i, d in enumerate(sorted(dirs)):
        path = os.path.join(data_path, d)
        fit_path = os.path.join(path, "outcmaes", "fit.dat")

        if "design" in d or "lqr" in d:
            data = np.loadtxt(fit_path, skiprows=1)
            evaluations = data.T[1]
            best = data.T[4]

            ev_lists.append(evaluations)
            volume_lists.append(-1.*best)


            if "design" in d:
                sequence.append("d")
                # final_model_par = np.loadtxt(os.path.join(path, "model_par.csv"))
            elif "lqr" in d:
                sequence.append("c")
                # final_lqr_par = np.loadtxt(os.path.join(path, "controller_par.csv"))
        elif "init" in d:
            sequence.append("i")
            ev_lists.append(np.array([0]))
            volume_lists.append(np.array([np.loadtxt(os.path.join(path, "vol"))]))

    evs_sum = 0

    lqr_label_plotted = False
    model_label_plotted = False
    fig = plt.figure(figsize=(16, 12))
    for i, s in enumerate(sequence):
        if s == "d":
            color = "red"
            label = "design opt"
        if s == "c":
            color = "blue"
            label = "lqr opt"
        if i > 0:
            evs = np.asarray(ev_lists[i]) + evs_sum  # add evs from previous opts
            x = np.insert(evs, 0, evs_sum)  # insert last data point
            y = np.insert(volume_lists[i], 0, volume_lists[i-1][-1])
            if s=="d" and not model_label_plotted:
                plt.plot(x, y, color=color, label=label)
                model_label_plotted = True
            elif s=="c" and not lqr_label_plotted:
                plt.plot(x, y, color=color, label=label)
                lqr_label_plotted = True
            else:
                plt.plot(x, y, color=color)
        else:
            plt.plot(ev_lists[i], volume_lists[i], "ro", color="black")
        evs_sum += ev_lists[i][-1]

    # plt.text(0., 3., "final parameters:\nModel parameters:\n"+
    #                    str(final_model_par)+
    #                    "\nLQR parameters:\n"+str(final_lqr_par),
    #                    fontsize=16)

    plt.xlabel("Evaluations", fontsize=24)
    plt.ylabel("ROA Volume", fontsize=24)
    plt.legend(loc="upper left", fontsize=24)
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
