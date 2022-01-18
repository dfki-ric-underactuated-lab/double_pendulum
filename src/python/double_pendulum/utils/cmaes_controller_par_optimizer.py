import numpy as np
import cma


def swingup_test(pars, simulator, plant, controller,
                 t_final, dt, x0,
                 integrator, goal, goal_accuracy,
                 kpos_pre, kvel_pre, ken_pre):

    controller.set_hyperpar(kpos=kpos_pre*pars[0],
                            kvel=kvel_pre*pars[1],
                            ken=ken_pre*pars[2])

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
        time = np.copy(t)
        goal_dist = np.max(np.abs(x - goal))
        if goal_dist < closest_dist:
            closest_dist = np.copy(goal_dist)
            # closest_state = np.copy(x)
        # if np.max(np.abs(x - goal) - goal_accuracy) < 0:
        #     break
    return float(closest_dist)


class swingup_loss():
    def __init__(self,
                 simulator,
                 plant,
                 controller,
                 t_final,
                 dt,
                 x0,
                 integrator,
                 goal,
                 goal_accuracy,
                 par_prefactors):

        self.simulator = simulator
        self.plant = plant
        self.controller = controller
        self.t_final = t_final
        self.dt = dt
        self.x0 = np.asarray(x0)
        self.integrator = integrator
        self.goal = np.asarray(goal)
        self.goal_accuracy = goal_accuracy
        self.par_prefactors = np.asarray(par_prefactors)

    def __call__(self, pars):
        p = self.par_prefactors*np.asarray(pars)
        self.controller.set_parameters(p)
        self.controller.init()

        time = 0.0
        self.simulator.set_state(time, self.x0)
        # sim.set_state(time, 0.1*np.random.uniform(size=4))
        self.simulator.reset_data_recorder()
        t, x = self.simulator.get_state()
        # closest_state = np.copy(x0)
        closest_dist = 99999.

        while (time <= self.t_final):
            tau = self.controller.get_control_output(x)
            self.simulator.step(tau, self.dt, integrator=self.integrator)
            t, x = self.simulator.get_state()
            time = np.copy(t)
            goal_dist = np.max(np.abs(x - self.goal))
            if goal_dist < closest_dist:
                closest_dist = np.copy(goal_dist)
                # closest_state = np.copy(x)
            # if np.max(np.abs(x - goal) - goal_accuracy) < 0:
            #     break
        return float(closest_dist)


def cma_par_optimization(loss_func, init_pars, bounds):
    es = cma.CMAEvolutionStrategy(init_pars,
                                  0.4,
                                  {'bounds': bounds,
                                   'verbose': -3,
                                   'popsize_factor': 3})
    es.optimize(loss_func)
    return es.result.xbest
