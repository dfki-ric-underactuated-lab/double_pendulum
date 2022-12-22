import numpy as np

from double_pendulum.utils.wrap_angles import wrap_angles_top, wrap_angles_diff
from double_pendulum.controller.lqr.roa.ellipsoid import (quadForm,
                                                          sampleFromEllipsoid,
                                                          volEllipsoid)
from double_pendulum.utils.plotting import plot_timeseries


def check_x0(simulator, controller, x0, dt, t_final,
             integrator="runge_kutta",
             goal=np.array([np.pi, 0., 0., 0.]),
             eps=np.array([1., 1., 10.0, 10.0])):

    controller.init()
    simulator.reset_data_recorder()
    T, X, U = simulator.simulate(
                t0=0.0, x0=x0,
                tf=t_final, dt=dt, controller=controller,
                integrator=integrator)

    valid = True
    for x in X:
        err = np.array(x) - goal
        err = wrap_angles_diff(err)
        err = np.abs(err)
        if np.max(err - eps) > 0.:
            valid = False
            #print("stabilization failed", x, err, err-eps)
            break
    return valid


def compute_roa_prob(simulator, controller, dt, t_final,
                     integrator="runge_kutta",
                     goal=np.array([np.pi, 0., 0., 0.]),
                     eps=np.array([1., 1., 10.0, 10.0]),
                     n_iter=1000, n_check_sims=5,
                     xbar_max=np.array([1., 1., 1., 1.])):

    S = np.asarray(controller.S)
    #rho = float(quadForm(S, xbar_max))
    rho = 1.

    for i in range(n_iter):
        x0_err = sampleFromEllipsoid(S, rho)
        x0 = wrap_angles_top(x0_err - goal)

        valid = True
        for j in range(n_check_sims):
            valid = check_x0(simulator,
                             controller,
                             x0,
                             dt,
                             t_final,
                             integrator,
                             goal,
                             eps)
            if not valid:
                break

        if not valid:
            # shrink ellipse
            rho = quadForm(S, x0_err)
        #print(rho, x0)
    vol = volEllipsoid(rho, S)
    return vol


class roa_prob_loss():
    def __init__(self,
                 simulator, controller, dt, t_final, integrator,
                 bounds,
                 goal=np.array([np.pi, 0., 0., 0.]),
                 eps=np.array([1., 1., 10.0, 10.0]),
                 n_iter=1000, n_check_sims=5):

        self.simulator = simulator
        self.controller = controller
        self.dt = dt
        self.t_final = t_final
        self.integrator = integrator
        self.bounds = bounds
        self.goal = goal
        self.eps = eps
        self.n_iter = n_iter
        self.n_check_sims = n_check_sims

    def __call__(self, costs):

        real_costs = self.rescale_pars(costs)

        self.controller.set_cost_parameters_(real_costs)
        self.controller.init()

        vol = compute_roa_prob(self.simulator,
                               self.controller,
                               self.dt,
                               self.t_final,
                               self.integrator,
                               self.goal,
                               self.eps,
                               self.n_iter,
                               self.n_check_sims)

        return -vol

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
