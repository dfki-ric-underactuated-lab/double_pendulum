import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.plant import DoublePendulumPlant
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.controller.lqr.roa.ellipsoid import (
    quadForm,
    sampleFromEllipsoid,
    volEllipsoid,
)
from double_pendulum.controller.lqr.roa.roa_check import lqr_check_isnotNaN
from double_pendulum.controller.lqr.roa.roa_estimation import (
    estimate_roa_najafi,
    # estimate_roa_najafi_direct,
    estimate_roa_probabilistic,
    bisect_and_verify,
    estimate_roa_sos_constrained,
)


class caprr_coopt_interface:
    def __init__(
        self,
        model_par,
        goal,
        Q,
        R,
        backend="sos_con",
        log_obj_fct=False,
        verbose=False,
        estimate_clbk=None,
        najafi_evals=10000,
        robot="acrobot",
    ):
        """
        Object for design/parameter co-optimization.
        It helps keeping track of design parameters during cooptimization.

        Intended usage:
        call `design_opt_obj` within the design optimization

        `backend` must be one of the following:
        - `sos`:        unconstrained SOS problem.
                        Fastest, but maybe not a good approximation for the actual dynamics
        - `sos_con`:    SOS problem that models a constrained dynamics.
                        Best trade-off between time and precision for the RoA estimation.
        - `sos_eq`:     unconstrained SOS problem with the so called equality constrained formulation.
                        Obtains the same result of the previous one if the closed loop dynamics is not too bad.
                        It seems to be the slowest SOS method.
        - `prob`(TODO):       probabilistic simulation based
        - `najafi`:     probabilistic evaluation of Lyapunov function only.
                        The computational time is very long for obtaining a good estimation.

        `robot` must be `acrobot` or `pendubot` depending on the underactuated system that we are considering.
        """
        # robot type
        self.robot = robot

        self.goal = goal

        # design params and controller gains. these are mutable
        self.model_par = model_par

        # already synthesize controller here to get self.S and self.K
        self._update_lqr(Q, R)

        self.backend = backend

        # history to store dicts with complete design parameters
        self.param_hist = []

        if self.backend == "sos":
            self.verification_hyper_params = {
                "taylor_deg": 3,
                "lambda_deg": 4,
                "mode": 0,
            }
        if self.backend == "sos_con":
            self.verification_hyper_params = {
                "taylor_deg": 3,
                "lambda_deg": 2,
                "mode": 2,
            }
        if self.backend == "sos_eq":
            self.verification_hyper_params = {
                "taylor_deg": 3,
                "lambda_deg": 3,
                "mode": 2,
            }

        if self.backend == "prob":
            pass

        self.verbose = verbose

        # callback function called from the estimate.
        # user can define this in order to get insight during optimization
        self.estimate_clbk = estimate_clbk

        # number of evals for the najafi method
        self.najafi_evals = najafi_evals

    def combined_opt_obj(self, y_comb):
        """
        y_comb contains the following entries (in this order):
        m2,l1,l2,q11,q22,q33,q44,r11,r22
        """
        m1 = self.model_par["m"][0]
        m2 = y_comb[0]
        l1 = y_comb[1]
        l2 = y_comb[2]

        # update new design parameters
        self.model_par.m[1] = m2
        self.model_par.l[0] = l1
        self.model_par.l[1] = l2
        self.model_par.r[0] = l1
        self.model_par.r[1] = l2
        self.model_par.I[0] = m1 * l1**2
        self.model_par.I[1] = m2 * l2**2

        Q = np.diag((y_comb[3], y_comb[4], y_comb[5], y_comb[6]))
        R = np.diag((y_comb[7], y_comb[8]))

        self._update_lqr(Q, R)

        vol, _, _ = self._estimate()

        if self.verbose:
            print(self.model_par)

        return -vol

    def combined_reduced_opt_obj(self, y_comb):
        """
        y_comb contains the following entries (in this order):
        m2,l1,l2,q11&q22,q33&q44,r11&r22
        """
        m1 = self.model_par["m"][0]
        m2 = y_comb[0]
        l1 = y_comb[1]
        l2 = y_comb[2]

        # update new design parameters
        self.model_par.m[1] = m2
        self.model_par.l[0] = l1
        self.model_par.l[1] = l2
        self.model_par.r[0] = l1
        self.model_par.r[1] = l2
        self.model_par.I[0] = m1 * l1**2
        self.model_par.I[1] = m2 * l2**2

        Q = np.diag((y_comb[3], y_comb[3], y_comb[4], y_comb[4]))

        R = np.diag((y_comb[5], y_comb[5]))

        self._update_lqr(Q, R)

        vol, _, _ = self._estimate()

        if self.verbose:
            print(self.model_par)

        return -vol

    def design_opt_obj(self, y):
        """
        objective function for design optimization
        """

        # update new design parameters
        self.model_par.m[1] = y[0]
        self.model_par.l[0] = y[1]
        self.model_par.l[1] = y[2]
        self.model_par.r[0] = y[1]
        self.model_par.r[1] = y[2]
        self.model_par.I[0] = self.model_par.m[0] * y[1] ** 2
        self.model_par.I[1] = y[0] * y[2] ** 2

        # update lqr for the new parameters. K and S are also computed here.
        # here this just recomputes S and K for the new design
        self._update_lqr(self.Q, self.R)

        vol, rho_f, S = self._estimate()

        return -vol

    def lqr_param_opt_obj(self, Q, R, verbose=False):

        self._update_lqr(Q, R)

        vol, _, _ = self._estimate()

        return -vol

    def lqr_param_reduced_opt_obj(self, y_comb, verbose=False):

        Q = np.diag((y_comb[0], y_comb[0], y_comb[1], y_comb[1]))
        R = np.diag((y_comb[2], y_comb[2]))

        self._update_lqr(Q, R)

        vol, _, _ = self._estimate()

        return -vol

    def design_and_lqr_opt_obj(self, Q, R, y, verbose=False):
        m1 = self.model_par.m[0]
        m2 = y[0]
        l1 = y[1]
        l2 = y[2]

        # update new design parameters
        self.model_par.m[1] = m2
        self.model_par.l[0] = l1
        self.model_par.l[1] = l2
        self.model_par.r[0] = l1
        self.model_par.r[1] = l2
        self.model_par.I[0] = m1 * l1**2
        self.model_par.I[1] = m2 * l2**2

        # update lqr. K and S are also contained in design_params
        self._update_lqr(Q, R)

        vol, _, _ = self._estimate()

        if verbose:
            print(self.model_par)

        return -vol

    def _estimate(self):

        rho_f = 0.0
        if self.backend == "sos" or self.backend == "sos_con":
            rho_f = bisect_and_verify(
                self.model_par,
                self.goal,
                self.S,
                self.K,
                self.robot,
                self.verification_hyper_params,
                verbose=self.verbose,
                rho_min=1e-10,
                rho_max=5,
                maxiter=15,
            )

        if self.backend == "sos_eq":
            rho_f = estimate_roa_sos_constrained(
                self.model_par,
                self.goal,
                self.S,
                self.K,
                self.robot,
                self.verification_hyper_params["taylor_deg"],
                self.verification_hyper_params["lambda_deg"],
                verbose=self.verbose,
            )

        if self.backend == "prob":
            plant = DoublePendulumPlant(model_pars=self.model_par)
            eminem = lqr_check_isnotNaN(plant, self.controller)
            conf = {
                "x0Star": self.goal,
                "S": self.S,
                "xBar0Max": np.array([+0.5, +0.0, 0.0, 0.0]),
                "nSimulations": 250,
            }

            # create estimation object
            estimator = estimate_roa_probabilistic(conf, eminem.sim_callback)
            # do the actual estimation
            rho_hist, simSuccesHist = estimator.doEstimate()
            rho_f = rho_hist[-1]

        if self.backend == "najafi":
            plant = DoublePendulumPlant(model_pars=self.model_par)
            rho_f = estimate_roa_najafi(
                plant, self.controller, self.goal, self.S, self.najafi_evals
            )
            # print(rho_f)

        vol = volEllipsoid(rho_f, self.S)

        if self.estimate_clbk is not None:
            self.estimate_clbk(self.model_par, rho_f, vol, self.Q, self.R)

        return vol, rho_f, self.S

    def _update_lqr(self, Q, R):
        self.controller = LQRController(model_pars=self.model_par)

        self.Q = Q
        self.R = R

        self.controller.set_cost_matrices(Q=self.Q, R=self.R)
        self.controller.init()
        self.K = np.array(self.controller.K)
        self.S = np.array(self.controller.S)

    # def log(self):
    #     pass

    # def set_design_params(self,params):
    #     pass

    # def set_lqr_params(self,Q,R):
    #     pass


class logger:
    def __init__(self):
        self.Q_log = []
        self.R_log = []
        self.m2_log = []
        self.l1_log = []
        self.l2_log = []
        self.vol_log = []

    def log_and_print_clbk(self, model_par, rho_f, vol, Q, R):
        print("design params: ")
        print(model_par)
        print("Q:")
        print(Q)
        print("R")
        print(R)
        print("rho final: " + str(rho_f))
        print("volume final: " + str(vol))
        print("")
        self.Q_log.append(Q)
        self.R_log.append(R)
        self.m2_log.append(model_par.m[1])
        self.l1_log.append(model_par.l[0])
        self.l2_log.append(model_par.l[1])
        self.vol_log.append(vol)
