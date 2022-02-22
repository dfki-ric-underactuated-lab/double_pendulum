import numpy as np

from roatools.obj_fcts import caprr_coopt_interface


class roa_lqrpar_lossfunc():
    def __init__(self,
                 par_prefactors=[100, 100, 100, 100, 10],
                 roa_backend="sos"):

        self.par_prefactors = np.asarray(par_prefactors)
        self.roa_backend = roa_backend

    def set_model_parameters(self,
                             mass=[0.608, 0.630],
                             length=[0.3, 0.2],
                             com=[0.275, 0.166],
                             damping=[0.081, 0.0],
                             coulomb_fric=[0.093, 0.186],
                             gravity=9.81,
                             inertia=[0.05472, 0.02522],
                             torque_limit=[0.0, 6.0]):

        self.design_params = {"m": mass,
                              "l": length,
                              "lc": com,
                              "b": damping,
                              "fc": coulomb_fric,
                              "g": gravity,
                              "I": inertia,
                              "tau_max": torque_limit}

    def __call__(self, pars):
        # p = np.asarray(pars)*self.par_prefactors
        p = self.rescale_pars(pars)

        Q = np.diag((p[0], p[1], p[2], p[3]))
        R = np.diag((p[4], p[4]))

        roa_calc = caprr_coopt_interface(self.design_params, Q, R,
                                         backend=self.roa_backend)
        loss = roa_calc.lqr_param_opt_obj(Q, R)
        return loss

    def rescale_pars(self, pars):
        p = np.copy(pars)
        p *= self.par_prefactors
        return p


class roa_modelpar_lossfunc():
    def __init__(self,
                 par_prefactors=[1., 1., 1.],
                 bounds=[[0.1, 1.0], [0.3, 1.0], [0.1, 1.0]],
                 roa_backend="sos"):

        self.par_prefactors = np.asarray(par_prefactors)
        self.roa_backend = roa_backend
        self.bounds = np.asarray(bounds)

    def set_model_parameters(self,
                             mass=[0.608, 0.630],
                             length=[0.3, 0.2],
                             com=[0.275, 0.166],
                             damping=[0.081, 0.0],
                             coulomb_fric=[0.093, 0.186],
                             gravity=9.81,
                             inertia=[0.05472, 0.02522],
                             torque_limit=[0.0, 6.0]):

        # mass2, length1 and length2 will be overwritten during optimization

        self.design_params = {"m": mass,
                              "l": length,
                              "lc": com,
                              "b": damping,
                              "fc": coulomb_fric,
                              "g": gravity,
                              "I": inertia,
                              "tau_max": torque_limit}

    def set_cost_parameters(self,
                            p1p1_cost=1.,
                            p2p2_cost=1.,
                            v1v1_cost=1.,
                            v2v2_cost=1.,
                            p1p2_cost=0.,
                            v1v2_cost=0.,
                            p1v1_cost=0.,
                            p1v2_cost=0.,
                            p2v1_cost=0.,
                            p2v2_cost=0.,
                            u1u1_cost=0.01,
                            u2u2_cost=0.01,
                            u1u2_cost=0.):
        # state cost matrix
        self.Q = np.array([[p1p1_cost, p1p2_cost, p1v1_cost, p1v2_cost],
                           [p1p2_cost, p2p2_cost, p2v1_cost, p2v2_cost],
                           [p1v1_cost, p2v1_cost, v1v1_cost, v1v2_cost],
                           [p1v2_cost, p2v2_cost, v1v2_cost, v2v2_cost]])

        # control cost matrix
        self.R = np.array([[u1u1_cost, u1u2_cost], [u1u2_cost, u2u2_cost]])

    def __call__(self, pars):
        # p = np.asarray(pars)*self.par_prefactors
        # p = self.bounds.T[0] + p*(self.bounds.T[1]-self.bounds.T[0])

        p = self.rescale_pars(pars)

        model_par = [p[0], p[1], p[2]]  # m2, l1, l2

        roa_calc = caprr_coopt_interface(self.design_params, self.Q, self.R,
                                         backend=self.roa_backend)
        loss = roa_calc.design_opt_obj(model_par)
        return loss

    def rescale_pars(self, pars):
        p = np.copy(pars)
        p = self.bounds.T[0] + p*(self.bounds.T[1]-self.bounds.T[0])
        p *= self.par_prefactors
        p[-1] = p[-1]*(p[-2] - 0.1)
        return p


class roa_lqrandmodelpar_lossfunc():
    def __init__(self,
                 par_prefactors=[100, 100, 100, 100, 10, 1., 1., 1.],
                 bounds=[[0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.],
                         [0.1, 1.0], [0.3, 1.0], [0.1, 1.0]],
                 roa_backend="sos"):

        self.par_prefactors = np.asarray(par_prefactors)
        self.roa_backend = roa_backend
        self.bounds = np.asarray(bounds)

    def set_model_parameters(self,
                             mass=[0.608, 0.630],
                             length=[0.3, 0.2],
                             com=[0.275, 0.166],
                             damping=[0.081, 0.0],
                             coulomb_fric=[0.093, 0.186],
                             gravity=9.81,
                             inertia=[0.05472, 0.02522],
                             torque_limit=[0.0, 6.0]):

        # mass2, length1 and length2 will be overwritten during optimization

        self.design_params = {"m": mass,
                              "l": length,
                              "lc": com,
                              "b": damping,
                              "fc": coulomb_fric,
                              "g": gravity,
                              "I": inertia,
                              "tau_max": torque_limit}

    def __call__(self, pars):
        # p = np.asarray(pars)*self.par_prefactors
        # p = self.bounds.T[0] + p*(self.bounds.T[1]-self.bounds.T[0])

        p = self.rescale_pars(pars)

        Q = np.diag((p[0], p[1], p[2], p[3]))
        R = np.diag((p[4], p[4]))

        model_par = [p[5], p[6], p[7]]  # m2, l1, l2

        roa_calc = caprr_coopt_interface(self.design_params, Q, R,
                                         backend=self.roa_backend)
        loss = roa_calc.design_and_lqr_opt_obj(Q, R, model_par)
        return loss

    def rescale_pars(self, pars):
        p = np.copy(pars)
        p = self.bounds.T[0] + p*(self.bounds.T[1]-self.bounds.T[0])
        p *= self.par_prefactors
        p[-1] = p[-1]*(p[-2] - 0.1)
        return p
