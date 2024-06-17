import os
import time
import numpy as np
import yaml

from double_pendulum.controller.lqr.roa.coopt_interface import caprr_coopt_interface
from double_pendulum.utils.optimization import cma_optimization  # , plot_cma_results


class roa_lqrpar_lossfunc:
    def __init__(
        self,
        par_prefactors=[100, 100, 100, 100, 10],
        bounds=[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
        roa_backend="sos",
        najafi_evals=100000,
        robot="acrobot",
    ):
        self.par_prefactors = np.asarray(par_prefactors)
        self.roa_backend = roa_backend
        self.bounds = np.asarray(bounds)
        self.najafi_evals = najafi_evals
        self.robot = robot

    def set_model_parameters(
        self,
        mass=[0.608, 0.630],
        length=[0.3, 0.2],
        com=[0.275, 0.166],
        damping=[0.081, 0.0],
        coulomb_fric=[0.093, 0.186],
        gravity=9.81,
        inertia=[0.05472, 0.02522],
        torque_limit=[0.0, 5.0],
    ):
        self.design_params = {
            "m": mass,
            "l": length,
            "lc": com,
            "b": damping,
            "fc": coulomb_fric,
            "g": gravity,
            "I": inertia,
            "tau_max": torque_limit,
        }

    def __call__(self, pars):
        # p = np.asarray(pars)*self.par_prefactors
        p = self.rescale_pars(pars)

        Q = np.diag((p[0], p[1], p[2], p[3]))
        R = np.diag((p[4], p[4]))

        roa_calc = caprr_coopt_interface(
            self.design_params,
            Q,
            R,
            backend=self.roa_backend,
            najafi_evals=self.najafi_evals,
            robot=self.robot,
        )
        loss = roa_calc.lqr_param_opt_obj(Q, R)
        return loss

    def rescale_pars(self, pars):
        p = np.copy(pars)
        p = self.bounds.T[0] + p * (self.bounds.T[1] - self.bounds.T[0])
        p *= self.par_prefactors
        return p

    def unscale_pars(self, pars):
        p = np.copy(pars)
        p /= self.par_prefactors
        p = (p - self.bounds.T[0]) / (self.bounds.T[1] - self.bounds.T[0])
        return p


class roa_modelpar_lossfunc:
    def __init__(
        self,
        par_prefactors=[1.0, 1.0, 1.0],
        bounds=[[0.1, 1.0], [0.3, 1.0], [0.1, 1.0]],
        roa_backend="sos",
        najafi_evals=100000,
        robot="acrobot",
    ):
        self.par_prefactors = np.asarray(par_prefactors)
        self.roa_backend = roa_backend
        self.bounds = np.asarray(bounds)
        self.najafi_evals = najafi_evals
        self.robot = robot

    def set_model_parameters(
        self,
        mass=[0.608, 0.630],
        length=[0.3, 0.2],
        com=[0.275, 0.166],
        damping=[0.081, 0.0],
        coulomb_fric=[0.093, 0.186],
        gravity=9.81,
        inertia=[0.05472, 0.02522],
        torque_limit=[0.0, 5.0],
    ):
        # mass2, length1 and length2 will be overwritten during optimization

        self.design_params = {
            "m": mass,
            "l": length,
            "lc": com,
            "b": damping,
            "fc": coulomb_fric,
            "g": gravity,
            "I": inertia,
            "tau_max": torque_limit,
        }

    def set_cost_parameters(
        self,
        p1p1_cost=1.0,
        p2p2_cost=1.0,
        v1v1_cost=1.0,
        v2v2_cost=1.0,
        p1p2_cost=0.0,
        v1v2_cost=0.0,
        p1v1_cost=0.0,
        p1v2_cost=0.0,
        p2v1_cost=0.0,
        p2v2_cost=0.0,
        u1u1_cost=0.01,
        u2u2_cost=0.01,
        u1u2_cost=0.0,
    ):
        # state cost matrix
        self.Q = np.array(
            [
                [p1p1_cost, p1p2_cost, p1v1_cost, p1v2_cost],
                [p1p2_cost, p2p2_cost, p2v1_cost, p2v2_cost],
                [p1v1_cost, p2v1_cost, v1v1_cost, v1v2_cost],
                [p1v2_cost, p2v2_cost, v1v2_cost, v2v2_cost],
            ]
        )

        # control cost matrix
        self.R = np.array([[u1u1_cost, u1u2_cost], [u1u2_cost, u2u2_cost]])

    def __call__(self, pars):
        # p = np.asarray(pars)*self.par_prefactors
        # p = self.bounds.T[0] + p*(self.bounds.T[1]-self.bounds.T[0])

        p = self.rescale_pars(pars)

        model_par = [p[0], p[1], p[2]]  # m2, l1, l2

        roa_calc = caprr_coopt_interface(
            self.design_params,
            self.Q,
            self.R,
            backend=self.roa_backend,
            najafi_evals=self.najafi_evals,
            robot=self.robot,
        )
        loss = roa_calc.design_opt_obj(model_par)
        return loss

    def rescale_pars(self, pars):
        p = np.copy(pars)
        p = self.bounds.T[0] + p * (self.bounds.T[1] - self.bounds.T[0])
        p *= self.par_prefactors
        # p[-1] = p[-1]*(p[-2] + 0.1)
        return p

    def unscale_pars(self, pars):
        p = np.copy(pars)

        # p[-1] = p[-1]/(p[-2] + 0.1)
        p /= self.par_prefactors
        p = (p - self.bounds.T[0]) / (self.bounds.T[1] - self.bounds.T[0])
        return p


class roa_lqrandmodelpar_lossfunc:
    def __init__(
        self,
        par_prefactors=[100, 100, 100, 100, 10, 1.0, 1.0, 1.0],
        bounds=[
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.1, 1.0],
            [0.3, 1.0],
            [0.1, 1.0],
        ],
        roa_backend="sos",
        najafi_evals=1000,
        robot="acrobot",
    ):
        self.par_prefactors = np.asarray(par_prefactors)
        self.roa_backend = roa_backend
        self.bounds = np.asarray(bounds)
        self.najafi_evals = najafi_evals
        self.robot = robot

    def set_model_parameters(
        self,
        mass=[0.608, 0.630],
        length=[0.3, 0.2],
        com=[0.275, 0.166],
        damping=[0.081, 0.0],
        coulomb_fric=[0.093, 0.186],
        gravity=9.81,
        inertia=[0.05472, 0.02522],
        torque_limit=[0.0, 6.0],
    ):
        # mass2, length1 and length2 will be overwritten during optimization

        self.design_params = {
            "m": mass,
            "l": length,
            "lc": com,
            "b": damping,
            "fc": coulomb_fric,
            "g": gravity,
            "I": inertia,
            "tau_max": torque_limit,
        }

    def __call__(self, pars):
        # p = np.asarray(pars)*self.par_prefactors
        # p = self.bounds.T[0] + p*(self.bounds.T[1]-self.bounds.T[0])

        p = self.rescale_pars(pars)

        Q = np.diag((p[0], p[1], p[2], p[3]))
        R = np.diag((p[4], p[4]))

        model_par = [p[5], p[6], p[7]]  # m2, l1, l2

        roa_calc = caprr_coopt_interface(
            self.design_params,
            Q,
            R,
            backend=self.roa_backend,
            najafi_evals=self.najafi_evals,
            robot=self.robot,
        )
        loss = roa_calc.design_and_lqr_opt_obj(Q, R, model_par)
        return loss

    def rescale_pars(self, pars):
        # [0, 1] -> real values
        p = np.copy(pars)
        p = self.bounds.T[0] + p * (self.bounds.T[1] - self.bounds.T[0])
        p *= self.par_prefactors
        # p[-1] = p[-1]*(p[-2] + 0.1)
        return p

    def unscale_pars(self, pars):
        # real values -> [0, 1]
        p = np.copy(pars)

        # p[-1] = p[-1]/(p[-2] + 0.1)
        p /= self.par_prefactors
        p = (p - self.bounds.T[0]) / (self.bounds.T[1] - self.bounds.T[0])
        return p


def calc_roa(
    c_par=[1.0, 1.0, 1.0, 1.0, 1.0],
    m_par=[0.63, 0.3, 0.2],
    roa_backend="najafi",
    najafi_evals=1000,
    robot="acrobot",
    save_dir="data/",
    plots=False,
):
    os.makedirs(save_dir)

    mass = [0.608, m_par[0]]
    length = [m_par[1], m_par[2]]
    com = [length[0], length[1]]
    damping = [0.0, 0.0]
    cfric = [0.0, 0.0]
    gravity = 9.81
    inertia = [mass[0] * length[0] ** 2, mass[1] * length[1] ** 2]
    if robot == "acrobot":
        torque_limit = [0.0, 5.0]
    if robot == "pendubot":
        torque_limit = [5.0, 0.0]

    goal = [np.pi, 0, 0, 0]

    design_params = {
        "m": mass,
        "l": length,
        "lc": com,
        "b": damping,
        "fc": cfric,
        "g": gravity,
        "I": inertia,
        "tau_max": torque_limit,
    }

    Q = np.diag((c_par[0], c_par[1], c_par[2], c_par[3]))
    R = np.diag((c_par[4], c_par[4]))

    roa_calc = caprr_coopt_interface(
        design_params=design_params,
        Q=Q,
        R=R,
        backend=roa_backend,
        najafi_evals=najafi_evals,
        robot=robot,
    )
    roa_calc._update_lqr(Q=Q, R=R)
    vol, rho_f, S = roa_calc._estimate()

    np.savetxt(os.path.join(save_dir, "rho"), [rho_f])
    np.savetxt(os.path.join(save_dir, "vol"), [vol])
    # np.savetxt(os.path.join(save_dir, "rhohist"), rhoHist)

    # if plots:
    #     plotEllipse(goal[0], goal[1], 0, 1, rho_f, S,
    #                 save_to=os.path.join(save_dir, "roaplot"),
    #                 show=False)

    np.savetxt(
        os.path.join(save_dir, "controller_par.csv"),
        [Q[0, 0], Q[1, 1], Q[2, 2], Q[3, 3], R[0, 0]],
    )

    par_dict = {
        "mass1": mass[0],
        "mass2": float(mass[1]),
        "length1": float(length[0]),
        "length2": float(length[1]),
        "com1": float(com[0]),
        "com2": float(com[1]),
        "inertia1": float(inertia[0]),
        "inertia2": float(inertia[1]),
        "damping1": damping[0],
        "damping2": damping[1],
        "coulomb_friction1": cfric[0],
        "coulomb_friction2": cfric[1],
        "gravity": gravity,
        "torque_limit1": torque_limit[0],
        "torque_limit2": torque_limit[1],
        "goal_pos1": goal[0],
        "goal_pos2": goal[1],
        "goal_vel1": goal[2],
        "goal_vel2": goal[3],
        "Q1": float(c_par[0]),
        "Q2": float(c_par[1]),
        "Q3": float(c_par[2]),
        "Q4": float(c_par[3]),
        "R": float(c_par[4]),
        "roa_beackend": roa_backend,
        "najafi_evaluations": najafi_evals,
    }

    with open(os.path.join(save_dir, "parameters.yml"), "w") as f:
        yaml.dump(par_dict, f)

    return vol


def roa_lqr_opt(
    model_pars=[0.63, 0.3, 0.2],
    init_pars=[1.0, 1.0, 1.0, 1.0, 1.0],
    par_prefactors=[20.0, 20.0, 10.0, 10.0, 10.0],
    bounds=[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
    maxfevals=1000,
    sigma0=0.4,
    roa_backend="najafi",
    najafi_evals=10000,
    robot="acrobot",
    save_dir="data/",
    plots=False,
    num_proc=0,
):
    mass = [0.608, model_pars[0]]
    length = [model_pars[1], model_pars[2]]
    com = [length[0], length[1]]
    damping = [0.0, 0.0]
    cfric = [0.0, 0.0]
    # damping = [0.081, 0.0]
    # cfric = [0.093, 0.186]
    gravity = 9.81
    inertia = [mass[0] * length[0] ** 2.0, mass[1] * length[1] ** 2.0]
    if robot == "acrobot":
        torque_limit = [0.0, 5.0]
    if robot == "pendubot":
        torque_limit = [5.0, 0.0]

    goal = [np.pi, 0, 0, 0]

    popsize_factor = 4

    # timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
    # save_dir = os.path.join("data", robot, "lqr", "roa_paropt", timestamp)
    os.makedirs(save_dir)

    # loss function setup
    loss_func = roa_lqrpar_lossfunc(
        par_prefactors=par_prefactors,
        bounds=bounds,
        roa_backend=roa_backend,
        najafi_evals=najafi_evals,
        robot=robot,
    )
    loss_func.set_model_parameters(
        mass=mass,
        length=length,
        com=com,
        damping=damping,
        gravity=gravity,
        coulomb_fric=cfric,
        inertia=inertia,
        torque_limit=torque_limit,
    )

    inits = loss_func.unscale_pars(init_pars)

    # optimization
    t0 = time.time()
    best_par = cma_optimization(
        loss_func=loss_func,
        init_pars=inits,
        bounds=[0, 1],
        save_dir=os.path.join(save_dir, "outcmaes"),
        sigma0=sigma0,
        popsize_factor=popsize_factor,
        maxfevals=maxfevals,
        num_proc=num_proc,
    )
    opt_time = (time.time() - t0) / 3600  # time in h

    best_par = loss_func.rescale_pars(best_par)
    print(best_par)

    np.savetxt(os.path.join(save_dir, "controller_par.csv"), best_par)
    np.savetxt(os.path.join(save_dir, "time.txt"), [opt_time])

    par_dict = {
        "optimization": "lqr",
        "mass1": mass[0],
        "mass2": float(mass[1]),
        "length1": float(length[0]),
        "length2": float(length[1]),
        "com1": float(com[0]),
        "com2": float(com[1]),
        "inertia1": float(inertia[0]),
        "inertia2": float(inertia[1]),
        "damping1": damping[0],
        "damping2": damping[1],
        "coulomb_friction1": cfric[0],
        "coulomb_friction2": cfric[1],
        "gravity": gravity,
        "torque_limit1": torque_limit[0],
        "torque_limit2": torque_limit[1],
        "goal_pos1": goal[0],
        "goal_pos2": goal[1],
        "goal_vel1": goal[2],
        "goal_vel2": goal[3],
        "Q_init1": float(init_pars[0]),
        "Q_init2": float(init_pars[1]),
        "Q_init3": float(init_pars[2]),
        "Q_init4": float(init_pars[3]),
        "R_init": float(init_pars[4]),
        "par_prefactors": par_prefactors,
        "popsize_factor": popsize_factor,
        "maxfevals": maxfevals,
        "roa_backend": roa_backend,
        "najafi_evals": najafi_evals,
        # "tolfun": tolfun,
        # "tolx": tolx,
        # "tolstagnation": tolstagnation
    }

    with open(os.path.join(save_dir, "parameters.yml"), "w") as f:
        yaml.dump(par_dict, f)

    # recalculate the roa for the best parameters and save plot
    best_Q = np.diag((best_par[0], best_par[1], best_par[2], best_par[3]))
    best_R = np.diag((best_par[4], best_par[4]))

    design_params = {
        "m": mass,
        "l": length,
        "lc": com,
        "b": damping,
        "fc": cfric,
        "g": gravity,
        "I": inertia,
        "tau_max": torque_limit,
    }

    roa_calc = caprr_coopt_interface(
        design_params=design_params,
        Q=best_Q,
        R=best_R,
        backend=roa_backend,
        robot=robot,
    )
    roa_calc._update_lqr(Q=best_Q, R=best_R)
    vol, rho_f, S = roa_calc._estimate()

    np.savetxt(os.path.join(save_dir, "rho"), [rho_f])
    np.savetxt(os.path.join(save_dir, "vol"), [vol])
    # np.savetxt(os.path.join(save_dir, "rhohist"), rhoHist)
    np.savetxt(os.path.join(save_dir, "Smatrix"), S)

    if plots:
        plotEllipse(
            goal[0],
            goal[1],
            0,
            1,
            rho_f,
            S,
            save_to=os.path.join(save_dir, "roaplot"),
            show=False,
        )

        plot_cma_results(
            data_path=save_dir,
            sign=-1.0,
            save_to=os.path.join(save_dir, "history"),
            show=False,
        )

    return best_par


def roa_design_opt(
    lqr_pars=[1.0, 1.0, 1.0, 1.0, 1.0],
    init_pars=[0.63, 0.3, 0.2],
    par_prefactors=[1.0, 1.0, 1.0],
    bounds=[[0.3, 1], [0.3, 0.5], [0.5, 1.0]],
    maxfevals=1000,
    sigma0=0.4,
    roa_backend="najafi",
    najafi_evals=100000,
    robot="acrobot",
    save_dir="data/",
    plots=False,
    num_proc=0,
):
    mass = [0.608, init_pars[0]]
    length = [init_pars[1], init_pars[2]]
    com = [length[0], length[1]]
    damping = [0.0, 0.0]
    cfric = [0.0, 0.0]
    # damping = [0.081, 0.0]
    # cfric = [0.093, 0.186]
    gravity = 9.81
    inertia = [mass[0] * length[0] ** 2.0, mass[1] * length[1] ** 2.0]
    if robot == "acrobot":
        torque_limit = [0.0, 5.0]
    if robot == "pendubot":
        torque_limit = [5.0, 0.0]

    goal = [np.pi, 0, 0, 0]

    popsize_factor = 4

    # timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
    # save_dir = os.path.join("data", robot, "lqr", "roa_paropt", timestamp)
    os.makedirs(save_dir)

    # loss function setup
    loss_func = roa_modelpar_lossfunc(
        par_prefactors=par_prefactors,
        roa_backend=roa_backend,
        najafi_evals=najafi_evals,
        bounds=bounds,
        robot=robot,
    )
    loss_func.set_model_parameters(
        mass=mass,
        length=length,
        com=com,
        damping=damping,
        gravity=gravity,
        coulomb_fric=cfric,
        inertia=inertia,
        torque_limit=torque_limit,
    )

    Q = np.diag((lqr_pars[0], lqr_pars[1], lqr_pars[2], lqr_pars[3]))
    R = np.diag((lqr_pars[4], lqr_pars[4]))

    loss_func.set_cost_parameters(
        p1p1_cost=Q[0, 0],
        p2p2_cost=Q[1, 1],
        v1v1_cost=Q[2, 2],
        v2v2_cost=Q[3, 3],
        p1v1_cost=0.0,
        p1v2_cost=0.0,
        p2v1_cost=0.0,
        p2v2_cost=0.0,
        u1u1_cost=R[0, 0],
        u2u2_cost=R[1, 1],
        u1u2_cost=0.0,
    )

    # optimization
    t0 = time.time()
    best_par = cma_optimization(
        loss_func=loss_func,
        init_pars=init_pars,
        bounds=[0, 1],
        save_dir=os.path.join(save_dir, "outcmaes"),
        sigma0=sigma0,
        popsize_factor=popsize_factor,
        maxfevals=maxfevals,
        num_proc=num_proc,
    )
    opt_time = (time.time() - t0) / 3600  # time in h

    best_par = loss_func.rescale_pars(best_par)
    print(best_par)

    np.savetxt(os.path.join(save_dir, "model_par.csv"), best_par)
    np.savetxt(os.path.join(save_dir, "time.txt"), [opt_time])

    par_dict = {
        "optimization": "design",
        "mass1": mass[0],
        "mass2": float(mass[1]),
        "length1": float(length[0]),
        "length2": float(length[1]),
        "com1": float(com[0]),
        "com2": float(com[1]),
        "inertia1": float(inertia[0]),
        "inertia2": float(inertia[1]),
        "damping1": damping[0],
        "damping2": damping[1],
        "coulomb_friction1": cfric[0],
        "coulomb_friction2": cfric[1],
        "gravity": gravity,
        "torque_limit1": torque_limit[0],
        "torque_limit2": torque_limit[1],
        "goal_pos1": goal[0],
        "goal_pos2": goal[1],
        "goal_vel1": goal[2],
        "goal_vel2": goal[3],
        "Q1": float(lqr_pars[0]),
        "Q2": float(lqr_pars[1]),
        "Q3": float(lqr_pars[2]),
        "Q4": float(lqr_pars[3]),
        "R": float(lqr_pars[4]),
        "par_prefactors": par_prefactors,
        "init_pars": init_pars,
        "bounds": bounds,
        "popsize_factor": popsize_factor,
        "maxfevals": maxfevals,
        "roa_backend": roa_backend,
        "najafi_evals": najafi_evals,
        # "tolfun": tolfun,
        # "tolx": tolx,
        # "tolstagnation": tolstagnation
    }

    with open(os.path.join(save_dir, "parameters.yml"), "w") as f:
        yaml.dump(par_dict, f)

    # recalculate the roa for the best parameters and save plot

    design_params = {
        "m": [mass[0], best_par[0]],
        "l": [best_par[1], best_par[2]],
        "lc": [best_par[1], best_par[2]],
        "b": damping,
        "fc": cfric,
        "g": gravity,
        "I": [mass[0] * best_par[1] ** 2, best_par[0] * best_par[2] ** 2],
        "tau_max": torque_limit,
    }

    roa_calc = caprr_coopt_interface(
        design_params=design_params, Q=Q, R=R, backend=roa_backend, robot=robot
    )
    roa_calc._update_lqr(Q=Q, R=R)
    vol, rho_f, S = roa_calc._estimate()

    np.savetxt(os.path.join(save_dir, "rho"), [rho_f])
    np.savetxt(os.path.join(save_dir, "vol"), [vol])
    # np.savetxt(os.path.join(save_dir, "rhohist"), rhoHist)

    # if plots:
    #     plotEllipse(goal[0], goal[1], 0, 1, rho_f, S,
    #                 save_to=os.path.join(save_dir, "roaplot"),
    #                 show=False)

    #     plot_cma_results(data_path=save_dir,
    #                      sign=-1.,
    #                      save_to=os.path.join(save_dir, "history"),
    #                      show=False)

    return best_par


def roa_alternate_opt(
    init_pars=[1.0, 1.0, 1.0, 1.0, 1.0, 0.63, 0.3, 0.2],
    par_prefactors=[20.0, 20.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0],
    bounds=[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0.3, 1], [0.3, 0.5], [0.5, 1.0]],
    maxfevals_per_opt=1000,
    sigma_dec=1.0,
    opt_order=["d", "c"],
    roa_backend="najafi",
    najafi_evals=100000,
    robot="acrobot",
    save_dir="data/",
    plots=False,
    num_proc=0,
):
    c_par = init_pars[:5]
    m_par = init_pars[5:]

    sigma_lqr = 0.4
    sigma_design = 0.4

    counter = 0
    _ = calc_roa(
        c_par=c_par,
        m_par=m_par,
        roa_backend=roa_backend,
        robot=robot,
        save_dir=os.path.join(save_dir, str(counter).zfill(2) + "_init"),
        plots=False,
    )

    counter += 1
    for o in opt_order:
        if o == "d":
            print("starting design optimization")
            m_par = roa_design_opt(
                lqr_pars=c_par,
                init_pars=m_par,
                par_prefactors=par_prefactors[5:],
                bounds=bounds[5:],
                maxfevals=maxfevals_per_opt,
                sigma0=sigma_design,
                roa_backend=roa_backend,
                najafi_evals=najafi_evals,
                robot=robot,
                save_dir=os.path.join(save_dir, str(counter).zfill(2) + "_design"),
                plots=plots,
                num_proc=num_proc,
            )
            sigma_design *= sigma_dec
        elif o == "c":
            print("starting controller optimization")
            c_par = roa_lqr_opt(
                model_pars=m_par,
                init_pars=c_par,
                par_prefactors=par_prefactors[:5],
                bounds=bounds[:5],
                maxfevals=maxfevals_per_opt,
                sigma0=sigma_lqr,
                roa_backend=roa_backend,
                najafi_evals=najafi_evals,
                robot=robot,
                save_dir=os.path.join(save_dir, str(counter).zfill(2) + "_lqr"),
                plots=plots,
                num_proc=num_proc,
            )
            sigma_lqr *= sigma_dec
        counter += 1
    best_par = [
        c_par[0],
        c_par[1],
        c_par[2],
        c_par[3],
        c_par[4],
        m_par[0],
        m_par[1],
        m_par[2],
    ]
    return best_par
