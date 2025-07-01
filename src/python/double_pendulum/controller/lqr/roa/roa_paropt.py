import os
import time
import numpy as np
import yaml

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.lqr.roa.ellipsoid import plotEllipse
from double_pendulum.controller.lqr.roa.roa_estimation import calc_roa
from double_pendulum.controller.lqr.roa.coopt_interface import caprr_coopt_interface
from double_pendulum.utils.optimization import cma_optimization  # , plot_cma_results


def replace_opt_vars_in_model_par(
    model_par=model_parameters(), optimization_model_par=np.zeros(3)
):
    model_par.m[1] = optimization_model_par[0]
    model_par.l[0] = optimization_model_par[1]
    model_par.l[1] = optimization_model_par[2]
    model_par.r[0] = optimization_model_par[1]
    model_par.r[1] = optimization_model_par[2]
    model_par.I[0] = model_par.m[0] * optimization_model_par[1] ** 2
    model_par.I[1] = optimization_model_par[0] * optimization_model_par[2] ** 2
    return model_par


def get_Q_and_R_from_opt_parameters(optimization_controller_par=np.zeros(5)):
    Q = np.diag(
        (
            optimization_controller_par[0],
            optimization_controller_par[1],
            optimization_controller_par[2],
            optimization_controller_par[3],
        )
    )
    R = np.diag((optimization_controller_par[4], optimization_controller_par[4]))
    return Q, R


class roa_lqrpar_lossfunc:
    def __init__(
        self,
        goal=[np.pi, 0, 0, 0],
        par_prefactors=[100, 100, 100, 100, 10],
        bounds=[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
        roa_backend="sos",
        najafi_evals=100000,
        robot="acrobot",
    ):
        self.goal = goal
        self.par_prefactors = np.asarray(par_prefactors)
        self.roa_backend = roa_backend
        self.bounds = np.asarray(bounds)
        self.najafi_evals = najafi_evals
        self.robot = robot

    def set_model_parameters(self, model_par=model_parameters()):
        self.model_par = model_par

    def __call__(self, pars):
        # p = np.asarray(pars)*self.par_prefactors
        p = self.rescale_pars(pars)

        Q = np.diag((p[0], p[1], p[2], p[3]))
        R = np.diag((p[4], p[4]))

        roa_calc = caprr_coopt_interface(
            self.model_par,
            self.goal,
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
        goal=[np.pi, 0, 0, 0],
        par_prefactors=[1.0, 1.0, 1.0],
        bounds=[[0.1, 1.0], [0.3, 1.0], [0.1, 1.0]],
        roa_backend="sos",
        najafi_evals=100000,
        robot="acrobot",
    ):
        self.goal = goal
        self.par_prefactors = np.asarray(par_prefactors)
        self.roa_backend = roa_backend
        self.bounds = np.asarray(bounds)
        self.najafi_evals = najafi_evals
        self.robot = robot

    def set_model_parameters(
        self,
        model_par=model_parameters(),
    ):
        # mass2, length1 and length2 will be overwritten during optimization
        self.model_par = model_par

    def set_cost_parameters(
        self,
        Q,
        R,
    ):
        self.Q = Q

        # control cost matrix
        self.R = R

    def __call__(self, pars):

        p = self.rescale_pars(pars)

        opt_model_par = [p[0], p[1], p[2]]  # m2, l1, l2

        roa_calc = caprr_coopt_interface(
            self.model_par,
            self.goal,
            self.Q,
            self.R,
            backend=self.roa_backend,
            najafi_evals=self.najafi_evals,
            robot=self.robot,
        )
        loss = roa_calc.design_opt_obj(opt_model_par)
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
        goal=[np.pi, 0, 0, 0],
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
        self.goal = goal
        self.par_prefactors = np.asarray(par_prefactors)
        self.roa_backend = roa_backend
        self.bounds = np.asarray(bounds)
        self.najafi_evals = najafi_evals
        self.robot = robot

    def set_model_parameters(
        self,
        model_par=model_parameters(),
    ):
        # mass2, length1 and length2 will be overwritten during optimization
        self.model_par = model_par

    def __call__(self, pars):

        p = self.rescale_pars(pars)

        Q = np.diag((p[0], p[1], p[2], p[3]))
        R = np.diag((p[4], p[4]))

        model_par = [p[5], p[6], p[7]]  # m2, l1, l2

        roa_calc = caprr_coopt_interface(
            self.model_par,
            self.goal,
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


def roa_lqr_opt(
    model_par=model_parameters(),
    goal=[np.pi, 0, 0, 0],
    # optimization_model_par=[0.63, 0.3, 0.2],
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
    popsize_factor=4,
):

    os.makedirs(save_dir)

    # mpar = replace_opt_vars_in_model_par(model_par, optimization_model_par)

    # loss function setup
    loss_func = roa_lqrpar_lossfunc(
        par_prefactors=par_prefactors,
        bounds=bounds,
        roa_backend=roa_backend,
        najafi_evals=najafi_evals,
        robot=robot,
    )
    loss_func.set_model_parameters(model_par)

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
    opt_time = (time.time() - t0) / 3600.0  # time in h

    best_par = loss_func.rescale_pars(best_par)
    print(best_par)

    np.savetxt(os.path.join(save_dir, "controller_par.csv"), best_par)
    np.savetxt(os.path.join(save_dir, "time.txt"), [opt_time])

    par_dict = {
        "optimization": "lqr",
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

    model_par.save_dict(os.path.join(save_dir, "model_pars.yml"))

    with open(os.path.join(save_dir, "roa_opt_parameters.yml"), "w") as f:
        yaml.dump(par_dict, f)

    # recalculate the roa for the best parameters and save plot
    best_Q = np.diag((best_par[0], best_par[1], best_par[2], best_par[3]))
    best_R = np.diag((best_par[4], best_par[4]))

    roa_calc = caprr_coopt_interface(
        model_par=model_par,
        goal=goal,
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

        # plot_cma_results(
        #     data_path=save_dir,
        #     sign=-1.0,
        #     save_to=os.path.join(save_dir, "history"),
        #     show=False,
        # )

    return best_par


def roa_design_opt(
    model_par=model_parameters(),
    goal=[np.pi, 0, 0, 0],
    # lqr_par=[1.0, 1.0, 1.0, 1.0, 1.0],
    Q=[1.0, 1.0, 1.0, 1.0],
    R=[1.0, 1.0],
    init_model_par=[0.63, 0.3, 0.2],
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
    popsize_factor=4,
):

    mpar = replace_opt_vars_in_model_par(model_par, init_model_par)

    os.makedirs(save_dir)

    # loss function setup
    loss_func = roa_modelpar_lossfunc(
        par_prefactors=par_prefactors,
        roa_backend=roa_backend,
        najafi_evals=najafi_evals,
        bounds=bounds,
        robot=robot,
    )
    loss_func.set_model_parameters(mpar)

    # Q = np.diag((lqr_par[0], lqr_par[1], lqr_par[2], lqr_par[3]))
    # R = np.diag((lqr_par[4], lqr_par[4]))

    loss_func.set_cost_parameters(Q, R)

    # optimization
    t0 = time.time()
    optimized_model_par_list = cma_optimization(
        loss_func=loss_func,
        init_pars=init_model_par,
        bounds=[0, 1],
        save_dir=os.path.join(save_dir, "outcmaes"),
        sigma0=sigma0,
        popsize_factor=popsize_factor,
        maxfevals=maxfevals,
        num_proc=num_proc,
    )
    opt_time = (time.time() - t0) / 3600  # time in h

    optimized_model_par_list = loss_func.rescale_pars(optimized_model_par_list)

    np.savetxt(os.path.join(save_dir, "model_par.csv"), optimized_model_par_list)
    np.savetxt(os.path.join(save_dir, "time.txt"), [opt_time])

    par_dict = {
        "optimization": "design",
        "goal_pos1": goal[0],
        "goal_pos2": goal[1],
        "goal_vel1": goal[2],
        "goal_vel2": goal[3],
        "Q1": float(Q[0][0]),
        "Q2": float(Q[1][1]),
        "Q3": float(Q[2][2]),
        "Q4": float(Q[3][3]),
        "R": float(R[0][0]),
        "par_prefactors": par_prefactors,
        "init_pars": init_model_par,
        "bounds": bounds,
        "popsize_factor": popsize_factor,
        "maxfevals": maxfevals,
        "roa_backend": roa_backend,
        "najafi_evals": najafi_evals,
        # "tolfun": tolfun,
        # "tolx": tolx,
        # "tolstagnation": tolstagnation
    }

    mpar.save_dict(os.path.join(save_dir, "model_pars.yml"))

    with open(os.path.join(save_dir, "roa_parameters.yml"), "w") as f:
        yaml.dump(par_dict, f)

    # recalculate the roa for the best parameters and save plot

    mpar = replace_opt_vars_in_model_par(model_par, optimized_model_par_list)

    roa_calc = caprr_coopt_interface(
        model_par=mpar, goal=goal, Q=Q, R=R, backend=roa_backend, robot=robot
    )
    roa_calc._update_lqr(Q=Q, R=R)
    vol, rho_f, S = roa_calc._estimate()

    np.savetxt(os.path.join(save_dir, "rho"), [rho_f])
    np.savetxt(os.path.join(save_dir, "vol"), [vol])
    # np.savetxt(os.path.join(save_dir, "rhohist"), rhoHist)

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

    #     plot_cma_results(data_path=save_dir,
    #                      sign=-1.,
    #                      save_to=os.path.join(save_dir, "history"),
    #                      show=False)

    return optimized_model_par_list


def roa_coopt(
    model_par=model_parameters(),
    goal=[np.pi, 0, 0, 0],
    init_opt_par=[1.0, 1.0, 1.0, 1.0, 1.0, 0.63, 0.3, 0.2],
    par_prefactors=[20.0, 20.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0],
    bounds=[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0.3, 1], [0.3, 0.5], [0.5, 1.0]],
    maxfevals=1000,
    sigma0=0.4,
    roa_backend="najafi",
    najafi_evals=100000,
    robot="acrobot",
    save_dir="data/",
    plots=False,
    num_proc=0,
    popsize_factor=4,
):

    mpar = replace_opt_vars_in_model_par(model_par, init_opt_par[:5])

    os.makedirs(save_dir)

    # loss function setup
    loss_func = roa_lqrandmodelpar_lossfunc(
        goal=goal,
        par_prefactors=par_prefactors,
        roa_backend=roa_backend,
        najafi_evals=najafi_evals,
        bounds=bounds,
        robot=robot,
    )
    loss_func.set_model_parameters(mpar)

    # optimization
    t0 = time.time()
    best_par = cma_optimization(
        loss_func=loss_func,
        init_pars=init_opt_par,
        bounds=[0, 1],
        save_dir=os.path.join(save_dir, "outcmaes"),
        sigma0=sigma0,
        popsize_factor=popsize_factor,
        maxfevals=maxfevals,
        num_proc=num_proc,
    )
    opt_time = (time.time() - t0) / 3600.0  # time in h

    best_par = loss_func.rescale_pars(best_par)
    print(best_par)

    np.savetxt(os.path.join(save_dir, "model_par.csv"), best_par)
    np.savetxt(os.path.join(save_dir, "time.txt"), [opt_time])

    par_dict = {
        "optimization": "design",
        "goal_pos1": goal[0],
        "goal_pos2": goal[1],
        "goal_vel1": goal[2],
        "goal_vel2": goal[3],
        "Q1": float(best_par[0]),
        "Q2": float(best_par[1]),
        "Q3": float(best_par[2]),
        "Q4": float(best_par[3]),
        "R": float(best_par[4]),
        "par_prefactors": par_prefactors,
        "init_pars": init_opt_par,
        "bounds": bounds,
        "popsize_factor": popsize_factor,
        "maxfevals": maxfevals,
        "roa_backend": roa_backend,
        "najafi_evals": najafi_evals,
        # "tolfun": tolfun,
        # "tolx": tolx,
        # "tolstagnation": tolstagnation
    }

    mpar = replace_opt_vars_in_model_par(model_par, best_par[5:])
    mpar.save_dict(os.path.join(save_dir, "model_pars.yml"))

    Q, R = get_Q_and_R_from_opt_parameters(best_par[:5])

    with open(os.path.join(save_dir, "roa_parameters.yml"), "w") as f:
        yaml.dump(par_dict, f)

    # recalculate the roa for the best parameters and save plot

    roa_calc = caprr_coopt_interface(
        model_par=mpar, goal=goal, Q=Q, R=R, backend=roa_backend, robot=robot
    )
    roa_calc._update_lqr(Q=Q, R=R)
    vol, rho_f, S = roa_calc._estimate()

    np.savetxt(os.path.join(save_dir, "rho"), [rho_f])
    np.savetxt(os.path.join(save_dir, "vol"), [vol])
    # np.savetxt(os.path.join(save_dir, "rhohist"), rhoHist)

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

    #     plot_cma_results(data_path=save_dir,
    #                      sign=-1.,
    #                      save_to=os.path.join(save_dir, "history"),
    #                      show=False)

    return best_par


def roa_alternate_opt(
    model_par=model_parameters(),
    goal=[np.pi, 0, 0, 0],
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
    mpar_init = replace_opt_vars_in_model_par(model_par, init_pars[5:])
    Q_init, R_init = get_Q_and_R_from_opt_parameters(init_pars[:5])
    _ = calc_roa(
        model_par=mpar_init,
        goal=goal,
        Q=Q_init,
        R=R_init,
        roa_backend=roa_backend,
        robot=robot,
        save_dir=os.path.join(save_dir, str(counter).zfill(2) + "_init"),
        plots=plots,
    )

    counter += 1
    for o in opt_order:
        if o == "d":
            print("starting design optimization")
            Q, R = get_Q_and_R_from_opt_parameters(c_par)
            m_par = roa_design_opt(
                model_par=model_par,
                goal=goal,
                Q=Q,
                R=R,
                init_model_par=m_par,
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
            model_par = replace_opt_vars_in_model_par(model_par, m_par)
            c_par = roa_lqr_opt(
                model_par=model_par,
                goal=goal,
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
