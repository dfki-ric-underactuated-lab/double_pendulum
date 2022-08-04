import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.system_identification.dynamics import build_identification_matrices
from double_pendulum.system_identification.loss import errfunc
from double_pendulum.system_identification.plotting import plot_torques
from double_pendulum.utils.optimization import solve_least_squares, cma_optimization, scipy_par_optimization
from double_pendulum.utils.csv_trajectory import load_trajectory, concatenate_trajectories


def run_system_identification(measured_data_csv,
                              fixed_mpar,
                              variable_mpar,
                              mp0,
                              bounds,
                              read_with="pandas",
                              keys="shoulder-elbow",
                              optimization_method="least_squares",
                              save_dir=".",
                              num_proc=0,
                              sigma0=0.1,
                              rescale=False,
                              maxfevals=10000):

    Q, phi = build_identification_matrices(fixed_mpar,
                                           variable_mpar,
                                           measured_data_csv,
                                           read_with=read_with,
                                           keys=keys)

    if optimization_method in ["least_squares", "least-squares"]:

        loss_function = errfunc(Q, phi, bounds, rescale, False)
        if rescale:
            x0 = loss_function.rescale_pars(mp0)
            b = np.asarray(len(x0)*[[0., 1.]]).T
        else:
            x0 = np.copy(mp0)
            b = np.copy(bounds)

        mp_opt_raw = solve_least_squares(loss_function,
                x0,
                b,
                maxfevals=maxfevals)
    elif optimization_method in ["cma", "cma-es"]:

        loss_function = errfunc(Q, phi, bounds, rescale, True)
        if rescale:
            x0 = loss_function.rescale_pars(mp0)
            b = np.asarray(len(x0)*[[0., 1.]]).T
        else:
            x0 = np.copy(mp0)
            b = np.copy(bounds)

        mp_opt_raw = cma_optimization(loss_function,
                        x0,
                        b,
                        save_dir=os.path.join(save_dir, "outcmaes/"),
                        num_proc=num_proc,
                        sigma0=sigma0,
                        popsize_factor=3,
                        maxfevals=maxfevals)
    else:

        loss_function = errfunc(Q, phi, bounds, rescale, True)
        if rescale:
            x0 = loss_function.rescale_pars(mp0)
            b = np.asarray(len(x0)*[[0., 1.]]).T
        else:
            x0 = np.copy(mp0)
            b = np.copy(bounds)

        mp_opt_raw = scipy_par_optimization(loss_function,
                x0,
                b.T,
                method=optimization_method,
                maxfevals=maxfevals)

    if rescale:
        mp_opt = loss_function.unscale_pars(mp_opt_raw)
    else:
        mp_opt = mp_opt_raw

    print('Identified Parameters:')
    for i in range(len(variable_mpar)):
        print("{:10s} = {:+.3e}".format(variable_mpar[i], mp_opt[i]))

    # calculate errors
    Q_opt = phi.dot(mp_opt)
    mae = mean_absolute_error(Q.flatten(), Q_opt.flatten())
    rmse = mean_squared_error(Q.flatten(), Q_opt.flatten(), squared=False)

    print("Mean absolute error: ", mae)
    print("Mean root mean squared error: ", rmse)

    # plotting results
    T, X, U = concatenate_trajectories(measured_data_csv,
                                       read_withs=read_with,
                                       keys=keys,
                                       with_tau=True)
    plot_torques(T, Q[::2, 0], Q[1::2, 0], Q_opt[::2], Q_opt[1::2])

    all_par = fixed_mpar
    for i, key in enumerate(variable_mpar):
        if key == "m1r1":
            all_par["m1"] = mp_opt[i] / fixed_mpar["l1"]
            all_par["r1"] = fixed_mpar["l1"]
        elif key == "m2r2":
            all_par["r2"] = mp_opt[i] / mp_opt[i+1]
            # this requires the order ..., "m2r2", "m2", .. in variable_mpar
            # Todo: find better solution
        else:
            all_par[key] = mp_opt[i]
    mpar = model_parameters()
    mpar.load_dict(all_par)

    return mp_opt, mpar
