import os
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.system_identification.dynamics import (
    build_identification_matrices,
    build_identification_function,
)
from double_pendulum.system_identification.loss import errfunc, errfunc_nl
from double_pendulum.system_identification.plotting import plot_torques
from double_pendulum.utils.optimization import (
    solve_least_squares,
    cma_optimization,
    scipy_par_optimization,
)
from double_pendulum.utils.csv_trajectory import concatenate_trajectories


def run_system_identification(
    measured_data_csv,
    fixed_mpar,
    variable_mpar,
    mp0,
    bounds,
    optimization_method="least_squares",
    save_dir=".",
    num_proc=0,
    sigma0=0.1,
    rescale=False,
    maxfevals=10000,
    filt="butterworth",
    show_plot=True,
):
    Q, phi, T = build_identification_matrices(
        fixed_mpar, variable_mpar, measured_data_csv, filt=filt
    )

    Q_noisy, _, _ = build_identification_matrices(
        fixed_mpar, variable_mpar, measured_data_csv, filt=None
    )

    if optimization_method in ["least_squares", "least-squares"]:
        loss_function = errfunc(Q, phi, bounds, rescale, False)
        if rescale:
            x0 = loss_function.rescale_pars(mp0)
            b = np.asarray(len(x0) * [[0.0, 1.0]]).T
        else:
            x0 = np.copy(mp0)
            b = np.copy(bounds)

        mp_opt_raw = solve_least_squares(loss_function, x0, b, maxfevals=maxfevals)
    elif optimization_method in ["cma", "cma-es"]:
        loss_function = errfunc(Q, phi, bounds, rescale, True)
        if rescale:
            x0 = loss_function.rescale_pars(mp0)
            b = np.asarray(len(x0) * [[0.0, 1.0]]).T
        else:
            x0 = np.copy(mp0)
            b = np.copy(bounds)

        mp_opt_raw = cma_optimization(
            loss_function,
            x0,
            b,
            save_dir=os.path.join(save_dir, "outcmaes/"),
            num_proc=num_proc,
            sigma0=sigma0,
            popsize_factor=3,
            maxfevals=maxfevals,
        )
    else:
        loss_function = errfunc(Q, phi, bounds, rescale, True)
        if rescale:
            x0 = loss_function.rescale_pars(mp0)
            b = np.asarray(len(x0) * [[0.0, 1.0]]).T
        else:
            x0 = np.copy(mp0)
            b = np.copy(bounds)

        mp_opt_raw = scipy_par_optimization(
            loss_function, x0, b.T, method=optimization_method, maxfevals=maxfevals
        )

    if rescale:
        mp_opt = loss_function.unscale_pars(mp_opt_raw)
    else:
        mp_opt = mp_opt_raw

    print("Identified Parameters:")
    for i in range(len(variable_mpar)):
        print("{:10s} = {:+.3e}".format(variable_mpar[i], mp_opt[i]))

    # calculate errors
    Q_opt = phi.dot(mp_opt)
    mae = mean_absolute_error(Q.flatten(), Q_opt.flatten())
    rmse = mean_squared_error(Q.flatten(), Q_opt.flatten(), squared=False)
    mae_noisy = mean_absolute_error(Q_noisy.flatten(), Q_opt.flatten())
    rmse_noisy = mean_squared_error(Q_noisy.flatten(), Q_opt.flatten(), squared=False)

    print("Mean absolute error (Filtered data): ", mae)
    print("Mean root mean squared error (Filtered data): ", rmse)
    print("Mean absolute error (Noisy data): ", mae_noisy)
    print("Mean root mean squared error (Noisy data): ", rmse_noisy)

    # plotting results
    # T, X, U = concatenate_trajectories(measured_data_csv, with_tau=True)
    plot_torques(
        T,
        Q[::2, 0],
        Q[1::2, 0],
        Q_opt[::2],
        Q_opt[1::2],
        save_to=os.path.join(save_dir, "torques.svg"),
        show=show_plot,
    )

    all_par = fixed_mpar
    for i, key in enumerate(variable_mpar):
        if key == "m1r1":
            all_par["m1"] = mp_opt[i] / fixed_mpar["l1"]
            all_par["r1"] = fixed_mpar["l1"]
        elif key == "m2r2":
            all_par["r2"] = mp_opt[i] / mp_opt[i + 1]
            # this requires the order ..., "m2r2", "m2", .. in variable_mpar
            # Todo: find better solution
        else:
            all_par[key] = mp_opt[i]
    mpar = model_parameters()
    mpar.load_dict(all_par)

    return mp_opt, mpar


def run_system_identification_nl(
    measured_data_csv,
    fixed_mpar,
    variable_mpar,
    mp0,
    bounds,
    optimization_method="cma-es",
    save_dir=".",
    num_proc=0,
    sigma0=0.1,
    rescale=False,
    maxfevals=10000,
    filt="butterworth",
    show_plot=True,
):
    dyn_func, T, X, ACC, U = build_identification_function(
        fixed_mpar, variable_mpar, measured_data_csv, filt=filt
    )

    dyn_func_noisy, T_n, X_n, ACC_n, U_n = build_identification_function(
        fixed_mpar, variable_mpar, measured_data_csv, filt=filt
    )
    if optimization_method in ["cma", "cma-es"]:
        # not working with lambdified function
        loss_function = errfunc_nl(dyn_func, bounds, X, ACC, U, rescale, True)
        if rescale:
            x0 = loss_function.rescale_pars(mp0)
            b = np.asarray(len(x0) * [[0.0, 1.0]]).T
        else:
            x0 = np.copy(mp0)
            b = np.copy(bounds)

        mp_opt_raw = cma_optimization(
            loss_function,
            x0,
            b,
            save_dir=os.path.join(save_dir, "outcmaes/"),
            num_proc=num_proc,
            sigma0=sigma0,
            popsize_factor=3,
            maxfevals=maxfevals,
        )
    else:
        loss_function = errfunc_nl(dyn_func, bounds, X, ACC, U, rescale, True)
        if rescale:
            x0 = loss_function.rescale_pars(mp0)
            b = np.asarray(len(x0) * [[0.0, 1.0]]).T
        else:
            x0 = np.copy(mp0)
            b = np.copy(bounds)

        mp_opt_raw = scipy_par_optimization(
            loss_function, x0, b.T, method=optimization_method, maxfevals=maxfevals
        )

    if rescale:
        mp_opt = loss_function.unscale_pars(mp_opt_raw)
    else:
        mp_opt = mp_opt_raw

    print("Identified Parameters:")
    for i in range(len(variable_mpar)):
        print("{:10s} = {:+.3e}".format(variable_mpar[i], mp_opt[i]))
    # calculate errors
    U_pred = dyn_func(
        X.T[0],
        X.T[1],
        X.T[2],
        X.T[3],
        ACC.T[0],
        ACC.T[1],
        # U.T[0], U_n.T[1],
        *mp_opt
    )[:, 0, :].T
    U_pred_n = dyn_func(
        X_n.T[0],
        X_n.T[1],
        X_n.T[2],
        X_n.T[3],
        ACC_n.T[0],
        ACC_n.T[1],
        # U_n.T[0], U_n.T[1],
        *mp_opt
    )[:, 0, :].T

    mae = np.sum(np.abs(U_pred - U))
    rmse = np.sqrt(np.sum(np.square(U_pred - U)))
    mae_noisy = np.sum(np.abs(U_pred_n - U_n))
    rmse_noisy = np.sqrt(np.sum(np.square(U_pred_n - U_n)))

    print("Mean absolute error (Filtered data): ", mae)
    print("Mean root mean squared error (Filtered data): ", rmse)
    print("Mean absolute error (Noisy data): ", mae_noisy)
    print("Mean root mean squared error (Noisy data): ", rmse_noisy)

    all_par = fixed_mpar
    for i, key in enumerate(variable_mpar):
        if key == "m1r1":
            all_par["m1"] = mp_opt[i] / fixed_mpar["l1"]
            all_par["r1"] = fixed_mpar["l1"]
        elif key == "m2r2":
            all_par["r2"] = mp_opt[i] / mp_opt[i + 1]
            # this requires the order ..., "m2r2", "m2", .. in variable_mpar
            # Todo: find better solution
        else:
            all_par[key] = mp_opt[i]
    mpar = model_parameters()
    mpar.load_dict(all_par)

    # plotting results
    plot_torques(
        T,
        U.T[0],
        U.T[1],
        U_pred.T[0],
        U_pred.T[1],
        save_to=os.path.join(save_dir, "torques.svg"),
        show=show_plot,
    )
    plot_torques(
        T,
        U.T[0],
        U.T[1],
        U_pred_n.T[0],
        U_pred_n.T[1],
        save_to=os.path.join(save_dir, "torques_unfilterd.svg"),
        show=show_plot,
    )

    return mp_opt, mpar
