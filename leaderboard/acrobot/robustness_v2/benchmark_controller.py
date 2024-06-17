import os
import argparse
import importlib
import numpy as np
import yaml

# import pickle
# import pprint

from double_pendulum.analysis.benchmark import benchmarker
from double_pendulum.analysis.utils import get_par_list
from double_pendulum.analysis.benchmark_scores import get_scores
from double_pendulum.analysis.benchmark_plot import plot_benchmark_results

from sim_parameters import (
    mpar,
    dt,
    t_final,
    x0,
    goal,
    integrator,
    # t0,
    # design,
    # model,
    # robot,
)


def benchmark_controller(controller, save_dir, controller_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # cost par (only for performance cost calculation)
    sCu = [1.0, 1.0]
    sCp = [1.0, 1.0]
    sCv = [0.01, 0.01]
    fCp = [100.0, 100.0]
    fCv = [10.0, 10.0]

    Q = np.array(
        [
            [sCp[0], 0.0, 0.0, 0.0],
            [0.0, sCp[1], 0.0, 0.0],
            [0.0, 0.0, sCv[0], 0.0],
            [0.0, 0.0, 0.0, sCv[1]],
        ]
    )
    Qf = np.array(
        [
            [fCp[0], 0.0, 0.0, 0.0],
            [0.0, fCp[1], 0.0, 0.0],
            [0.0, 0.0, fCv[0], 0.0],
            [0.0, 0.0, 0.0, fCv[1]],
        ]
    )
    R = np.array([[sCu[0], 0.0], [0.0, sCu[1]]])

    # benchmark parameters
    eps = [0.35, 0.35, 1.0, 1.0]
    check_only_final_state = True
    # eps = [0.1, 0.1, 0.5, 0.5]
    # check_only_final_state = False

    N_var = 21

    mpar_vars = ["Ir", "m1r1", "I1", "b1", "cf1", "m2r2", "m2", "I2", "b2", "cf2"]

    Ir_var_list = np.linspace(0.0, 1e-4, N_var)
    m1r1_var_list = get_par_list(mpar.m[0] * mpar.r[0], 0.75, 1.25, N_var)
    I1_var_list = get_par_list(mpar.I[0], 0.75, 1.25, N_var)
    b1_var_list = np.linspace(-0.1, 0.1, N_var)
    cf1_var_list = np.linspace(-0.2, 0.2, N_var)
    m2r2_var_list = get_par_list(mpar.m[1] * mpar.r[1], 0.75, 1.25, N_var)
    m2_var_list = get_par_list(mpar.m[1], 0.75, 1.25, N_var)
    I2_var_list = get_par_list(mpar.I[1], 0.75, 1.25, N_var)
    b2_var_list = np.linspace(-0.1, 0.1, N_var)
    cf2_var_list = np.linspace(-0.2, 0.2, N_var)

    modelpar_var_lists = {
        "Ir": Ir_var_list,
        "m1r1": m1r1_var_list,
        "I1": I1_var_list,
        "b1": b1_var_list,
        "cf1": cf1_var_list,
        "m2r2": m2r2_var_list,
        "m2": m2_var_list,
        "I2": I2_var_list,
        "b2": b2_var_list,
        "cf2": cf2_var_list,
    }

    meas_noise_mode = "vel"
    meas_noise_sigma_list = np.linspace(0.0, 0.5, N_var)

    u_noise_sigma_list = np.linspace(0.0, 1.1, N_var)

    u_responses = np.linspace(0.1, 2.0, N_var)

    delay_mode = "posvel"
    delays = np.linspace(0.0, 0.04, N_var)

    perturbation_repetitions = 50
    perturbations_per_joint = 3
    perturbation_min_t_dist = 1.0
    perturbation_sigma_minmax = [0.05, 0.1]
    perturbation_amp_minmax = [0.5, 5.0]

    ben = benchmarker(
        controller=controller,
        x0=x0,
        dt=dt,
        t_final=t_final,
        goal=goal,
        epsilon=eps,
        check_only_final_state=check_only_final_state,
        integrator=integrator,
    )
    ben.set_model_parameter(model_pars=mpar)
    ben.set_cost_par(Q=Q, R=R, Qf=Qf)
    ben.compute_ref_cost()
    res = ben.benchmark(
        compute_model_robustness=True,
        compute_noise_robustness=True,
        compute_unoise_robustness=True,
        compute_uresponsiveness_robustness=True,
        compute_delay_robustness=True,
        compute_perturbation_robustness=True,
        mpar_vars=mpar_vars,
        modelpar_var_lists=modelpar_var_lists,
        meas_noise_mode=meas_noise_mode,
        meas_noise_sigma_list=meas_noise_sigma_list,
        u_noise_sigma_list=u_noise_sigma_list,
        u_responses=u_responses,
        delay_mode=delay_mode,
        delays=delays,
        perturbation_repetitions=perturbation_repetitions,
        perturbations_per_joint=perturbations_per_joint,
        perturbation_min_t_dist=perturbation_min_t_dist,
        perturbation_sigma_minmax=perturbation_sigma_minmax,
        perturbation_amp_minmax=perturbation_amp_minmax,
    )
    # pprint.pprint(res)

    mpar.save_dict(os.path.join(save_dir, "model_parameters.yml"))
    controller.save(save_dir)
    ben.save(save_dir)

    plot_benchmark_results(
        save_dir,
        "benchmark_results.pkl",
        costlim=[0, 5],
        show=False,
        save=True,
        file_format="png",
        scale=0.5,
    )
    scores = get_scores(save_dir, "benchmark_results.pkl")

    with open(os.path.join(save_dir, "scores.yml"), "w") as f:
        yaml.dump(scores, f)

    if os.path.exists(f"readmes/{controller_name}.md"):
        os.system(f"cp readmes/{controller_name}.md {save_dir}/README.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("controller", help="name of the controller to simulate")
    controller_arg = parser.parse_args().controller
    if controller_arg[-3:] == ".py":
        controller_arg = controller_arg[:-3]

    controller_name = controller_arg[4:]
    print(f"Simulating controller {controller_name}")

    save_dir = f"data/{controller_name}"

    imp = importlib.import_module(controller_arg)
    controller = imp.controller

    benchmark_controller(controller, save_dir)
