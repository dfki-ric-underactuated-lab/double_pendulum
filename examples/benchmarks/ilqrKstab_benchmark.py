import os
from datetime import datetime
import numpy as np
import pprint

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.trajectory_following.trajectory_controller import (
    TrajectoryController,
)
from double_pendulum.analysis.benchmark import benchmarker
from double_pendulum.analysis.utils import get_par_list
from double_pendulum.filter.lowpass import lowpass_filter
from double_pendulum.utils.csv_trajectory import load_trajectory, trajectory_properties

design = "design_A.0"
model = "model_2.1"
robot = "pendubot"

# model parameters
if robot == "acrobot":
    torque_limit = [0.0, 6.0]
elif robot == "pendubot":
    torque_limit = [6.0, 0.0]
else:
    torque_limit = [6.0, 6.0]

model_par_path = (
    "../../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model[:-1]
    + "0"
    + "/model_parameters.yml"
)
mpar = model_parameters(filepath=model_par_path)
mpar.set_motor_inertia(0.0)
mpar.set_damping([0.0, 0.0])
mpar.set_cfric([0.0, 0.0])
mpar.set_torque_limit(torque_limit)

# init trajectory
init_csv_path = os.path.join(
    "../../data/trajectories", design, model, robot, "ilqr_1/trajectory.csv"
)

# swingup parameters
start = [0.0, 0.0, 0.0, 0.0]
goal = [np.pi, 0.0, 0.0, 0.0]

# simulation parameter
T_des, X_des, U_des = load_trajectory(init_csv_path)
dt, t_final, _, _ = trajectory_properties(T_des, X_des)
dt = 0.005
t_final = 10.0  # 4.985
integrator = "runge_kutta"

# filter args
# lowpass_alpha = [1.0, 1.0, 0.3, 0.3]
# filter_velocity_cut = 0.1
lowpass_alpha = [1.0, 1.0, 1.0, 1.0]
filter_velocity_cut = 0.0

# cost par (only for performance cost calculation)
if robot == "acrobot":
    sCu = [9.97938814e00, 9.97938814e00]
    sCp = [2.06969312e00, 7.69967729e00]
    sCv = [1.55726136e-02, 5.42226523e-01]
    sCen = 0.0
    fCp = [3.82623819e02, 7.05315590e03]
    fCv = [5.89790058e01, 9.01459500e01]
    fCen = 0.0
if robot == "pendubot":
    sCu = [1.0, 1.0]
    sCp = [0.1, 0.2]
    sCv = [0.0, 0.0]
    sCen = 0.0
    fCp = [10.0, 20.0]
    fCv = [10.0, 20.0]
    fCen = 0.0

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

N_var = 21

compute_model_robustness = True
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

compute_noise_robustness = True
meas_noise_mode = "vel"
meas_noise_sigma_list = np.linspace(0.0, 0.5, N_var)  # [0.0, 0.05, 0.1, 0.3, 0.5]

compute_unoise_robustness = True
u_noise_sigma_list = np.linspace(0.0, 1.1, N_var)

compute_uresponsiveness_robustness = True
u_responses = np.linspace(0.1, 2.0, N_var)  # [1.0, 1.3, 1.5, 2.0]

compute_delay_robustness = True
delay_mode = "posvel"
delays = np.linspace(0.0, 0.04, N_var)  # [0.0, dt, 2*dt, 5*dt, 10*dt]

compute_perturbation_robustness = True
perturbation_repetitions = 10
perturbations_per_joint = 3
perturbation_min_t_dist = 1.0
perturbation_sigma_minmax = [0.5, 1.0]
perturbation_amp_minmax = [1.0, 3.0]

# create save directory
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join(
    "data", design, model, robot, "ilqr", "benchmark_K_stab", timestamp
)
os.makedirs(save_dir)

# filter
filter = lowpass_filter(lowpass_alpha, start, filter_velocity_cut)

# construct simulation objects
controller = TrajectoryController(
    csv_path=init_csv_path, torque_limit=torque_limit, kK_stabilization=True
)
controller.set_filter(filter)
controller.init()

ben = benchmarker(
    controller=controller,
    x0=start,
    dt=dt,
    t_final=t_final,
    goal=goal,
    epsilon=eps,
    check_only_final_state=check_only_final_state,
    integrator=integrator,
)
ben.set_model_parameter(model_pars=mpar)
ben.set_init_traj(init_csv_path)
ben.set_cost_par(Q=Q, R=R, Qf=Qf)
ben.compute_ref_cost()
res = ben.benchmark(
    compute_model_robustness=compute_model_robustness,
    compute_noise_robustness=compute_noise_robustness,
    compute_unoise_robustness=compute_unoise_robustness,
    compute_uresponsiveness_robustness=compute_uresponsiveness_robustness,
    compute_delay_robustness=compute_delay_robustness,
    compute_perturbation_robustness=compute_perturbation_robustness,
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
pprint.pprint(res)

# saving
os.system(f"cp {init_csv_path} " + os.path.join(save_dir, "init_trajectory.csv"))
mpar.save_dict(os.path.join(save_dir, "model_parameters.yml"))
controller.save(save_dir)
ben.save(save_dir)
