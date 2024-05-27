import os
from datetime import datetime
import pickle
import pprint
import yaml
import numpy as np

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.tvlqr.tvlqr_controller import TVLQRController
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.analysis.benchmark import benchmarker
from double_pendulum.filter.lowpass import lowpass_filter
from double_pendulum.utils.wrap_angles import wrap_angles_top
from double_pendulum.analysis.utils import get_par_list

design = "design_C.1"
model = "model_1.1"
traj_model = "model_1.1"
robot = "acrobot"

# # model parameters
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

# simulation parameter
dt = 0.005
t_final = 10.0  # 4.985
integrator = "runge_kutta"
start = [0.0, 0.0, 0.0, 0.0]
goal = [np.pi, 0.0, 0.0, 0.0]

# filter args
lowpass_alpha = [1.0, 1.0, 0.3, 0.3]
filter_velocity_cut = 0.1
# lowpass_alpha = [1.0, 1.0, 1.0, 1.0]
# filter_velocity_cut = 0.0

# controller parameters
# N = 20
N = 100
con_dt = dt
N_init = 1000
max_iter = 10
# max_iter = 100
max_iter_init = 1000
regu_init = 1.0
max_regu = 10000.0
min_regu = 0.01
break_cost_redu = 1e-6
trajectory_stabilization = True
shifting = 1

init_csv_path = os.path.join(
    "../../data/trajectories", design, traj_model, robot, "ilqr_1/trajectory.csv"
)

## controller parameters
if robot == "acrobot":
    Q = np.diag([0.64, 0.56, 0.13, 0.067])
    R = np.eye(2) * 0.82
    Q_lqr = 0.1 * np.diag([0.65, 0.00125, 93.36, 0.000688])
    R_lqr = 100.0 * np.diag((0.025, 0.025))
elif robot == "pendubot":
    # Q = np.diag([0.64, 0.64, 0.4, 0.2])
    # R = np.eye(2)*0.82
    Q = 3.0 * np.diag([0.64, 0.64, 0.1, 0.1])
    R = np.eye(2) * 0.82
    Q_lqr = np.diag([0.0125, 6.5, 6.88, 9.36])
    R_lqr = np.diag([2.4, 2.4])
else:
    Q = np.diag([1.0, 1.0, 1.0, 1.0])
    R = np.eye(2)
    Q_lqr = np.diag([1.0, 1.0, 1.0, 1.0])
    R_lqr = np.eye(2)

Qf = np.copy(Q)


# Q = np.array(
#     [
#         [sCp[0], 0.0, 0.0, 0.0],
#         [0.0, sCp[1], 0.0, 0.0],
#         [0.0, 0.0, sCv[0], 0.0],
#         [0.0, 0.0, 0.0, sCv[1]],
#     ]
# )
# Qf = np.array(
#     [
#         [fCp[0], 0.0, 0.0, 0.0],
#         [0.0, fCp[1], 0.0, 0.0],
#         [0.0, 0.0, fCv[0], 0.0],
#         [0.0, 0.0, 0.0, fCv[1]],
#     ]
# )
# R = np.array([[sCu[0], 0.0], [0.0, sCu[1]]])
def condition1(t, x):
    return False


def condition2(t, x):
    goal = [np.pi, 0.0, 0.0, 0.0]
    eps = [0.2, 0.2, 1.5, 1.5]
    # eps = [0.2, 0.2, 0.8, 0.8]
    # eps = [0.1, 0.1, 0.4, 0.4]
    # eps = [0.1, 0.2, 2.0, 1.]

    y = wrap_angles_top(x)

    delta = np.abs(np.subtract(y, goal))
    max_diff = np.max(np.subtract(delta, eps))
    if max_diff > 0.0:
        return False
    else:
        return True


# benchmark parameters
eps = [0.35, 0.35, 1.0, 1.0]
check_only_final_state = True

N_var = 21

compute_model_robustness = False
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

compute_noise_robustness = False
meas_noise_mode = "vel"
meas_noise_sigma_list = np.linspace(0.0, 0.5, N_var)  # [0.0, 0.05, 0.1, 0.3, 0.5]

compute_unoise_robustness = False
u_noise_sigma_list = np.linspace(0.0, 2.0, N_var)

compute_uresponsiveness_robustness = False
u_responses = np.linspace(0.1, 2.1, N_var)  # [1.0, 1.3, 1.5, 2.0]

compute_delay_robustness = False
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
save_dir = os.path.join("data", design, model, robot, "tvlqr", timestamp)
os.makedirs(save_dir)

# filter
filter = lowpass_filter(lowpass_alpha, start, filter_velocity_cut)

# controller
controller1 = TVLQRController(
    model_pars=mpar, csv_path=init_csv_path, torque_limit=torque_limit
)

controller1.set_cost_parameters(Q=Q, R=R, Qf=Qf)

controller2 = LQRController(model_pars=mpar)
controller2.set_goal(goal)
controller2.set_cost_matrices(Q=Q_lqr, R=R_lqr)
controller2.set_parameters(failure_value=0.0, cost_to_go_cut=100)

controller = CombinedController(
    controller1=controller1,
    controller2=controller2,
    condition1=condition1,
    condition2=condition2,
    compute_both=False,
)
controller.set_filter(filter)
# controller.set_friction_compensation(damping=mpar.b, coulomb_fric=mpar.cf)
# # controller.set_friction_compensation(damping=[0., mpar.b[1]], coulomb_fric=[0., mpar.cf[1]])
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
