import os
import numpy as np
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment
from double_pendulum.controller.mcpilco.mcpilco_controller import (
    Controller_multi_policy_sum_of_gaussians_with_angles_numpy,
)
from double_pendulum.simulation.perturbations import get_random_gauss_perturbation_array
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.controller.acados_mpc.acados_mpc import AcadosMpc
from double_pendulum.controller.global_policy_testing_controller import (
    GlobalPolicyTestingControllerV2,
)
from double_pendulum.analysis.leaderboard import leaderboard_scores

import pickle as pkl
import pandas
from double_pendulum.filter.lowpass import lowpass_filter
np.random.seed(0)

name = "acados_mpc_acrobot"
leaderboard_config = {
    "csv_path": "trajectory.csv",
    "name": name,
    "simple_name": "acados_mpc_acrobot",
    "short_description": "Acados mpc for acrobot",
    "readme_path": f"readmes/{name}.md",
    "username": "blanka",
}

design = "design_C.1"
robot = "acrobot"
model = "model_1.0"
seed = 0

model_par_path = (
    "../../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)
mpar = model_parameters(filepath=model_par_path)
from parameters import (
    mpar_nolim,
    t_final,
    goal,
    height,
    method,
    design,
    n_disturbances,
    reset_length,
    kp,
    ki,
    kd,
)

actuated_joint = 1

# controller parameters
N_horizon=20
prediction_horizon=0.5
Nlp_max_iter=40
vmax = 16 #rad/s
vf = 16

bend_the_rules = False
tl = mpar.tl
tl[actuated_joint] = 6.0
if bend_the_rules:
    tl[1-actuated_joint] = 0.5
    mpar.set_torque_limit(tl)
else:
    tl[1-actuated_joint] = 0.0
    mpar.set_torque_limit(tl)

Q_mat = 2*np.diag([1000, 1000, 100, 100])
Qf_mat = 2*np.diag([100000, 100000, 10000, 10000])
R_mat = 2*np.diag([200.0, 200.0])
mpar.set_damping([0.005, 0.02])
mpar.set_cfric([0.03314955511059797, 0.03521137546780113])

mpar.set_motor_inertia([5.1336718481407864e-05])

controller = AcadosMpc(
    model_pars=mpar,
)
dt = 0.003
x0 = [0,0,0,0]
goal = [np.pi,0,0,0]
t_final = 60
controller.set_start(x0)
controller.set_goal(goal)
controller.set_parameters(
    N_horizon=N_horizon,
    prediction_horizon=prediction_horizon,
    Nlp_max_iter=Nlp_max_iter,
    max_solve_time=.003,
    solver_type="SQP",
    wrap_angle=True,
    warm_start=True,
    fallback_on_solver_fail=True,
    nonuniform_grid=True,
    cheating_on_inactive_joint=bend_the_rules,
    mpc_cycle_dt=0.003,
    outer_cycle_dt=dt,
    qp_solver_tolerance = 0.01,
    qp_solver = 'PARTIAL_CONDENSING_HPIPM',
    hpipm_mode = 'ROBUST',
    vel_penalty=100000000000000000000000,
)

controller.set_velocity_constraints(v_max=vmax, v_final=vf)
controller.set_cost_parameters(Q_mat=Q_mat, Qf_mat=Qf_mat, R_mat=R_mat)

lowpass_alpha = [1.0, 1.0, 0.9, 0.9]
filter_velocity_cut = 0.1
filter = lowpass_filter(lowpass_alpha, x0, filter_velocity_cut)
controller.set_filter(filter)
controller.init()