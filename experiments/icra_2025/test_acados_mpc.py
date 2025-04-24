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

import pickle as pkl

design = "design_C.1"
robot = "pendubot"
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


actuated_joint = 1

# controller parameters
N_horizon=20
prediction_horizon=0.5
Nlp_max_iter=40
vmax = 25 #rad/s
vf = 25

bend_the_rules = True
tl = mpar.tl
tl[actuated_joint] = 6.0
if bend_the_rules:
    tl[1-actuated_joint] = 0.5
    mpar.set_torque_limit(tl)
else:
    tl[1-actuated_joint] = 0.0
    mpar.set_torque_limit(tl)

if actuated_joint == 1: #acrobot
    Q_mat = 2*np.diag([100, 100, 10, 10])
    Qf_mat = 2*np.diag([100000, 100000, 1000, 1000])
    R_mat = 2*np.diag([0.000001, 0.000001])

if actuated_joint == 0: #pendubot
    Q_mat = 2*np.diag([100, 100, 10, 10])
    Qf_mat = 2*np.diag([100000, 100000, 1000, 1000]) 
    R_mat = 2*np.diag([0.000001, 0.000001])

controller = AcadosMpc(
    model_pars=mpar,
)
dt = 0.002
x0 = [0,0,0,0]
goal = [np.pi,0,0,0]
t_final = 60
controller.set_start(x0)
controller.set_goal(goal)
controller.set_parameters(
    N_horizon=N_horizon,
    prediction_horizon=prediction_horizon,
    Nlp_max_iter=Nlp_max_iter,
    max_solve_time=.01,
    solver_type="SQP_RTI",
    wrap_angle=False,
    warm_start=True,
    fallback_on_solver_fail=True,
    nonuniform_grid=False,
    cheating_on_inactive_joint=bend_the_rules,
    mpc_cycle_dt=0.002,
    outer_cycle_dt=dt,
    qp_solver_tolerance = 0.01,
    qp_solver = 'PARTIAL_CONDENSING_HPIPM',
    hpipm_mode = 'ROBUST'
)

controller.set_velocity_constraints(v_max=vmax, v_final=vf)
controller.set_cost_parameters(Q_mat=Q_mat, Qf_mat=Qf_mat, R_mat=R_mat)
#controller.load_init_traj(csv_path=init_csv_path)
controller.init()

run_experiment(
    controller=controller,
    dt=dt,
    t_final=t_final,
    can_port="can0",
    motor_ids=[7, 1],
    motor_directions=[1.0, -1.0],
    tau_limit=mpar.tl,
    save_dir=os.path.join("data", design, robot, "tmotors/acados_mpc"),
    record_video=True,
    safety_velocity_limit=30.0
)