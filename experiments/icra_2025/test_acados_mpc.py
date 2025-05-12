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
    Q_mat = 2*np.diag([1000, 1000, 100, 100])
    Qf_mat = 2*np.diag([100000, 100000, 10000, 10000])
    R_mat = 2*np.diag([25.0, 25.0])
    #mpar.set_cfric([0.05, 0.04])
    #mpar.set_damping([0.001, 0.05])
    mpar.set_damping([0.005, 0.02])
    mpar.set_cfric([0.03314955511059797, 0.03521137546780113])

if actuated_joint == 0: #pendubot
    Q_mat = 2*np.diag([1000, 1000, 100, 100])
    Qf_mat = 2*np.diag([100000, 100000, 10000, 10000]) 
    R_mat = 2*np.diag([50.0, 50.0])
    mpar.set_cfric([0.05, 0.04])
    mpar.set_damping([0.005, 0.04])
    #mpar.set_damping([0.001, 0.07])

#mpar.set_cfric([0.03314955511059797, 0.03521137546780113])
#mpar.set_damping([0.005, 0.04])
mpar.set_motor_inertia([5.1336718481407864e-05])

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
    wrap_angle=True,
    warm_start=True,
    fallback_on_solver_fail=True,
    nonuniform_grid=True,
    cheating_on_inactive_joint=bend_the_rules,
    mpc_cycle_dt=0.002,
    outer_cycle_dt=dt,
    qp_solver_tolerance = 0.01,
    qp_solver = 'PARTIAL_CONDENSING_HPIPM',
    hpipm_mode = 'ROBUST',
    vel_penalty=100000000000000000000000,
)

controller.set_velocity_constraints(v_max=vmax, v_final=vf)
controller.set_cost_parameters(Q_mat=Q_mat, Qf_mat=Qf_mat, R_mat=R_mat)
#controller.load_init_traj(csv_path=init_csv_path)

lowpass_alpha = [1.0, 1.0, 0.9, 0.9]
filter_velocity_cut = 0.1
filter = lowpass_filter(lowpass_alpha, x0, filter_velocity_cut)
controller.set_filter(filter)
controller.init()

global_policy_testing_controller = GlobalPolicyTestingControllerV2(
    controller,
    goal=goal,
    n_disturbances=11,
    t_max=t_final,
    reset_length=.5,
    method=method,
    height=height,
    mpar=mpar_nolim,
    kp=4.0,
    ki=ki,
    kd=1.0,
)

print(mpar.tl)

save_dir = os.path.join("data", design, robot, "tmotors/acados_mpc")
run_experiment(
    controller=global_policy_testing_controller,
    dt=dt,
    t_final=t_final,
    can_port="can0",
    motor_ids=[2, 1],
    motor_directions=[1.0, -1.0],
    tau_limit=[6.0,6.0],
    save_dir=save_dir,
    record_video=True,
    safety_velocity_limit=20.0,
    safety_position_limit=4*np.pi,
    velocities_from_positions=False
)

run_directories = os.listdir(save_dir)
print("run_directories=", run_directories)
run_directories = [os.path.join(save_dir, d) for d in run_directories]
run_directories.sort(key=lambda x: os.path.getmtime(x))
save_dir_date = run_directories[-1]

save_lb_to = os.path.join(save_dir_date, "leaderboard_entry.csv")

name = "donothing"
leaderboard_config = {
    "csv_path": "trajectory.csv",
    "name": name,
    "simple_name": "acados_mpc",
    "short_description": "Acados mpc",
    "readme_path": f"readmes/{name}.md",
    "username": "blanka",
}



leaderboard_config["csv_path"] = os.path.join(save_dir_date, leaderboard_config["csv_path"])
data_paths = {}
data_paths[leaderboard_config["name"]] = leaderboard_config

leaderboard_scores(
    data_paths=data_paths,
    save_to=save_lb_to,
    mpar=mpar_nolim,
    weights={
        # "swingup_time": 0.0,  # not used
        # "max_tau": 0.0,  # not used
        # "energy": 0.0,  # not used
        # "integ_tau": 0.0,  # not used
        # "tau_cost": 0.0,  # not used
        # "tau_smoothness": 0.0,  # not used
        # "velocity_cost": 0.0,  # not used
        "uptime": 1.0,
        # "n_swingups": 0.0,  # not used
    },
    normalize={
        # "swingup_time": 1.0,  # not used
        # "max_tau": 1.0,  # not used
        # "energy": 1.0,  # not used
        # "integ_tau": 1.0,  # not used
        # "tau_cost": 1.0,  # not used
        # "tau_smoothness": 1.0,  # not used
        # "velocity_cost": 1.0,  # not used
        "uptime": t_final,
        # "n_swingups": 1.0,  # not used
    },
    link_base="",
    score_version="v3",
)
df = pandas.read_csv(save_lb_to)
print(df.sort_values(by=["RealAI Score"], ascending=False).to_markdown(index=False))

#horizon 
#vmax
#