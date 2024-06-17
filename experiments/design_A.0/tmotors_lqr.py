import os
import numpy as np

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment
from double_pendulum.filter.lowpass import lowpass_filter


design = "design_A.0"
model = "model_2.0"
traj_model = "model_2.1"
robot = "acrobot"

torque_limit = [5.0, 5.0]
friction_compensation = True

# model parameters
if robot == "pendubot":
    torque_limit_con = [5.0, 0.0]
    active_act = 0
elif robot == "acrobot":
    torque_limit_con = [0.0, 5.0]
    active_act = 1

model_par_path = (
    "../../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)
mpar = model_parameters(filepath=model_par_path)

mpar_con = model_parameters(filepath=model_par_path)
# mpar_con.set_motor_inertia(0.)
if friction_compensation:
    mpar_con.set_damping([0.0, 0.0])
    mpar_con.set_cfric([0.0, 0.0])
mpar_con.set_torque_limit(torque_limit_con)

dt = 0.002
t_final = 30.0
x0 = [0.0, 0.0, 0.0, 0.0]
goal = [np.pi, 0.0, 0.0, 0.0]

# filter args
lowpass_alpha = [1.0, 1.0, 0.2, 0.2]
filter_velocity_cut = 0.1

# controller
if robot == "acrobot":
    # Q = np.diag((0.97, 0.93, 0.39, 0.26))
    # R = np.diag((0.11, 0.11))
    # Q = np.diag((0.97, 0.93, 0.39, 0.26))
    # R = np.diag((111.1, 111.1))

    # Q = np.diag([6.5, 0.0125, 9.36, 6.88])
    # R = np.diag([25., 25.])
    # Q = np.diag((0.97, 0.93, 0.39, 0.26))
    # R = np.diag((.11, .11))
    Q = 0.1 * np.diag([0.65, 0.00125, 93.6, 0.000688])
    # R = 100.*np.diag((.025, .025))
    R = 100.0 * np.diag((0.025, 0.025))

    ctg_cut = 1000000

elif robot == "pendubot":
    Q = np.diag([0.0125, 6.5, 6.88, 9.36])
    R = np.diag([0.024, 0.024])
    # R = np.diag([0.24, 0.24])
    # R = np.diag([2.5, 2.5])
    ctg_cut = 1000

# filter
filter = lowpass_filter(lowpass_alpha, x0, filter_velocity_cut)

controller = LQRController(model_pars=mpar_con)
controller.set_goal(goal)
controller.set_cost_matrices(Q=Q, R=R)
controller.set_parameters(failure_value=0.0, cost_to_go_cut=ctg_cut)
controller.set_filter(filter)

if friction_compensation:
    # controller.set_friction_compensation(damping=[0.001, 0.001], coulomb_fric=[0.09, 0.078])
    controller.set_friction_compensation(damping=mpar.b, coulomb_fric=mpar.cf)
controller.init()

run_experiment(
    controller=controller,
    dt=dt,
    t_final=t_final,
    can_port="can0",
    motor_ids=[7, 8],
    tau_limit=torque_limit,
    save_dir=os.path.join("data", design, robot, "tmotors/lqr_results"),
)
