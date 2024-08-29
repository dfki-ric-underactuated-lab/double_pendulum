import os
import numpy as np
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment
from double_pendulum.controller.mcpilco.mcpilco_controller import (
    Controller_multi_policy_sum_of_gaussians_with_angles_numpy,
)
import pickle as pkl

design = "design_C.1"
robot = "pendubot"
model = "model_1.0"

if robot == "acrobot":
    torque_limit = [0.5, 5.0]
if robot == "pendubot":
    torque_limit = [5.0, 0.5]

model_par_path = (
    "../../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)
mpar_con = model_parameters(filepath=model_par_path)

## trajectory parameters
t_final = 10.0
x0 = [0.0] * 4
goal = [np.pi, 0.0, 0.0, 0.0]
## controller parameters
dt = 1.0 / 33.0  # 33 Hz
n_dof = 2
controlled_joint = [0, 1]

policy_par_path = (
    "../../data/policies/" + design + "/" + model + "/" + robot + "/MC-PILCO/policy.np"
)
file = open(policy_par_path, "rb")
parameters = pkl.load(file)

ctrl_rate = 1

controller = Controller_multi_policy_sum_of_gaussians_with_angles_numpy(
    parameters,
    ctrl_rate,
    torque_limit,
    n_dof,
    controlled_joint,
    active_pos_list=[[0, 1], [0, 1]],
    active_vel_list=[[2, 3], [2, 3]],
    wait_steps=10,
)

controller.init()
run_experiment(
    controller=controller,
    dt=dt,
    t_final=t_final,
    can_port="can0",
    motor_ids=[3, 1],
    motor_directions=[1.0, -1.0],
    tau_limit=torque_limit,
    save_dir=os.path.join("data", design, robot, "tmotors/mcpilco"),
)
