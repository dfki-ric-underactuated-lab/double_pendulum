import numpy as np
import pickle as pkl
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.mcpilco.mcpilco_controller import Controller_multi_policy_sum_of_gaussians_with_angles_numpy
from parameters import goal, design, model

robot = "acrobot"
torque_limit = [0.5, 6.0]

name = "mcpilco"
leaderboard_config = {
    "csv_path": "trajectory.csv",
    "name": name,
    "simple_name": "mcpilco",
    "short_description": "Global Controller trained with MC-PILCO.",
    "readme_path": f"readmes/{name}.md",
    "username": "turcato-niccolo",
}

dt = 1 / 330.0  # trained on 33 Hz
ctrl_rate = 10  # control is still 33Hz, but control loop is working at matching frequency

n_dof = 2
controlled_joint = [0, 1]

policy_par_path = (
    "../../data/policies/" + design + "/" + model + "/" + robot + "/MC-PILCO/global_policy_" + robot + ".np"
)
file = open(policy_par_path, "rb")
parameters = pkl.load(file)

controller = Controller_multi_policy_sum_of_gaussians_with_angles_numpy(
    parameters, ctrl_rate, torque_limit, n_dof, controlled_joint,
    active_pos_list=[[0, 1], [0, 1]],
    active_vel_list=[[2, 3], [2, 3]], wait_steps=0
)
controller.set_goal(goal)
controller.init()
