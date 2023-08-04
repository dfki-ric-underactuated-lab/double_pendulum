from double_pendulum.controller.mcpilco.mcpilco_controller import Controller_multi_policy_sum_of_gaussians_with_angles_numpy
import pickle as pkl

from exp_parameters import t_final, dt, x0, goal, mpar, design, model, robot

u_max = [0.5, 5.0]
n_dof = 2
controlled_joint = [0, 1]
dt_con = 1. / 33.  # 33 Hz

policy_par_path = (
    "../../../data/policies/"
    + design
    + "/"
    + model
    + "/"
    + robot
    + "/MC-PILCO/policy.np"
)
file = open(policy_par_path, 'rb')
parameters = pkl.load(file)

ctrl_rate = int(dt_con/dt)

controller = Controller_multi_policy_sum_of_gaussians_with_angles_numpy(parameters, ctrl_rate,
                                                                        u_max, n_dof,
                                                                        controlled_joint,
                                                                        active_pos_list=[[0, 1], [0, 1]],
                                                                        active_vel_list=[[2, 3], [2, 3]],
                                                                        wait_steps=10)

controller.init()
