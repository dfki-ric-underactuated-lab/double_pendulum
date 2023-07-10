import numpy as np
import dill as pkl
import os

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.mcpilco.mcpilco_controller import Controller_sum_of_Gaussians_with_angles_numpy
from double_pendulum.controller.lqr.lqr_controller import LQRController, LQRController_nonsymbolic
from double_pendulum.controller.combined_controller import CombinedController
from sim_parameters import (
    mpar,
    dt,
    t_final,
    t0,
    x0,
    goal,
    integrator,
    design,
    model,
    robot,
)

name = "mcpilco"
leaderboard_config = {"csv_path": name + "/sim_swingup.csv",
                      "name": name,
                      "simple_name": "mcpilco",
                      "short_description": "Swingup trained with MBRL algorithm MC-PILCO + stabilization with LQR.",
                      "readme_path": f"readmes/{name}.md",
                      "username": 'turcato-niccolo'}

torque_limit = [0.0, 6.0]

T_control = 0.02
#T_control = 0.01
T_sym = 0.002

# controller parameters
ctrl_rate = int(T_control/T_sym)
u_max = torque_limit[1] * .7
n_dof = 2
controlled_joint = [1]
switch_stabilization = True

policy_par_path = (
    "../../../data/policies/"
    + design
    + "/"
    + model
    + "/"
    + robot
    + "/MC-PILCO/policy_mcpilco_nostab.np"
)
file = open(policy_par_path, 'rb')
parameters = pkl.load(file)

controller = Controller_sum_of_Gaussians_with_angles_numpy(parameters, ctrl_rate, u_max, n_dof, controlled_joint)

if switch_stabilization:
    stabilization_controller = LQRController_nonsymbolic(model_pars=mpar)
    # Q = np.diag((0.97, 0.93, 0.39, 0.26))
    # R = np.diag((0.11, 0.11))
    # stabilization_controller.set_cost_matrices(Q=Q, R=R)
    stabilization_controller.set_cost_parameters(u2u2_cost=100, p1p1_cost=100, p2p2_cost=100)
    stabilization_controller.set_parameters(failure_value=0., cost_to_go_cut=10 ** 3)
    condition_policy = lambda t, x: abs(x[0]) < 2.7 #abs(x[0]) - np.pi > 0.1 or abs(x[1]) > 0.1
    condition_stabilization = lambda t, x: abs(x[0]) > 2.8
    comb_controller = CombinedController(controller1=controller, controller2=stabilization_controller,
                                         condition1=condition_policy, condition2=condition_stabilization,
                                         verbose=False)
    controller = comb_controller

controller.set_goal(goal)
controller.init()
controller.init_()
