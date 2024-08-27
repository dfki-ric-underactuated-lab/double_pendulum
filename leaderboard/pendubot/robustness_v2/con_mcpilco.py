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
from double_pendulum.utils.wrap_angles import wrap_angles_top

name = "mcpilco"
leaderboard_config = {"csv_path": name + "/sim_swingup.csv",
                      "name": name,
                      "simple_name": "mcpilco",
                      "short_description": "Swingup trained with MBRL algorithm MC-PILCO + stabilization with LQR.",
                      "readme_path": f"readmes/{name}.md",
                      "username": 'turcato-niccolo'}

torque_limit = [6.0, 0.0]

T_control = 0.02
#T_control = 0.01
T_sym = 0.002

# controller parameters
ctrl_rate = int(T_control/T_sym)
u_max = 1.5
n_dof = 2
controlled_joint = [0]
switch_stabilization = True


policy_par_path = (
    "../../../data/policies/"
    + design
    + "/"
    + model
    + "/"
    + robot
    + "/MC-PILCO/policy_1_5NM.np"
)
file = open(policy_par_path, 'rb')
parameters = pkl.load(file)

controller = Controller_sum_of_Gaussians_with_angles_numpy(parameters, ctrl_rate, u_max, n_dof, controlled_joint)
if switch_stabilization:
    stabilization_controller = LQRController_nonsymbolic(model_pars=mpar)
    Q = np.diag((0.97, 0.93, 0.39, 0.26))
    R = np.diag((0.11, 0.11))
    stabilization_controller.set_cost_matrices(Q=Q, R=R)
    load_path = "data/pendubot/lqr/roa"

    rho = 2.349853516578003232e-01
    S = np.array(
        [
            [
                9.770536750948697318e02,
                4.412387317512778395e02,
                1.990562043567418016e02,
                1.018948893750672369e02,
            ],
            [
                4.412387317512778395e02,
                1.999223464452055055e02,
                8.995900469226445750e01,
                4.605280324531641156e01,
            ],
            [
                1.990562043567418016e02,
                8.995900469226445750e01,
                4.059381113966859544e01,
                2.077912430021438439e01,
            ],
            [
                1.018948893750672369e02,
                4.605280324531641156e01,
                2.077912430021438439e01,
                1.063793947790017036e01,
            ],
        ]
    )

    def check_if_state_in_roa(S, rho, x):
        xdiff = x - np.array([np.pi, 0.0, 0.0, 0.0])
        rad = np.einsum("i,ij,j", xdiff, S, xdiff)
        return rad < rho, rad

    def condition2(t, x):
        y = wrap_angles_top(x)
        flag, rad = check_if_state_in_roa(S, rho, y)
        if flag:
            return flag
        return flag

    condition_policy = lambda t, x: np.abs(x[0]) < 2.7

    comb_controller = CombinedController(controller1=controller, controller2=stabilization_controller,
                                         condition1=condition_policy, condition2=condition2,
                                         verbose=False)
    controller = comb_controller


controller.set_goal(goal)
controller.init()
controller.init_()

