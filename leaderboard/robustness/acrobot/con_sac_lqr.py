import numpy as np
from stable_baselines3 import SAC

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.utils.wrap_angles import wrap_angles_top
from double_pendulum.controller.SAC.SAC_controller import SACController
from double_pendulum.simulation.gym_env import (
    double_pendulum_dynamics_func,
)
from double_pendulum.utils.wrap_angles import wrap_angles_diff

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

name = "sac_lqr"
leaderboard_config = {
    "csv_path": name + "/sim_swingup.csv",
    "name": name,
    "simple_name": "SAC LQR",
    "short_description": "Swing-up with an RL Policy learned with SAC.",
    "readme_path": f"readmes/{name}.md",
    "username": "chiniklas",
}

# class SACController(AbstractController):
#     def __init__(self, model_path, dynamics_func, dt):
#         super().__init__()
#
#         self.model = SAC.load(model_path)
#         self.dynamics_func = dynamics_func
#         self.dt = dt
#
#     def get_control_output_(self, x, t=None):
#         obs = self.dynamics_func.normalize_state(x)
#         action = self.model.predict(obs)
#         u = self.dynamics_func.unscale_action(action)
#         return u


torque_limit = [0.0, 5.0]


# simulation parameters
Q = np.diag((0.97, 0.93, 0.39, 0.26))
R = np.diag((0.11, 0.11))
load_path = "data/acrobot/lqr/roa"

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
    return rad < 1.5 * rho, rad


def condition1(t, x):
    return False


def condition2(t, x):
    y = wrap_angles_top(x)
    flag, rad = check_if_state_in_roa(S, rho, y)
    if flag:
        return flag
    return flag


######################################################

dynamics_func = double_pendulum_dynamics_func(
    simulator=None,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=2,
    max_velocity=20.0,
    torque_limit=torque_limit,
)

controller1 = SACController(
    model_path="../../../data/policies/design_C.0/model_3.0/acrobot/SAC/sac_model.zip",
    dynamics_func=dynamics_func,
    dt=dt,
)

controller2 = LQRController(model_pars=mpar)
controller2.set_goal(goal)
controller2.set_cost_matrices(Q=Q, R=R)
controller2.set_parameters(failure_value=0.0, cost_to_go_cut=15)

controller = CombinedController(
    controller1=controller1,
    controller2=controller2,
    condition1=condition1,
    condition2=condition2,
    compute_both=False,
)
controller.init()
