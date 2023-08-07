import numpy as np
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.controller.DQN.DQN_controller import DQNController
from double_pendulum.controller.DQN.environment import get_environment
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.simulation.gym_env import (
    double_pendulum_dynamics_func,
)
from double_pendulum.utils.wrap_angles import wrap_angles_diff

from sim_parameters import mpar, goal, dt, integrator, robot

name = "dqn_lqr"
leaderboard_config = {
    "csv_path": name + "/sim_swingup.csv",
    "name": name,
    "simple_name": "DQN LQR",
    "short_description": "Swing-up with an RL Policy learned with DQN.",
    "readme_path": f"readmes/{name}.md",
    "username": "theovincent",
}

torque_limit = [5.0, 0.0]
Q_lqr = np.diag([1.92, 1.92, 0.3, 0.3])
R_lqr = np.diag([0.82, 0.82])

dynamics_func = double_pendulum_dynamics_func(
    simulator=None,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=2,
    max_velocity=20.0,
    torque_limit=torque_limit,
)

experiment_path = "../../../data/policies/design_A.0/model_2.0/pendubot/DQN/"
env = get_environment(9)
controller1 = DQNController(
    experiment_path,
    env.actions,
    dynamics_func=dynamics_func,
    dt=dt,
)

controller2 = LQRController(model_pars=mpar)
controller2.set_goal(goal)
controller2.set_cost_matrices(Q=Q_lqr, R=R_lqr)
controller2.set_parameters(failure_value=0.0, cost_to_go_cut=100)


def condition1(t, x):
    goal = [np.pi, 0.0, 0.0, 0.0]

    delta = wrap_angles_diff(np.subtract(x, goal))

    rho = 1.690673829091186575e-01

    S = np.array(
        [
            [
                7.857934201124567153e01,
                5.653751913776947191e01,
                1.789996146741196981e01,
                8.073612858295813766e00,
            ],
            [
                5.653751913776947191e01,
                4.362786774581156379e01,
                1.306971194928728330e01,
                6.041705515910111401e00,
            ],
            [
                1.789996146741196981e01,
                1.306971194928728330e01,
                4.125964000971944046e00,
                1.864116086667296113e00,
            ],
            [
                8.073612858295813766e00,
                6.041705515910111401e00,
                1.864116086667296113e00,
                8.609202333737846491e-01,
            ],
        ]
    )

    switch = False
    if np.einsum("i,ij,j", delta, S, delta) > 1.1 * rho:
        switch = True

    return switch


def condition2(t, x):
    goal = [np.pi, 0.0, 0.0, 0.0]

    rho = 1.690673829091186575e-01

    S = np.array(
        [
            [
                7.857934201124567153e01,
                5.653751913776947191e01,
                1.789996146741196981e01,
                8.073612858295813766e00,
            ],
            [
                5.653751913776947191e01,
                4.362786774581156379e01,
                1.306971194928728330e01,
                6.041705515910111401e00,
            ],
            [
                1.789996146741196981e01,
                1.306971194928728330e01,
                4.125964000971944046e00,
                1.864116086667296113e00,
            ],
            [
                8.073612858295813766e00,
                6.041705515910111401e00,
                1.864116086667296113e00,
                8.609202333737846491e-01,
            ],
        ]
    )

    delta = wrap_angles_diff(np.subtract(x, goal))

    switch = False
    if np.einsum("i,ij,j", delta, S, delta) < rho:
        switch = True

    return switch


controller = CombinedController(
    controller1=controller1,
    controller2=controller2,
    condition1=condition1,
    condition2=condition2,
    compute_both=False,
)
controller.init()
