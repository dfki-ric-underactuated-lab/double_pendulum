import os
import numpy as np

from double_pendulum.controller.trajectory_following.trajectory_controller import (
    TrajectoryController,
)
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.utils.wrap_angles import wrap_angles_top

from sim_parameters import mpar, goal, design, robot

name = "ilqr_riccati_lqr"
leaderboard_config = {
    "csv_path": name + "/sim_swingup.csv",
    "name": name,
    "simple_name": "iLQR Riccati Gains",
    "short_description": "Stabilization of iLQR trajectory with Riccati gains. Top stabilization with LQR.",
    "readme_path": f"readmes/{name}.md",
    "username": "fwiebe",
}

traj_model = "model_1.1"

torque_limit = [0.0, 6.0]

# init trajectory
init_csv_path = os.path.join(
    "../../../data/trajectories", design, traj_model, robot, "ilqr_1/trajectory.csv"
)

# LQR controller
Q = np.diag((0.97, 0.93, 0.39, 0.26))
R = np.diag((0.11, 0.11))


def condition1(t, x):
    return False


def condition2(t, x):
    goal = [np.pi, 0.0, 0.0, 0.0]
    eps = [0.05, 0.05, 0.1, 0.1]

    y = wrap_angles_top(x)

    delta = np.abs(np.subtract(y, goal))
    max_diff = np.max(np.subtract(delta, eps))
    if max_diff > 0.0:
        return False
    else:
        return True


# construct simulation objects
controller1 = TrajectoryController(
    csv_path=init_csv_path, torque_limit=torque_limit, kK_stabilization=True
)
controller2 = LQRController(model_pars=mpar)
controller2.set_goal(goal)
controller2.set_cost_matrices(Q=Q, R=R)
controller2.set_parameters(failure_value=0.0, cost_to_go_cut=1000)
controller = CombinedController(
    controller1=controller1,
    controller2=controller2,
    condition1=condition1,
    condition2=condition2,
)
controller.init()
