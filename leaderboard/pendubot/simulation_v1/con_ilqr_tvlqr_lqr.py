import os
import numpy as np

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.tvlqr.tvlqr_controller import TVLQRController
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.utils.wrap_angles import wrap_angles_top

from sim_parameters import mpar, dt, t_final, t0, x0, goal, integrator, design, model, robot

name = "ilqr_tvlqr_lqr"
leaderboard_config = {"csv_path": name + "/sim_swingup.csv",
                      "name": name,
                      "simple_name": "TVLQR",
                      "short_description": "Stabilization of iLQR trajectory with time-varying LQR.",
                      "readme_path": f"readmes/{name}.md",
                      "username": "fwiebe"}

traj_model = "model_2.1"

## trajectory parameters
csv_path = os.path.join("../../../data/trajectories", design, traj_model, robot, "ilqr_1/trajectory.csv")


## controller parameters
Q = np.diag([0.64, 0.56, 0.13, 0.037])
R = np.eye(2) * 0.82
# Qf = np.diag([0.64, 0.56, 0.13, 0.037])

# LQR controller
Q_lqr = np.diag((0.97, 0.93, 0.39, 0.26))
R_lqr = np.diag((0.11, 0.11))

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

controller2 = LQRController(model_pars=mpar)
controller2.set_goal(goal)
controller2.set_cost_matrices(Q=Q_lqr, R=R_lqr)
controller2.set_parameters(failure_value=0.0,
                          cost_to_go_cut=1000)
controller2.init()

Qf = controller2.S

controller1 = TVLQRController(model_pars=mpar, csv_path=csv_path, torque_limit=mpar.tl)
controller1.set_cost_parameters(Q=Q, R=R, Qf=Qf)

controller = CombinedController(
        controller1=controller1,
        controller2=controller2,
        condition1=condition1,
        condition2=condition2)

controller.init()