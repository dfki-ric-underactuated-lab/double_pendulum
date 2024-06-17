import os
import numpy as np

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.tvlqr.tvlqr_controller import TVLQRController

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

name = "ilqr_tvlqr"
leaderboard_config = {
    "csv_path": name + "/sim_swingup.csv",
    "name": name,
    "simple_name": "TVLQR",
    "short_description": "Stabilization of iLQR trajectory with time-varying LQR.",
    "readme_path": f"readmes/{name}.md",
    "username": "fwiebe",
}

traj_model = "model_3.1"

## trajectory parameters
csv_path = os.path.join(
    "../../../data/trajectories", design, traj_model, robot, "ilqr_2/trajectory.csv"
)


## controller parameters
Q = np.diag([0.64, 0.56, 0.13, 0.037])
R = np.eye(2) * 0.82
Qf = 100 * np.diag([0.64, 0.56, 0.13, 0.037])

controller = TVLQRController(model_pars=mpar, csv_path=csv_path, torque_limit=mpar.tl)

controller.set_cost_parameters(Q=Q, R=R, Qf=Qf)

controller.init()
