import os
import numpy as np

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.prx.prx_acrobot_analytical import PrxAcrobotAnalyticalController


name = "prx_acrobot_analytical"
leaderboard_config = {
    "csv_path": name + "/sim_swingup.csv",
    "name": name,
    "simple_name": "prx_analytical",
    "short_description": "LQR with extra exploration.",
    "readme_path": f"readmes/{name}.md",
    "username": "garygra",
}

traj_model = "model_1.1"

traj_filename="lqr_gains.txt"
controller = PrxAcrobotAnalyticalController(traj_filename)

controller.init()
