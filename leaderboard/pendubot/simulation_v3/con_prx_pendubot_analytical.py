import os
import numpy as np

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.prx.prx_pendubot_analytical import PrxPendubotAnalyticalController


name = "prx_pendubot_analytical"
leaderboard_config = {
    "csv_path": name + "/sim_swingup.csv",
    "name": name,
    "simple_name": "prx_analytical",
    "short_description": "LQR with extra exploration.",
    "readme_path": f"readmes/{name}.md",
    "username": "garygra",
}

traj_model = "model_1.1"

traj_filename="pendubot_dp_250314_074722_877141370_lqr_traj.txt"
controller = PrxPendubotAnalyticalController(traj_filename)

controller.init()
