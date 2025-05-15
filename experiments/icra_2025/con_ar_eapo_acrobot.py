import os
from pathlib import Path
from double_pendulum.controller.AR_EAPO import NumbaJITController
from double_pendulum.filter.lowpass import lowpass_filter

from parameters import (
    design,
    model,
)

name = "ar_eapo"
robot = "acrobot"

leaderboard_config = {
    "csv_path": "trajectory.csv",
    "name": name,
    "simple_name": "AR-EAPO",
    "short_description": "Policy trained with average reward maximum entropy RL",
    "readme_path": f"readmes/{name}.md",
    "username": "rnilva",
}

model_path = Path(
    f"../../data/policies/{design}/{model}/{robot}/V3_AR_EAPO/model.zip"
)

dt = 0.002
max_velocity = 20.0
t_limit_value = 0.5
is_numba_clip_velocity = False

controller = NumbaJITController(
    model_path=model_path,
    robot=robot,
    max_torque=6.0,
    max_velocity=max_velocity,
    torque_compensation=t_limit_value,
    clip_velocity=is_numba_clip_velocity,
)

# filter
filter = lowpass_filter(
    alpha=[1.0, 1.0, 0.2, 0.2], x0=[0.0, 0.0, 0.0, 0.0], filt_velocity_cut=0.1
)
controller.set_filter(filter)

controller.init()
