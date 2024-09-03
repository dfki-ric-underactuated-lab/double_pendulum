import os
import sys

from double_pendulum.controller.history_sac import HistorySACController

name = "history_sac"
leaderboard_config = {
    "csv_path": name + "/sim_swingup.csv",
    "name": name,
    "simple_name": "History SAC",
    "short_description": "SAC using custom model architecture to encode system dynamics.",
    "readme_path": f"readmes/{name}.md",
    "username": "tfaust",
}

controller = HistorySACController("pendubot", model_path="../../../data/policies/design_C.1/model_1.1/pendubot/history_sac/pendubot")
controller.init()
