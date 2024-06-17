import os
import argparse
import importlib
import numpy as np
import pandas

from double_pendulum.analysis.leaderboard import leaderboard_scores

from sim_parameters import mpar, dt, t_final, t0, x0, goal, integrator
from sim_controller import simulate_controller

parser = argparse.ArgumentParser()
parser.add_argument("controller", help="name of the controller to simulate")
controller_arg = parser.parse_args().controller
if controller_arg[-3:] == ".py":
    controller_arg = controller_arg[:-3]

controller_name = controller_arg[4:]
imp = importlib.import_module(controller_arg)
controller = imp.controller

data_dir = "data"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
save_dir = f"{data_dir}/{controller_name}"
save_to = os.path.join(save_dir, "leaderboard_entry.csv")

if not os.path.exists(os.path.join(save_dir, "sim_swingup.csv")):
    print("Did not find simulation data. Simulating...")
    simulate_controller(controller, save_dir)
    print("Done")
else:
    print("Found simulation data.")

conf = imp.leaderboard_config

conf["csv_path"] = os.path.join(data_dir, imp.leaderboard_config["csv_path"])
data_paths = {}
data_paths[imp.leaderboard_config["name"]] = conf

leaderboard_scores(
    data_paths=data_paths,
    save_to=save_to,
    mpar=mpar,
    # weights={"swingup_time": 0.5, "max_tau": 0.1, "energy": 0.0, "integ_tau": 0.4, "tau_cost": 0.0, "tau_smoothness": 0.0},
    weights={
        "swingup_time": 1.0,
        "max_tau": 0.0,
        "energy": 1.0,
        "integ_tau": 0.0,
        "tau_cost": 1.0,
        "tau_smoothness": 1.0,
        "velocity_cost": 1.0,
    },
    normalize={
        "swingup_time": 20.0,
        "max_tau": 1.0,  # not used
        "energy": 60.0,
        "integ_tau": 1.0,  # not used
        "tau_cost": 20.0,
        "tau_smoothness": 0.1,
        "velocity_cost": 400,
    },
    link_base="",
    score_version="v2",
)
df = pandas.read_csv(save_to)
print(df.sort_values(by=["RealAI Score"], ascending=False).to_markdown(index=False))
# print(
#     pandas.read_csv(save_to)
#     .sort_values(by=["RealAI Score"], ascending=False)
#     .to_markdown(index=False)
# )
