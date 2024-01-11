import os
import importlib
import argparse
import pandas

from double_pendulum.analysis.leaderboard import leaderboard_scores

from sim_parameters import mpar
from sim_controller import simulate_controller


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir",
    dest="data_dir",
    help="Directory for saving data. Existing data will be kept.",
    default="data",
    required=False,
)
parser.add_argument(
    "--save_to",
    dest="save_to",
    help="Path for saving the leaderbaord csv file.",
    default="leaderboard.csv",
    required=False,
)
parser.add_argument(
    "--force-recompute",
    dest="recompute",
    help="Whether to force the recomputation of the leaderboard even without new data.",
    default=False,
    required=False,
    type=int,
)
parser.add_argument(
    "--link-base",
    dest="link",
    help="base-link for hosting data. Not needed for local execution",
    default="",
    required=False,
)


data_dir = parser.parse_args().data_dir
save_to = parser.parse_args().save_to
recompute_leaderboard = bool(parser.parse_args().recompute)
link_base = parser.parse_args().link

if not os.path.exists(save_to):
    recompute_leaderboard = True

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

existing_list = os.listdir(data_dir)
for con in existing_list:
    if not os.path.exists(os.path.join(data_dir, con, "sim_swingup.csv")):
        existing_list.remove(con)

for file in os.listdir("."):
    if file[:4] == "con_":
        if file[4:-3] in existing_list:
            print(f"Simulation data for {file} found.")
        else:
            print(f"Simulating new controller {file}")

            controller_arg = file[:-3]
            controller_name = controller_arg[4:]

            save_dir = os.path.join(data_dir, f"{controller_name}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            imp = importlib.import_module(controller_arg)

            controller = imp.controller

            simulate_controller(controller, save_dir, controller_name)
            recompute_leaderboard = True

if recompute_leaderboard:
    src_dir = "."
    data_paths = {}

    for f in os.listdir(src_dir):
        if f[:4] == "con_":
            mod = importlib.import_module(f[:-3])
            if hasattr(mod, "leaderboard_config"):
                if os.path.exists(
                    os.path.join(data_dir, mod.leaderboard_config["csv_path"])
                ):
                    print(
                        f"Found leaderboard_config and data for {mod.leaderboard_config['name']}"
                    )
                    conf = mod.leaderboard_config
                    conf["csv_path"] = os.path.join(
                        data_dir, mod.leaderboard_config["csv_path"]
                    )
                    data_paths[mod.leaderboard_config["name"]] = conf

    leaderboard_scores(
        data_paths=data_paths,
        save_to=save_to,
        mpar=mpar,
        # weights={"swingup_time": 0.5, "max_tau": 0.1, "energy": 0.0, "integ_tau": 0.4, "tau_cost": 0.0, "tau_smoothness": 0.0},
        weights={
            "swingup_time": 0.2,
            "max_tau": 0.1,
            "energy": 0.1,
            "integ_tau": 0.1,
            "tau_cost": 0.1,
            "tau_smoothness": 0.2,
            "velocity_cost": 0.2,
        },
        normalize={
            "swingup_time": 10.0,
            "max_tau": 6.0,
            "energy": 100.0,
            "integ_tau": 60.0,
            "tau_cost": 360.0,
            "tau_smoothness": 12.0,
            "velocity_cost": 1000,
        },
        link_base=link_base,
    )

    df = pandas.read_csv(save_to)
    print(df.columns.values)
    df = df.drop(df.columns[1], axis=1)
    df = df.drop(df.columns[1], axis=1)
    print(df.sort_values(by=["RealAI Score"], ascending=False).to_markdown(index=False))
