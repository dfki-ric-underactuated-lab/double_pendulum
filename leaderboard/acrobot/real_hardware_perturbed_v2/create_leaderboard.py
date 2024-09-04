import os
import argparse
import pandas

from double_pendulum.analysis.leaderboard import leaderboard_scores

from exp_parameters import mpar


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir",
    dest="data_dir",
    help="Directory for loading data. Existing data will be kept.",
    default="src_data",
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
    if not os.path.exists(os.path.join(data_dir, con, "data_paths.csv")):
        existing_list.remove(con)

if recompute_leaderboard:
    src_dir = "."
    data_paths = {}

    for con_dir in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, con_dir)):
            paths = []
            for exp_dir in os.listdir(os.path.join(data_dir, con_dir)):
                if exp_dir[:10] == "experiment":
                    paths.append(
                        os.path.join(data_dir, con_dir, exp_dir, "trajectory.csv")
                    )
            with open(os.path.join(data_dir, con_dir, "name.txt"), "r") as file:
                name = file.read().replace("\n", "")
            with open(os.path.join(data_dir, con_dir, "username.txt"), "r") as file:
                username = file.read().replace("\n", "")
            with open(
                os.path.join(data_dir, con_dir, "short_description.txt"), "r"
            ) as file:
                short_description = file.read().replace("\n", "")
            data_paths[name] = {}
            data_paths[name]["csv_path"] = paths
            data_paths[name]["name"] = name
            data_paths[name]["username"] = username
            data_paths[name]["short_description"] = short_description

    leaderboard_scores(
        data_paths=data_paths,
        save_to=save_to,
        mpar=mpar,
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
            "tau_smoothness": 1.0,
            "velocity_cost": 400.0,
        },
        link_base=link_base,
        simulation=False,
        score_version="v2",
    )
    df = pandas.read_csv(save_to)
    df = df.drop(df.columns[1], axis=1)
    print(
        df.sort_values(by=["Average RealAI Score"], ascending=False).to_markdown(
            index=False
        )
    )
